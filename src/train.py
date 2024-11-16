import torch
import torch.nn as nn
import os
from collections import Counter
from src.utils.visualization import plot_top_misclassifications, plot_train_val_metrics, plot_misclassification_distribution
from collections import defaultdict
from tqdm import tqdm


def save_checkpoint(probe_name, probe, optimizer, epoch, val_loss, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'{probe_name}_best_checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': probe.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss
    }, checkpoint_path)
    print(f"Checkpoint saved for {probe_name} with validation loss: {val_loss:.4f}")

def load_best_checkpoint(probe_name, probe, optimizer, checkpoint_dir='checkpoints'):
    checkpoint_path = os.path.join(checkpoint_dir, f'{probe_name}_best_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        probe.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        print(f"Best checkpoint loaded for {probe_name}, epoch {epoch}, validation loss: {val_loss:.4f}")
    else:
        print(f"No checkpoint found for {probe_name}, skipping load.")
        

intermediate_outputs = {}
def register_hooks(model, layer_names):
    for layer_name in layer_names:
        layer = dict([*model.named_modules()])[layer_name]
        layer.register_forward_hook(lambda m, i, o, name=layer_name: intermediate_outputs.update({name: o}))

def train_probes(model: nn.Module, processor, probes, train_loader, val_loader, optimizers, criterion, epochs, device, checkpoint_dir='checkpoints', log_interval=10, patience=3):
    model.eval()  # CLIP model is frozen (not trained)
    best_val_loss = {probe_name: float('inf') for probe_name in probes}  # Initialize best validation loss for each probe
    early_stopping_counter = {probe_name: 0 for probe_name in probes}  # Track patience for early stopping

    register_hooks(model, list(probes.keys())[1:])
    
    # Track losses and accuracies
    train_losses = defaultdict(list)
    val_losses = defaultdict(list)
    train_accuracies = defaultdict(list)
    val_accuracies = defaultdict(list)

    for epoch in range(epochs):
        running_loss = {probe_name: 0.0 for probe_name in probes}
        running_corrects = {probe_name: 0 for probe_name in probes}  # TP
        running_false_positives = {probe_name: 0 for probe_name in probes}  # FP
        running_false_negatives = {probe_name: 0 for probe_name in probes}  # FN

        for _, probe in probes.items():
            probe.train()

        # Training step with tqdm progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as pbar:
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                input_ids = inputs['input_ids'].squeeze(dim=1).to(device)
                attention_mask = inputs['attention_mask'].squeeze(dim=1).to(device)
                labels = labels.to(device)

                # Forward pass through CLIP model to extract embeddings (if necessary)
                intermediate_outputs.clear()
                with torch.no_grad():
                    text_embeds = model(input_ids=input_ids, attention_mask=attention_mask)[0].detach()

                # Train each probe
                for probe_name, probe in probes.items():
                    optimizer = optimizers[probe_name]

                    if probe_name == "token_probe":
                        outputs = probe(input_ids / processor.tokenizer.vocab_size)  # Directly train on tokens
                    elif probe_name == "embedding_probe":
                        outputs = probe(text_embeds)  # Train on embeddings or other outputs
                    elif probe_name in list(intermediate_outputs.keys()):
                        outputs = probe(intermediate_outputs[probe_name][0])
                    else:
                        raise ValueError(f"Unknown probe name: {probe_name}")

                    loss = criterion(outputs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    running_loss[probe_name] += loss.item()

                    # Compute accuracy
                    if isinstance(criterion, nn.BCEWithLogitsLoss):  # Multilabel
                        preds = (outputs > 0.5).float()
                        correct_predictions = (preds == labels) & (labels == 1)  # True Positives
                        false_positives = (preds == 1) & (labels == 0)  # Predicted 1 but actually 0
                        false_negatives = (preds == 0) & (labels == 1)  # Predicted 0 but actually 1

                        running_corrects[probe_name] += correct_predictions.sum().item() # TP
                        running_false_positives[probe_name] += false_positives.sum().item()
                        running_false_negatives[probe_name] += false_negatives.sum().item()

                    else:  # Single-label
                        preds = outputs.argmax(dim=1)
                        running_corrects[probe_name] += (preds == labels).sum().item()
                        running_false_positives[probe_name] += (preds != labels).sum().item()
                        running_false_negatives[probe_name] += (labels != preds).sum().item()

                # Update progress bar every `log_interval` batches
                if (batch_idx + 1) % log_interval == 0 or (batch_idx + 1) == len(train_loader):
                    pbar.update(log_interval)

        # Save average training loss and accuracy for the epoch
        for probe_name in probes:
            train_losses[probe_name].append(running_loss[probe_name] / len(train_loader))
            precision = running_corrects[probe_name] / (running_corrects[probe_name] + running_false_positives[probe_name])
            recall = running_corrects[probe_name] / (running_corrects[probe_name] + running_false_negatives[probe_name])
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            train_accuracies[probe_name].append(100.0 * f1) # F1
            print(f"{probe_name} - Train Loss: {train_losses[probe_name][-1]:.4f}, Train Accuracy: {train_accuracies[probe_name][-1]:.2f}%")

        # Validation step after each epoch
        val_results = evaluate_probes(model, processor, probes, val_loader, criterion, device)
        for probe_name, val_metrics in val_results.items():
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']

            # Save validation loss and accuracy
            val_losses[probe_name].append(val_loss)
            val_accuracies[probe_name].append(val_accuracy)

            # Early stopping logic
            if val_loss < best_val_loss[probe_name]:
                best_val_loss[probe_name] = val_loss
                early_stopping_counter[probe_name] = 0  # Reset counter
                save_checkpoint(probe_name, probes[probe_name], optimizers[probe_name], epoch + 1, val_loss, checkpoint_dir)
            else:
                early_stopping_counter[probe_name] += 1

            # Check if patience is exceeded
            if early_stopping_counter[probe_name] >= patience:
                print(f"{probe_name} - Early stopping triggered after {patience} epochs of no improvement.")
                probes[probe_name].eval()  # Switch probe to eval mode
                continue

        # Check if all probes have stopped
        if all(early_stopping_counter[probe_name] >= patience for probe_name in probes):
            print("Early stopping for all probes. Exiting training.")
            break

    # Plotting after all epochs
    plot_train_val_metrics(train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_dir)
    return


def reverse_label(label_str):
    # Helper function to reverse label components
    parts = label_str.split('_')
    return f"{'_'.join(parts[0:-2])}_{parts[-1]}_{parts[-2]}"
    

def swap_relation(label_str, new_relation):
    # Helper function to replace the relation while keeping nouns in order
    parts = label_str.split('_')
    return f"{new_relation}_{parts[-2]}_{parts[-1]}"

def swap_relation_and_nouns(label_str, new_relation):
    # Helper function to swap both relation and nouns
    parts = label_str.split('_')
    return f"{new_relation}_{parts[-1]}_{parts[-2]}"

def evaluate_probes(model: nn.Module, processor, probes, test_loader, criterion, device, plot_misclassifications=False, output_dir='output'):
    results = {}
    classified_labels = Counter() 
    misclassified_labels = Counter()  # Track misclassified labels
    reverse_misclassified_labels = Counter()  # Track reverse misclassified labels
    swapped_relation_labels = Counter()  # Track mispredictions where only the relation was swapped
    swapped_relation_and_nouns_labels = Counter()  # Track mispredictions where both relation and nouns were swapped
    idx_to_label = test_loader.dataset.dataset.idx_to_label

    for probe_name, probe in probes.items():
        probe.eval()

    running_loss = {probe_name: 0.0 for probe_name in probes}
    true_positives = {probe_name: 0 for probe_name in probes}
    false_positives = {probe_name: 0 for probe_name in probes}
    false_negatives = {probe_name: 0 for probe_name in probes}

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            input_ids = inputs['input_ids'].squeeze(dim=1).to(device)
            attention_mask = inputs['attention_mask'].squeeze(dim=1).to(device)
            labels = labels.to(device)      
                  
            text_embeds = model(input_ids=input_ids, attention_mask=attention_mask)[0].detach() # B X 77 X 512

            for probe_name, probe in probes.items():
                if probe_name == "token_probe":
                    outputs = probe(input_ids / processor.tokenizer.vocab_size)
                elif probe_name == "embedding_probe":
                    outputs = probe(text_embeds)
                elif probe_name in list(intermediate_outputs.keys()):
                    outputs = probe(intermediate_outputs[probe_name][0])
                else:
                    raise ValueError(f"Unknown probe name: {probe_name}")

                loss = criterion(outputs, labels)
                running_loss[probe_name] += loss.item()

                if isinstance(criterion, nn.BCEWithLogitsLoss):  # Multilabel
                    predicted = (outputs > 0.5).float()  # Threshold predictions at 0.5
                        
                    # True Positives (TP)
                    true_positive_mask = (predicted == 1) & (labels == 1)
                    true_positives[probe_name] += true_positive_mask.sum().item()

                    # False Positives (FP)
                    false_positive_mask = (predicted == 1) & (labels == 0)
                    false_positives[probe_name] += false_positive_mask.sum().item()

                    # False Negatives (FN)
                    false_negative_mask = (predicted == 0) & (labels == 1)
                    false_negatives[probe_name] += false_negative_mask.sum().item()
                else:
                    _, predicted = torch.max(outputs.data, 1)
                    # True Positives (TP): Correct predictions
                    true_positive_mask = (predicted == labels)
                    true_positives[probe_name] += true_positive_mask.sum().item()

                    # False Positives (FP): Predicted as positive but incorrect
                    false_positive_mask = (predicted != labels) & (predicted >= 0)  # Predicted something incorrect
                    false_positives[probe_name] += false_positive_mask.sum().item()

                    # False Negatives (FN): Ground truth is positive but missed by the prediction
                    false_negative_mask = (predicted != labels) & (labels >= 0)  # Ground truth missed
                    false_negatives[probe_name] += false_negative_mask.sum().item()


                # Track misclassified labels and reverse misclassifications
                if isinstance(criterion, nn.BCEWithLogitsLoss):  # Multilabel
                    for pred_vec, label_vec in zip(predicted, labels):
                        pred_indices = torch.where(pred_vec == 1)[0]  # Indices of active predictions
                        label_indices = torch.where(label_vec == 1)[0]  # Indices of true labels

                        # Handle true positive labels
                        for label_idx in label_indices:
                            true_label = idx_to_label[label_idx.item()]
                            if label_idx in pred_indices:
                                classified_labels[true_label] += 1  # Correctly classified
                            else:
                                misclassified_labels[true_label] += 1  # Missed by the prediction

                        # Handle false positive predictions
                        for pred_idx in pred_indices:
                            if pred_idx not in label_indices:  # Predicted but not in true labels
                                pred_label = idx_to_label[pred_idx.item()]
                                true_label = None  # No corresponding true label
                                misclassified_labels[pred_label] += 1

                else:
                    for pred, label in zip(predicted, labels):
                        if pred != label:
                            true_label = idx_to_label[label.item()]
                            pred_label = idx_to_label[pred.item()]
                            misclassified_labels[true_label] += 1

                            # Check if the misclassification is a reverse of the true label
                            reversed_label = reverse_label(true_label)
                            if reversed_label == pred_label:
                                reverse_misclassified_labels[true_label] += 1

                            # Check if only the relation was swapped
                            swapped_relation_pred = swap_relation(true_label, new_relation=pred_label.split('_')[0])
                            if swapped_relation_pred == pred_label:
                                swapped_relation_labels[true_label] += 1

                            # Check if both relation and nouns were swapped
                            swapped_relation_and_nouns_pred = swap_relation_and_nouns(true_label, new_relation=pred_label.split('_')[0])
                            if swapped_relation_and_nouns_pred == pred_label:
                                swapped_relation_and_nouns_labels[true_label] += 1
                        else:
                            classified_labels[idx_to_label[label.item()]] += 1

    for probe_name in probes:
        precision = true_positives[probe_name] / (true_positives[probe_name] + false_positives[probe_name])
        recall = true_positives[probe_name] / (true_positives[probe_name] + false_negatives[probe_name])
        accuracy = 100*(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0) # F1 score
        avg_loss = running_loss[probe_name] / len(test_loader)
        print(f"{probe_name} - Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        results[probe_name] = {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall}

        if plot_misclassifications:
            if labels.dim() == 2:
                plot_misclassification_distribution(probe_name, accuracy, misclassified_labels, output_dir)
            else:
                plot_top_misclassifications(probe_name, accuracy, misclassified_labels, reverse_misclassified_labels, swapped_relation_labels, swapped_relation_and_nouns_labels, output_dir)

    return results
