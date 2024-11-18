import torch
import torch.nn as nn
import os
from collections import Counter
from collections import defaultdict
from tqdm import tqdm
from src.utils.visualization import plot_top_misclassifications, plot_train_val_metrics, plot_misclassification_distribution


# Save and load checkpoints
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
        checkpoint = torch.load(checkpoint_path)
        probe.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        val_loss = checkpoint['val_loss']
        print(f"Best checkpoint loaded for {probe_name}, epoch {epoch}, validation loss: {val_loss:.4f}")
    else:
        print(f"No checkpoint found for {probe_name}, skipping load.")


# Utility functions
def reverse_label(label_str):
    parts = label_str.split('_')
    return f"{'_'.join(parts[0:-2])}_{parts[-1]}_{parts[-2]}"


def swap_relation(label_str, new_relation):
    parts = label_str.split('_')
    return f"{new_relation}_{parts[-2]}_{parts[-1]}"


def swap_relation_and_nouns(label_str, new_relation):
    parts = label_str.split('_')
    return f"{new_relation}_{parts[-1]}_{parts[-2]}"


# Shared forward pass logic
def forward_pass(model, processor, probe_name, probe, inputs, intermediate_outputs, device):
    input_ids = inputs['input_ids'].squeeze(dim=1).to(device)
    attention_mask = inputs['attention_mask'].squeeze(dim=1).to(device)

    with torch.no_grad():
        text_embeds = model(input_ids=input_ids, attention_mask=attention_mask)[0].detach()

    if probe_name == "token_probe":
        outputs = probe(input_ids / processor.tokenizer.vocab_size)
    elif probe_name == "embedding_probe":
        outputs = probe(text_embeds)
    elif probe_name in intermediate_outputs:
        outputs = probe(intermediate_outputs[probe_name][0])
    else:
        raise ValueError(f"Unknown probe name: {probe_name}")

    return outputs


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

# Shared metrics calculation
def calc_metrics(criterion, outputs, labels, running_metrics, idx_to_label=None, misclassified_labels=None, reverse_misclassified_labels=None, swapped_relation_labels=None, swapped_relation_and_nouns_labels=None, has_super_labels=False):
    loss = criterion(outputs, labels)
    if isinstance(criterion, nn.BCEWithLogitsLoss):  # Multilabel
        preds = (outputs > 0.5).float()
        running_metrics['true_positives'] += ((preds == 1) & (labels == 1)).sum().item()
        running_metrics['false_positives'] += ((preds == 1) & (labels == 0)).sum().item()
        running_metrics['false_negatives'] += ((preds == 0) & (labels == 1)).sum().item()
        if misclassified_labels is not None:
            for pred_vec, label_vec in zip(preds, labels):
                pred_indices = torch.where(pred_vec == 1)[0]  # Indices of active predictions
                label_indices = torch.where(label_vec == 1)[0]  # Indices of true labels

                # Handle true positive labels
                for label_idx in label_indices:
                    true_label = idx_to_label[label_idx.item()]
                    if label_idx not in pred_indices:
                        misclassified_labels[true_label] += 1  # Missed by the prediction

                # Handle false positive predictions
                for pred_idx in pred_indices:
                    if pred_idx not in label_indices:  # Predicted but not in true labels
                        pred_label = idx_to_label[pred_idx.item()]
                        true_label = None  # No corresponding true label
                        misclassified_labels[pred_label] += 1
    else:  # Single-label
        preds = outputs.argmax(dim=1)
        running_metrics['true_positives'] += (preds == labels).sum().item()
        running_metrics['false_positives'] += (preds != labels).sum().item()
        running_metrics['false_negatives'] += (labels != preds).sum().item()
        if misclassified_labels is not None:
            for pred, label in zip(preds, labels):
                if pred != label:
                    true_label = idx_to_label[label.item()]
                    pred_label = idx_to_label[pred.item()]
                    misclassified_labels[true_label] += 1

                    if has_super_labels:
                        # Check if the misclassification is a reverse of the true label
                        reversed_label = reverse_label(true_label)
                        if reversed_label == pred_label:
                            reverse_misclassified_labels[true_label] += 1
                            continue
                        # Check if only the relation was swapped
                        swapped_relation_pred = swap_relation(true_label, new_relation=pred_label.split('_')[0])
                        if swapped_relation_pred == pred_label:
                            swapped_relation_labels[true_label] += 1
                            continue

                        # Check if both relation and nouns were swapped
                        swapped_relation_and_nouns_pred = swap_relation_and_nouns(true_label, new_relation=pred_label.split('_')[0])
                        if swapped_relation_and_nouns_pred == pred_label:
                            swapped_relation_and_nouns_labels[true_label] += 1
                            continue

    return loss

# Training function
def train_probes(model, processor, probes, train_loader, val_loader, optimizers, criterion, epochs, device, checkpoint_dir='checkpoints', log_interval=10, patience=3, has_super_labels=False):
    model.eval()  # Freeze CLIP model
    best_val_loss = {probe_name: float('inf') for probe_name in probes}
    early_stopping_counter = {probe_name: 0 for probe_name in probes}

    train_losses = defaultdict(list)
    val_losses = defaultdict(list)
    train_accuracies = defaultdict(list)
    val_accuracies = defaultdict(list)

    for epoch in range(epochs):
        running_metrics = {probe_name: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0} for probe_name in probes}
        running_loss = {probe_name: 0.0 for probe_name in probes}

        for probe in probes.values():
            probe.train()

        for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            labels = labels.to(device)

            for probe_name, probe in probes.items():
                optimizer = optimizers[probe_name]
                optimizer.zero_grad()

                outputs = forward_pass(model, processor, probe_name, probe, inputs, {}, device)
                loss = calc_metrics(criterion, outputs, labels, running_metrics[probe_name])
                running_loss[probe_name] += loss.item()

                loss.backward()
                optimizer.step()

        # Calculate training metrics after epoch
        for probe_name in probes:
            metrics = running_metrics[probe_name]
            precision = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_positives"] + 1e-9)
            recall = metrics["true_positives"] / (metrics["true_positives"] + metrics["false_negatives"] + 1e-9)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
            accuracy = 100.0 * f1

            train_accuracies[probe_name].append(accuracy)
            train_losses[probe_name].append(running_loss[probe_name] / len(train_loader))

        # Validation step
        val_results = evaluate_probes(model, processor, probes, val_loader, criterion, device, plot_misclassifications=False, has_super_labels=has_super_labels)
        for probe_name, val_metrics in val_results.items():
            val_loss = val_metrics['loss']
            val_accuracy = val_metrics['accuracy']
            val_losses[probe_name].append(val_loss)
            val_accuracies[probe_name].append(val_accuracy)

            if val_loss < best_val_loss[probe_name]:
                best_val_loss[probe_name] = val_loss
                early_stopping_counter[probe_name] = 0
                save_checkpoint(probe_name, probes[probe_name], optimizers[probe_name], epoch + 1, val_loss, checkpoint_dir)
            else:
                early_stopping_counter[probe_name] += 1

            if early_stopping_counter[probe_name] >= patience:
                print(f"{probe_name} - Early stopping triggered.")
                probes[probe_name].eval()

            print(f"{probe_name} - Epoch {epoch + 1}/{epochs}, Train Loss: {train_losses[probe_name][-1]:.4f}, Train Acc: {train_accuracies[probe_name][-1]:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
        if all(early_stopping_counter[probe_name] >= patience for probe_name in probes):
            print("Early stopping for all probes. Exiting training.")
            break

    plot_train_val_metrics(train_losses, val_losses, train_accuracies, val_accuracies, checkpoint_dir)
    return


# Evaluation function
def evaluate_probes(model, processor, probes, test_loader, criterion, device, plot_misclassifications=False, output_dir='output', has_super_labels=False):
    results = {}
    misclassified_labels = Counter()
    reverse_misclassified_labels = Counter()  # Track reverse misclassified labels
    swapped_relation_labels = Counter()  # Track mispredictions where only the relation was swapped
    swapped_relation_and_nouns_labels = Counter()  # Track mispredictions where both relation and nouns were swapped
    idx_to_label = test_loader.dataset.dataset.idx_to_label

    for probe_name, probe in probes.items():
        probe.eval()

    running_metrics = {probe_name: {'true_positives': 0, 'false_positives': 0, 'false_negatives': 0} for probe_name in probes}
    running_loss = {probe_name: 0.0 for probe_name in probes}

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader):
            labels = labels.to(device)

            for probe_name, probe in probes.items():
                outputs = forward_pass(model, processor, probe_name, probe, inputs, {}, device)
                # decoded_prompt = test_loader.dataset.dataset.processor.decode(inputs['input_ids'][0][0], skip_special_tokens=True)
                loss = calc_metrics(
                    criterion,
                    outputs,
                    labels,
                    running_metrics[probe_name],
                    idx_to_label,
                    misclassified_labels,
                    reverse_misclassified_labels,
                    swapped_relation_labels,
                    swapped_relation_and_nouns_labels,
                    has_super_labels=has_super_labels
                    )
                running_loss[probe_name] += loss.item()

    for probe_name in probes:
        metrics = running_metrics[probe_name]
        precision = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_positives'])
        recall = metrics['true_positives'] / (metrics['true_positives'] + metrics['false_negatives'])
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = 100.0 * f1
        avg_loss = running_loss[probe_name] / len(test_loader)

        results[probe_name] = {'loss': avg_loss, 'accuracy': accuracy, 'precision': precision, 'recall': recall}

        if plot_misclassifications:
            if has_super_labels:
                plot_top_misclassifications(probe_name, accuracy, misclassified_labels, reverse_misclassified_labels, swapped_relation_labels, swapped_relation_and_nouns_labels, output_dir)
            else:
                plot_misclassification_distribution(probe_name, accuracy, misclassified_labels, output_dir)

    return results
