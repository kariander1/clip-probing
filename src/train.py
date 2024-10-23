import torch
import torch.nn as nn

def train_probe(model: nn.Module, linear_probe: nn.Module, train_loader, optimizer, criterion, epochs, device):
    model.eval()  # CLIP model is frozen (not trained)
    linear_probe.train()  # Train the probe
    for epoch in range(epochs):
        running_loss = 0.0
        for input_ids, labels in train_loader:
            input_ids = input_ids.to(device)  # Move inputs to device
            labels = labels.to(device)  # Move labels to device

            with torch.no_grad():
                # Extract CLIP text embeddings (frozen)
                text_embeds = model.get_text_features(input_ids=input_ids).detach()

            # Train the linear probe
            outputs = linear_probe(text_embeds)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

def evaluate_probe(model: nn.Module, linear_probe: nn.Module, test_loader, criterion, device):
    model.eval()  # CLIP model stays frozen
    linear_probe.eval()  # Set probe to evaluation mode

    total = 0
    correct = 0
    running_loss = 0.0

    with torch.no_grad():
        for input_ids, labels in test_loader:
            input_ids = input_ids.to(device)
            labels = labels.to(device)

            # Extract CLIP text embeddings
            text_embeds = model.get_text_features(input_ids=input_ids).detach()

            # Get the predictions
            outputs = linear_probe(text_embeds)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(test_loader)
    print(f"Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy, avg_loss
