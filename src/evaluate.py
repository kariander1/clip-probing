import numpy as np
import torch

def evaluate_probe(model, linear_probe, test_loader):
    model.eval()
    linear_probe.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for input_ids, labels in test_loader:
            text_embeds = model.get_text_features(input_ids=input_ids).detach()
            outputs = linear_probe(text_embeds)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")
