import torch
import torch.optim as optim
import torch.nn as nn
import pyrallis

from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, random_split
from src.data.dataset import OfflineRelationalDataset, collate_fn
from src.model import LinearProbe
from src.train import train_probe, evaluate_probe
from src.config import Config

@pyrallis.wrap()
def main(config: Config):
    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP model and processor based on config, move model to the selected device
    processor = CLIPProcessor.from_pretrained(config.model.clip_model_name)
    model = CLIPModel.from_pretrained(config.model.clip_model_name).to(device)
    model.eval()

    # Create dataset and pass the processor to tokenize the prompts
    full_dataset = OfflineRelationalDataset(
        dataset_file=config.dataset.dataset_file,
        processor=processor,  # Pass processor here
        prompt_column=config.dataset.prompt_column,
        label_column=config.dataset.label_column
    )

    # Split dataset into training and test sets (80% training, 20% testing)
    train_size = int(config.train.train_split * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Create dataloaders for both train and test sets
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model setup: Pass the dynamically retrieved number of classes to the probe
    num_classes = full_dataset.get_num_classes()
    linear_probe = LinearProbe(input_dim=config.model.input_dim, num_classes=num_classes).to(device)
    optimizer = optim.Adam(linear_probe.parameters(), lr=config.train.lr)
    criterion = nn.CrossEntropyLoss()

    # Train the model
    train_probe(model, linear_probe, train_loader, optimizer, criterion, config.train.epochs, device)

    # Evaluate the model
    evaluate_probe(model, linear_probe, test_loader, criterion, device)

if __name__ == "__main__":
    main()