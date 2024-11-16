import torch
import torch.optim as optim
import torch.nn as nn
import pyrallis
import os
import random
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, random_split
from src.data.dataset import SingleLabelRelationalDataset, HuggingFaceDataset
from src.model import LinearProbe
from src.train import train_probes, evaluate_probes
from src.config import Config
from src.train import load_best_checkpoint

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
@pyrallis.wrap()
def main(config: Config):
    set_seed(config.train.seed)
    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load CLIP model and processor based on config, move model to the selected device
    processor = CLIPProcessor.from_pretrained(config.model.clip_model_name)
    model = CLIPModel.from_pretrained(config.model.clip_model_name).to(device)
    model.eval()

    # Create dataset and pass the processor to tokenize the prompts
    if os.path.exists(config.dataset.dataset_file):
        full_dataset = SingleLabelRelationalDataset(
            dataset_file=config.dataset.dataset_file,
            processor=processor,  # Pass processor here
            prompt_column=config.dataset.prompt_column,
            label_column=config.dataset.label_column
        )
    else:
        full_dataset = HuggingFaceDataset(
            hf_dataset_name=config.dataset.dataset_file,
            processor=processor,  # Pass processor here
            prompt_column=config.dataset.prompt_column,
            label_column=config.dataset.label_column
        )

    # Assuming labels are accessible in full_dataset and `get_labels()` gives label list
    labels = np.array(full_dataset.get_all_labels())

    # Step 1: Split into 80% train and 20% remaining (val + test)
    train_split = config.train.train_split  # 0.8 for train
    remaining_split = 1 - train_split  # 0.2 for validation and test

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_split, test_size=remaining_split, random_state=config.train.seed)
    _, remaining_indices = next(splitter.split(np.zeros(len(labels)), labels))

    # Step 2: Split the remaining 20% into 10% validation and 10% test
    val_test_splitter = StratifiedShuffleSplit(n_splits=1, train_size=config.train.val_split / remaining_split, random_state=config.train.seed)
    _, test_indices = next(val_test_splitter.split(np.zeros(len(remaining_indices)), labels[remaining_indices]))

    remaining_indices = remaining_indices.tolist()
    # Map remaining indices to original dataset
    test_indices = [remaining_indices[i] for i in test_indices]

    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Model setup: Pass the dynamically retrieved number of classes to the probe
    num_classes = full_dataset.get_num_classes()
        # Initialize multiple probes
    probes = {
        "embedding_probe": LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "token_probe": LinearProbe(processor.tokenizer.model_max_length, num_classes).to(device)
        # Add more probes as needed, for example:
        # "intermediate_layer_probe": create_linear_probe("intermediate_layer_probe", intermediate_layer_dim, num_classes, device)
    }

    # Initialize optimizers for each probe
    optim_name_to_optimizer = {
        'adam': optim.Adam,
    }
    optimizer = optim_name_to_optimizer[config.train.optimizer]
    optimizers = {
        "embedding_probe": optimizer(probes["embedding_probe"].parameters(), lr=config.train.lr),
    }

    criterion = nn.CrossEntropyLoss()
    dataset_file_name = os.path.basename(config.dataset.dataset_file)
    # After training, load the best checkpoints for each probe
    for probe_name in probes:
        load_best_checkpoint(probe_name, probes[probe_name], optimizers[probe_name], os.path.join('checkpoints', dataset_file_name))
        
        # Evaluate the model
    evaluate_probes(model, processor, probes, test_loader, criterion, device, plot_misclassifications=True)

if __name__ == "__main__":
    main()