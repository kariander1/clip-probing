import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, random_split

class OfflineRelationalDataset(Dataset):
    """
    Loads a relational dataset from an offline file (e.g., a CSV or JSON file).
    This class expects that the dataset has been pre-generated and saved.
    """
    def __init__(self, dataset_file, processor, prompt_column="prompt", label_column="label"):
        # Load dataset from a file using Hugging Face datasets library
        extension = dataset_file.split(".")[-1]
        if extension == "json":
            self.dataset = load_dataset('json', data_files=dataset_file)["train"]
        elif extension == "csv":
            self.dataset = load_dataset('csv', data_files=dataset_file)["train"]
        else:
            raise ValueError("Unsupported file format. Use either .json or .csv.")
        
        self.prompt_column = prompt_column
        self.label_column = label_column
        self.processor = processor  # Add processor for tokenization

        # Create a mapping from text labels to numeric categories
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.dataset[label_column])))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}  # Reverse mapping for convenience

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx][self.prompt_column]
        label = self.dataset[idx][self.label_column]
        label_idx = self.label_to_idx[label]  # Convert text label to categorical number

        # Use the CLIP processor to tokenize the prompt into input_ids
        inputs = self.processor(text=prompt, return_tensors="pt", padding="max_length")
        input_ids = inputs['input_ids'].squeeze()  # Get input IDs

        return input_ids, label_idx

    def get_num_classes(self):
        """Returns the number of unique classes in the dataset."""
        return len(self.label_to_idx)

    def get_classes(self):
        """Returns the list of unique classes in the dataset."""
        return list(self.label_to_idx.keys())
    
    def get_all_labels(self):
        """Returns a list of all labels in the dataset."""
        return [self.dataset[i][self.label_column] for i in range(len(self.dataset))]
class HuggingFaceDataset(Dataset):
    """
    Loads a dataset directly from Hugging Face datasets.
    """
    def __init__(self, hf_dataset_name, processor, prompt_column="text", label_column="label"):
        # Load the dataset from Hugging Face
        self.dataset = load_dataset(hf_dataset_name)["train"]

        self.prompt_column = prompt_column
        self.label_column = label_column
        self.processor = processor

        # Create a mapping from text labels to numeric categories
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(self.dataset[self.label_column])))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        prompt = self.dataset[idx][self.prompt_column]
        label = self.dataset[idx][self.label_column]
        label_idx = self.label_to_idx[label]

        # Tokenize the prompt using the processor
        inputs = self.processor(text=prompt, return_tensors="pt", padding="max_length")
        input_ids = inputs['input_ids'].squeeze()

        return input_ids, label_idx

    def get_num_classes(self):
        """Returns the number of unique classes in the dataset."""
        return len(self.label_to_idx)
    
# Step 4: Define a custom collate function to handle variable-length inputs
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # Pad the input_ids to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return input_ids_padded, labels

def split_dataset(config, full_dataset):
    
    logger.info(f"Loaded dataset with {len(full_dataset)} samples and {full_dataset.get_num_classes()} classes.")
    logger.info(f"Splitting dataset into train, validation, and test sets...")
    # Assuming labels are accessible in full_dataset and `get_labels()` gives label list
    labels = np.array(full_dataset.get_all_labels())

    # Step 1: Split into 80% train and 20% remaining (val + test)
    train_split = config.train.train_split  # 0.8 for train
    remaining_split = 1 - train_split  # 0.2 for validation and test

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=train_split, test_size=remaining_split, random_state=config.train.seed)
    train_indices, remaining_indices = next(splitter.split(np.zeros(len(labels)), labels))

    # Step 2: Split the remaining 20% into 10% validation and 10% test
    val_test_splitter = StratifiedShuffleSplit(n_splits=1, train_size=config.train.val_split / remaining_split, random_state=config.train.seed)
    val_indices, test_indices = next(val_test_splitter.split(np.zeros(len(remaining_indices)), labels[remaining_indices]))

    remaining_indices = remaining_indices.tolist()
    # Map remaining indices to original dataset
    val_indices = [remaining_indices[i] for i in val_indices]
    test_indices = [remaining_indices[i] for i in test_indices]

    # Create Subsets for each split
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices.tolist())
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)


    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader