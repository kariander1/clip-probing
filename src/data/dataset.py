import os
import torch
import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from datasets import load_dataset
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

class MultiLabelRelationalDataset(Dataset):
    """
    Handles a multi-label relational dataset where each sample has a list of labels.
    Converts labels into multi-hot vectors.
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

        # Create a mapping from text labels to numeric indices
        all_labels = [label for labels in self.dataset[label_column] for label in labels]
        unique_labels = sorted(set(all_labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}  # Reverse mapping for convenience

    def __len__(self):
        return len(self.dataset)

    def _multi_hot_to_labels(self, multi_hot):
        """Converts a single multi-hot vector into a list of labels."""
        indices = np.where(multi_hot > 0)[0]  # Find indices with a value > 0
        labels = [self.idx_to_label[idx] for idx in indices]
        return labels
    
    def convert_to_labels(self, multi_hot_vectors):
        """
        Converts multi-hot vectors into lists of labels.

        Args:
            multi_hot_vectors: A NumPy array or PyTorch tensor of multi-hot vectors.
                Can be 1D (single sample) or 2D (batch).

        Returns:
            If input is 1D, returns a list of labels.
            If input is 2D, returns a list of lists of labels.
        """
        if isinstance(multi_hot_vectors, torch.Tensor):
            multi_hot_vectors = multi_hot_vectors.numpy()  # Convert to NumPy array if needed

        # Check if input is 1D or 2D
        if multi_hot_vectors.ndim == 1:
            # Single multi-hot vector
            return self._multi_hot_to_labels(multi_hot_vectors)
        elif multi_hot_vectors.ndim == 2:
            # Batch of multi-hot vectors
            return [self._multi_hot_to_labels(multi_hot) for multi_hot in multi_hot_vectors]
        else:
            raise ValueError("Input multi_hot_vectors must be a 1D or 2D array.")
    
    def _labels_to_multi_hot(self, labels):
        """Converts a list of labels into a multi-hot vector."""
        multi_hot_labels = np.zeros(len(self.label_to_idx), dtype=np.float32)
        for label in labels:
            multi_hot_labels[self.label_to_idx[label]] = 1.0
        return multi_hot_labels
    
    def __getitem__(self, idx):
        prompt = self.dataset[idx.item()][self.prompt_column]
        labels = self.dataset[idx.item()][self.label_column]

        # Convert labels to multi-hot vector using helper method
        multi_hot_labels = torch.tensor(self._labels_to_multi_hot(labels), dtype=torch.float32)

        # Use the CLIP processor to tokenize the prompt into input_ids
        inputs = self.processor(text=prompt, return_tensors="pt", padding="max_length")
        # input_ids = inputs['input_ids'].squeeze()  # Get input IDs

        return inputs, multi_hot_labels

    def get_num_classes(self):
        """Returns the number of unique classes in the dataset."""
        return len(self.label_to_idx)

    def get_classes(self):
        """Returns the list of unique classes in the dataset."""
        return list(self.label_to_idx.keys())
        
    def get_all_labels(self):
        """Returns a NumPy array of multi-hot vectors for all samples in the dataset."""
        all_labels = [self._labels_to_multi_hot(self.dataset[idx][self.label_column]) for idx in range(len(self.dataset))]
        return np.array(all_labels)
    
    def split_dataset(self, config):
        """
        Splits a multi-label dataset into training, validation, and test sets.
        
        Args:
            config: Configuration object containing split ratios and other settings.
            full_dataset: An instance of MultiLabelRelationalDataset.
        
        Returns:
            train_loader, val_loader, test_loader: DataLoaders for train, validation, and test sets.
        """
        from sklearn.model_selection import train_test_split
        from torch.utils.data import Subset

        logger.info(f"Loaded dataset with {len(self)} samples and {self.get_num_classes()} classes.")
        logger.info(f"Splitting dataset into train, validation, and test sets...")

        # Step 1: Get all labels (multi-hot vectors) and compute sample-wise class representation
        labels = np.array(self.get_all_labels())  # This should return a multi-hot numpy array


        # Step 2: Assign each sample a dominant label for stratification
        dominant_labels = labels.argmax(axis=1)  # Take the most significant label for stratification

        # Step 3: Split into 80% train and 20% remaining (val + test)
        train_split = config.train.train_split  # 0.8 for train
        remaining_split = 1 - train_split  # 0.2 for validation and test
        train_indices, remaining_indices = train_test_split(
            np.arange(len(labels)),
            test_size=remaining_split,
            stratify=dominant_labels,
            random_state=config.train.seed
        )

        # Step 4: Split the remaining 20% into validation and test sets
        val_split = config.train.val_split / remaining_split  # Adjust validation split proportion
        remaining_dominant_labels = dominant_labels[remaining_indices]
        val_indices, test_indices = train_test_split(
            remaining_indices,
            test_size=1 - val_split,
            stratify=remaining_dominant_labels,
            random_state=config.train.seed
        )

        # Step 5: Create Subsets for each split
        train_dataset = Subset(self, train_indices)
        val_dataset = Subset(self, val_indices)
        test_dataset = Subset(self, test_indices)

        # Step 6: Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
class SingleLabelRelationalDataset(Dataset):
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
        # input_ids = inputs['input_ids'].squeeze()  # Get input IDs

        return inputs, label_idx

    def get_num_classes(self):
        """Returns the number of unique classes in the dataset."""
        return len(self.label_to_idx)

    def get_classes(self):
        """Returns the list of unique classes in the dataset."""
        return list(self.label_to_idx.keys())
    
    def get_all_labels(self):
        """Returns a list of all labels in the dataset."""
        return [self.dataset[i][self.label_column] for i in range(len(self.dataset))]
    
        
    def split_dataset(self, config):
        
        logger.info(f"Loaded dataset with {len(self)} samples and {self.get_num_classes()} classes.")
        logger.info(f"Splitting dataset into train, validation, and test sets...")
        # Assuming labels are accessible in self and `get_labels()` gives label list
        labels = np.array(self.get_all_labels())

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
        train_dataset = torch.utils.data.Subset(self, train_indices.tolist())
        val_dataset = torch.utils.data.Subset(self, val_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)


        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config.train.batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
# Step 4: Define a custom collate function to handle variable-length inputs
# def collate_fn(batch):
#     input_ids = [item[0] for item in batch['']]
#     if isinstance(batch[0][1], torch.Tensor):
#         labels = torch.stack([item[1] for item in batch])
#     else:
#         labels = torch.tensor([item[1] for item in batch])
    
#     # Pad the input_ids to the same length
#     input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
#     return input_ids_padded, labels

def get_dataset(config,dataset_class, dataset_path, processor):
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}")
    else:
        full_dataset = dataset_class(
            dataset_file=dataset_path,
            processor=processor,  # Pass processor here
            prompt_column=config.dataset.prompt_column,
            label_column=config.dataset.label_column
        )
    return full_dataset