import torch
from torch.utils.data import Dataset
from datasets import load_dataset

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
        inputs = self.processor(text=prompt, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs['input_ids'].squeeze()  # Get input IDs

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
