import torch
import torch.nn as nn
import torch.optim as optim
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader, Dataset
import numpy as np

EPOCHS = 20
# Step 1: Load Pre-trained CLIP Model and Processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()  # Set model to evaluation mode (freeze parameters)

# Example dataset of prompts and labels (dog, cat, or none)
data = [
    # Dog-related prompts (label: 0)
    ("A cute dog playing in the park", 0),
    ("A dog chasing a ball in the garden", 0),
    ("A small puppy chewing on a toy", 0),
    ("A golden retriever swimming in a lake", 0),
    ("A dog barking at the mailman", 0),
    ("Two dogs running together at the beach", 0),
    ("A dog wagging its tail happily", 0),
    ("A big brown dog sleeping under the tree", 0),
    ("A dog digging a hole in the backyard", 0),
    ("A dog fetching a frisbee thrown in the air", 0),
    
    # Cat-related prompts (label: 1)
    ("A fluffy cat sitting on the windowsill", 1),
    ("A black cat curled up on the sofa", 1),
    ("A cat chasing a mouse in the kitchen", 1),
    ("A kitten playing with a ball of yarn", 1),
    ("A cat licking its paws after eating", 1),
    ("A white cat napping on a pile of books", 1),
    ("A cat stretching lazily in the sun", 1),
    ("A cat jumping onto the dining table", 1),
    ("A cat hiding under the bed", 1),
    ("A ginger cat staring out of the window", 1),
    
    # None-related prompts (label: 2)
    ("A man riding a bicycle down the street", 2),
    ("A woman cooking pasta in the kitchen", 2),
    ("A family sitting at the dining table for dinner", 2),
    ("A car driving on the highway at sunset", 2),
    ("A child playing with blocks on the floor", 2),
    ("A group of friends playing soccer in the park", 2),
    ("A woman reading a book by the fireplace", 2),
    ("A man walking his bicycle across the road", 2),
    ("A person writing a letter at a desk", 2),
    ("A couple dancing in the rain", 2),
    
    # More dog-related prompts (label: 0)
    ("A dog running through the forest", 0),
    ("A puppy learning to climb the stairs", 0),
    ("A dog barking at its reflection in the mirror", 0),
    ("A dog catching a stick mid-air", 0),
    ("A police dog searching for evidence", 0),
    
    # More cat-related prompts (label: 1)
    ("A cat watching birds through the window", 1),
    ("A kitten climbing up a tall scratching post", 1),
    ("A cat playing with a laser pointer", 1),
    ("A tabby cat purring softly while being petted", 1),
    
    # More none-related prompts (label: 2)
    ("A chef preparing a large pot of soup", 2),
    ("A group of students studying for an exam", 2),
    ("A plane flying over a mountain range", 2),
    ("A person taking a photograph of a sunset", 2),
]

# Step 2: Create a custom dataset to handle text prompts and labels
class TextDataset(Dataset):
    def __init__(self, data, processor):
        self.texts = [item[0] for item in data]
        self.labels = [item[1] for item in data]
        self.processor = processor
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        return inputs['input_ids'].squeeze(), torch.tensor(label)

# Step 3: Prepare data
train_size = int(0.9 * len(data))  # 90% of the data
test_size = len(data) - train_size  # 10% of the data

train_data = data[:train_size]  # First 90% of the data for training
test_data = data[train_size:]   # Last 10% for testing

train_dataset = TextDataset(train_data, processor)
test_dataset = TextDataset(test_data, processor)

# Step 4: Define a custom collate function to handle variable-length inputs
def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    
    # Pad the input_ids to the same length
    input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    return input_ids_padded, labels

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

# Step 5: Define a simple linear probe (classifier)
class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(LinearProbe, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)  # Input is the text embedding size from CLIP

    def forward(self, x):
        return self.fc(x)

linear_probe = LinearProbe(input_dim=512, num_classes=3)  # CLIP text embeddings are 512-dim
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(linear_probe.parameters(), lr=0.001)

# Step 6: Train the linear probe using frozen CLIP text embeddings
def train_probe(model, linear_probe, train_loader, optimizer, criterion):
    model.eval()  # CLIP model is frozen (not trained)
    linear_probe.train()  # Train the probe
    for epoch in range(EPOCHS):  # Train for a few epochs
        for input_ids, labels in train_loader:
            with torch.no_grad():
                # Extract CLIP text embeddings (frozen)
                text_embeds = model.get_text_features(input_ids=input_ids).detach()

            # Train the linear probe
            optimizer.zero_grad()
            outputs = linear_probe(text_embeds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed with loss {loss.item()}")

# Step 7: Evaluate the linear probe
def evaluate_probe(model, linear_probe, test_loader):
    model.eval()
    linear_probe.eval()
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for input_ids, labels in test_loader:
            # Extract CLIP text embeddings (frozen)
            text_embeds = model.get_text_features(input_ids=input_ids).detach()

            # Predict using the trained linear probe
            outputs = linear_probe(text_embeds)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.numpy())
            y_pred.extend(predicted.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

# Step 8: Run training and evaluation
train_probe(model, linear_probe, train_loader, optimizer, criterion)
evaluate_probe(model, linear_probe, test_loader)
