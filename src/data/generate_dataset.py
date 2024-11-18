import pyrallis
import torch
import re
from tqdm import tqdm
from datasets import Dataset as HFDataset
from src.config import Config

# Relational words to use for prompt generation
RELATIONAL_WORDS = ["on", "behind", "on the left of", "inside"]

# Base entities for generating relational sentences
BASE_ENTITIES = list(set(["bowl","plate","cup","fork","knife","spoon","glass","bottle","table", "desk", "chair", "sofa", "bed", "lamp", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "toilet", "sink", "refrigerator", "oven", "microwave", "toaster", "sink", "mirror", "bathtub", "shower", "keyboard", "mouse", "remote", "cell phone", "microwave", "oven", "toaster", "refrigerator", "sink", "book", "clock"]))
BASE_PROMPTS = ["", "A photo of", "An image of", "A picture of", "A drawing of", "A painting of", "A high resolution image of", "Pastoral background,","rainy,","cloudy,","sunny,", "A depiction of", "A photograph of", "A close-up of", "A snapshot of", "A close-up close-up drawing of"]
def generate_variations(base_prompt ,entity1, entity2, relational_word):
    return [
        ([f"{base_prompt} a {entity1} {relational_word} a {entity2}"], [f"{'_'.join(relational_word.split(' '))}_{entity1}_{entity2}"]),
        ([f"{base_prompt} a {entity2} {relational_word} a {entity1}"], [f"{'_'.join(relational_word.split(' '))}_{entity2}_{entity1}"])
    ]

@pyrallis.wrap()
def main(config: Config):
    data = {
        "prompt": [],
        "label": []
    }

    # Generate relational prompts and labels
    for relational_word in tqdm(RELATIONAL_WORDS):
        for i, entity1 in enumerate(BASE_ENTITIES):
            for entity2 in BASE_ENTITIES[i + 1:]:
                for base_prompt in BASE_PROMPTS:
                    variations = generate_variations(base_prompt ,entity1, entity2, relational_word)
                    for prompt, label in variations:
                        data["prompt"].extend(prompt)
                        data["label"].extend(label)

    # Convert to Hugging Face Dataset
    hf_dataset = HFDataset.from_dict(data)

    # Save the dataset to disk (JSON or CSV)
    dataset_file = 'generated_dataset.json'
    if dataset_file.endswith(".json"):
        hf_dataset.to_json(dataset_file)
    elif dataset_file.endswith(".csv"):
        hf_dataset.to_csv(dataset_file)
    else:
        raise ValueError("Unsupported file format. Use either .json or .csv.")
    
    print(f"Dataset generated and saved to {dataset_file}")

if __name__ == "__main__":
    main()
