import pyrallis
import torch
import re
from tqdm import tqdm
from datasets import Dataset as HFDataset
from transformers import AutoProcessor, LlavaForConditionalGeneration
from src.config import Config

# Relational words to use for prompt generation
# RELATIONAL_WORDS = ["behind", "looking at", "to the right of", "pointing at", "inside a house, and outside"]
RELATIONAL_WORDS = ["behind", "looking at"]

# Base entities for generating relational sentences
# BASE_ENTITIES = ["man","woman", "dog", "cat", "car", "tree", "house", "bird"]
BASE_ENTITIES = ["dog", "cat"]
BASE_PROMPTS = ["A photo of", "An image of", "A picture of", "A drawing of", "A painting of", "A high resolution image of"]
def generate_variations(base_prompt ,entity1, entity2, relational_word, model, processor, num_variations, device):
    variation_1_prompt = f"{base_prompt} a {entity1} {relational_word} a {entity2}"
    variation_2_prompt = f"{base_prompt} a {entity2} {relational_word} a {entity1}"

    # Format prompts for LLaVA model using its processor
    conversation_1 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Please generate {num_variations} variations of the following prompt: '{variation_1_prompt}'"}]
        }
    ]
    conversation_2 = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Please generate {num_variations} variations of the following prompt: '{variation_2_prompt}'"}]
        }
    ]

    # Process the prompt for the LLaVA model
    prompt_1 = processor.apply_chat_template(conversation_1, add_generation_prompt=True)
    prompt_2 = processor.apply_chat_template(conversation_2, add_generation_prompt=True)

    # Generate the responses using LLaVA
    inputs_1 = processor(text=prompt_1, return_tensors='pt').to(device)
    inputs_2 = processor(text=prompt_2, return_tensors='pt').to(device)

    output_1 = model.generate(**inputs_1, max_new_tokens=200)
    output_2 = model.generate(**inputs_2, max_new_tokens=200)

    variation_1 = processor.decode(output_1[0], skip_special_tokens=True).strip().split('ASSISTANT:')[1].strip().strip('\'')
    variation_2 = processor.decode(output_2[0], skip_special_tokens=True).strip().split('ASSISTANT:')[1].strip().strip('\'')

    variations_1 = [s.strip().strip('\"').strip('\'') for s in re.split(r'\d+\.\s*', variation_1)[1:]]
    variations_2 = [s.strip().strip('\"').strip('\'') for s in re.split(r'\d+\.\s*', variation_2)[1:]]
    if len(variations_1) < num_variations or len(variations_2) < num_variations:
        print(f"Warning: Could not generate {num_variations} variations for the prompt: '{variation_1_prompt}' or '{variation_2_prompt}'")
        return [[[],[]],[[],[]]]
    return [
        (variations_1, [f"{'_'.join(relational_word.split(' '))}_{entity1}_{entity2}"]*len(variations_1)),
        (variations_2, [f"{'_'.join(relational_word.split(' '))}_{entity2}_{entity1}"]*len(variations_2))
        ]

@pyrallis.wrap()
def main(config: Config):
    # Initialize a text-generation pipeline (e.g., GPT-2, GPT-Neo)
    # text_generator = pipeline("text-generation", model=config.dataset.model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LlavaForConditionalGeneration.from_pretrained(config.dataset.model_name, torch_dtype=torch.float32).to(device)
    processor = AutoProcessor.from_pretrained(config.dataset.model_name)
    
    data = {
        "prompt": [],
        "label": []
    }

    # Generate relational prompts and labels
    for relational_word in tqdm(RELATIONAL_WORDS):
        for i, entity1 in enumerate(BASE_ENTITIES):
            for entity2 in BASE_ENTITIES[i + 1:]:
                for base_prompt in BASE_PROMPTS:
                    variations = generate_variations(base_prompt, entity1, entity2, relational_word, model, processor, config.dataset.num_variations, device)
                    for prompt, label in variations:
                        data["prompt"].extend(prompt)
                        data["label"].extend(label)

    # Convert to Hugging Face Dataset
    hf_dataset = HFDataset.from_dict(data)

    # Save the dataset to disk (JSON or CSV)
    dataset_file = config.dataset.dataset_file
    if dataset_file.endswith(".json"):
        hf_dataset.to_json(dataset_file)
    elif dataset_file.endswith(".csv"):
        hf_dataset.to_csv(dataset_file)
    else:
        raise ValueError("Unsupported file format. Use either .json or .csv.")
    
    print(f"Dataset generated and saved to {dataset_file}")

if __name__ == "__main__":
    main()
