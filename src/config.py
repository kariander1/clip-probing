import pyrallis
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class DatasetConfig:
    prompt_column: str = "prompt"
    label_column: str = "label"
    num_variations: int = 10  # Number of variations to generate per prompt pair
    categories: Optional[List[str]] = None  # Optional categories for base prompts
    model_name: str = "llava-hf/llava-v1.6-mistral-7b-hf"  # Model name for generation
    dataset_file: str = "data/relational_dataset.json"  # Where the dataset will be saved

@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 4  # Add batch_size here
    lr: float = 0.001  # Learning rate
    train_split: float = 0.9  # Train-test split
@dataclass
class ModelConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    input_dim: int = 512

@dataclass
class Config:
    dataset: DatasetConfig = DatasetConfig()
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()

@pyrallis.wrap()
def get_config() -> Config:
    return Config()
