import pyrallis
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DatasetConfig:
    prompt_column: str = "prompt"
    label_column: str = "label"
    num_variations: int = 10  # Number of variations to generate per prompt pair
    categories: Optional[List[str]] = None  # Optional categories for base prompts
    model_name: str = "llava-hf/llava-1.5-13b-hf"  # Model name for generation
    dataset_file: str = "data/relational_dummy.json"  # Where the dataset will be saved

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4  # Add batch_size here
    lr: float = 0.001  # Learning rate
    optimizer: str = "adam"
    train_split: float = 0.8  # Train split
    val_split: float = 0.1  # Validation split
    log_interval: int = 10  # Log interval
    seed: int = 42  # Seed for reproducibility
    override: bool = False  # Override existing checkpoints
    plot_distribution: bool = True  # Plot label distribution

@dataclass
class ModelConfig:
    clip_model_name: str = "openai/clip-vit-base-patch32"
    input_dim: int = 512

@dataclass
class Config:
    output_root: Path = field(default=Path("./output"))
    checkpoint_root: Path = field(default=Path("./checkpoints"))
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    
    @property
    def output_dir(self) -> Path:
        return self.output_root / Path(self.dataset.dataset_file).stem

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / Path(self.dataset.dataset_file).stem

@pyrallis.wrap()
def get_config() -> Config:
    return Config()

