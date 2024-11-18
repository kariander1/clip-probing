import pyrallis
import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from src.utils.enums import TextModel, Dataset_Type, OptimizerName

@dataclass
class DatasetConfig:
    prompt_column: str = "prompt"
    label_column: str = "label"
    dataset_type: Dataset_Type = Dataset_Type.RELATIONAL_POSITIONAL

@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 4  # Add batch_size here
    lr: float = 0.001  # Learning rate
    optimizer: OptimizerName = OptimizerName.ADAM  # Optimizer
    train_split: float = 0.8  # Train split
    val_split: float = 0.1  # Validation split
    seed: int = 42  # Seed for reproducibility
    patience: int = 7  # Patience for early stopping
    
@dataclass
class LoggingConfig:
    plot_distribution: bool = False  # Plot label distribution
    log_interval: int = 10  # Log interval

@dataclass
class ModelConfig:
    model_type: TextModel = TextModel.CLIP_VIT_B_16

@dataclass
class Config:
    output_root: Path = field(default=Path("./output"))
    checkpoint_root: Path = field(default=Path("./checkpoints"))
    override_results: bool = False  # Override existing checkpoints
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log: LoggingConfig = field(default_factory=LoggingConfig)
    
    @property
    def output_dir(self) -> Path:
        return self.output_root / self.dataset.dataset_type.name / self.model.model_type.name

    @property
    def checkpoint_dir(self) -> Path:
        return self.checkpoint_root / self.dataset.dataset_type.name / self.model.model_type.name

@pyrallis.wrap()
def get_config() -> Config:
    return Config()

