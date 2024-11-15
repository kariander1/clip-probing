import torch
import torch.optim as optim
import torch.nn as nn
import pyrallis
import os
import random
import numpy as np

from loguru import logger
from transformers import CLIPProcessor, CLIPModel
from src.data.dataset import OfflineRelationalDataset, HuggingFaceDataset,split_dataset
from src.model import LinearProbe
from src.train import train_probes, evaluate_probes, load_best_checkpoint
from src.config import Config
from src.visualization.visualization import plot_label_distribution

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def handle_paths(config: Config):
    if not os.path.exists(config.output_dir):
        logger.info(f"Creating output directory at {config.output_dir}")
        os.makedirs(config.output_dir)
        
    # Train the model        
    if os.path.exists(config.checkpoint_dir):
        if not config.train.override:
            logger.error(f"Checkpoints already exist at {config.checkpoint_dir}. Set 'override' to True to overwrite.")
            raise FileExistsError(f"Checkpoints already exist at {config.checkpoint_dir}. Set 'override' to True to overwrite.")
        else:
            logger.warning(f"Overwriting existing checkpoints at {config.checkpoint_dir}")
            os.rmdir(config.checkpoint_dir)
            
    logger.info(f"Creating checkpoint directory at {config.checkpoint_dir}")
    os.makedirs(config.checkpoint_dir)
        
    with (config.output_dir / 'config.yaml').open('w') as f:
        pyrallis.dump(config, f)
    logger.info('\n' + pyrallis.dump(config))
        
@pyrallis.wrap()
def main(config: Config):
    handle_paths(config)
    set_seed(config.train.seed)
    # Check if CUDA is available and select the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CLIP model and processor based on config, move model to the selected device
    processor = CLIPProcessor.from_pretrained(config.model.clip_model_name)
    model = CLIPModel.from_pretrained(config.model.clip_model_name).to(device)
    model.eval()

    # Create dataset and pass the processor to tokenize the prompts
    logger.info("Loading dataset...")
    if os.path.exists(config.dataset.dataset_file):
        full_dataset = OfflineRelationalDataset(
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

    train_loader, val_loader, test_loader = split_dataset(config, full_dataset)
    if config.train.plot_distribution:
        logger.info("Plotting label distributions. (If this takes too long, set 'plot_distribution' to False in the config)")
        plot_label_distribution(train_loader, title="Train Label Distribution", save_path = config.output_dir)
        plot_label_distribution(val_loader, title="Validation Label Distribution", save_path = config.output_dir)
        plot_label_distribution(test_loader, title="Test Label Distribution", save_path = config.output_dir)
    
    # Model setup: Pass the dynamically retrieved number of classes to the probe
    num_classes = full_dataset.get_num_classes()
        # Initialize multiple probes
    probes = {
        "embedding_probe": LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "encoder.layers.0" : LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "encoder.layers.3" : LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "encoder.layers.5" : LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "encoder.layers.7" : LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "encoder.layers.11" : LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device),
        # "token_probe": LinearProbe(processor.tokenizer.model_max_length, num_classes).to(device)
        # Add more probes as needed, for example:
        # "intermediate_layer_probe": create_linear_probe("intermediate_layer_probe", intermediate_layer_dim, num_classes, device)
    }
    # probes.update(
    #     {f"encoder.layers.{i}": LinearProbe(processor.tokenizer.model_max_length*config.model.input_dim, num_classes).to(device) for i in range(0, 12, 2)}
    # )

    # Initialize optimizers for each probe
    optim_name_to_optimizer = {
        'adam': optim.Adam,
    }
    
    optimizer = optim_name_to_optimizer[config.train.optimizer]
    optimizers = {
        "embedding_probe": optimizer(probes["embedding_probe"].parameters(), lr=config.train.lr),
        # "encoder.layers.0": optimizer(probes["encoder.layers.0"].parameters(), lr=config.train.lr),
        # "encoder.layers.3": optimizer(probes["encoder.layers.3"].parameters(), lr=config.train.lr),
        # "encoder.layers.5": optimizer(probes["encoder.layers.5"].parameters(), lr=config.train.lr),
        # "encoder.layers.7": optimizer(probes["encoder.layers.7"].parameters(), lr=config.train.lr),
        # "encoder.layers.11": optimizer(probes["encoder.layers.11"].parameters(), lr=config.train.lr),
        # "token_probe": optim.Adam(probes["token_probe"].parameters(), lr=config.train.lr)
    }
    # optimizers.update(
    #     {f"encoder.layers.{i}": optimizer(probes[f"encoder.layers.{i}"].parameters(), lr=config.train.lr) for i in range(0, 12, 2)}
    # )

    # linear_probe = LinearProbe(input_dim=config.model.input_dim, num_classes=num_classes).to(device)
    # optimizer = optim.Adam(linear_probe.parameters(), lr=config.train.lr)
    criterion = nn.CrossEntropyLoss()        
    
    train_probes(model, processor, probes, train_loader, val_loader, optimizers, criterion, config.train.epochs, device, config.checkpoint_dir,config.train.log_interval)
    logger.success("Training complete.")
    # After training, load the best checkpoints for each probe
    for probe_name in probes:
        load_best_checkpoint(probe_name, probes[probe_name], optimizers[probe_name], config.checkpoint_dir)
        
    # Evaluate the model
    evaluate_probes(model, processor, probes, test_loader, criterion, device, plot_misclassifications=True, output_dir=config.output_dir)

if __name__ == "__main__":
    main()