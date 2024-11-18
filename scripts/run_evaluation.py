import torch
import pyrallis
from loguru import logger
from src.data.dataset import get_dataset
from src.model import LinearProbe
from src.train import evaluate_probes, load_best_checkpoint
from src.config import Config
from src.utils.enums_utils import (
    get_text_processor_and_model,
    dataset_type_to_class,
    dataset_type_to_path,
    dataset_type_to_criterion,
    dataset_labels_are_super_labels,
)

@pyrallis.wrap()
def main(config: Config):
    # Set device
    print(pyrallis.dump(config))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load CLIP model and processor
    processor, model, tokenizer_max_length, hidden_size = get_text_processor_and_model(config.model.model_type, device)

    # Load dataset and DataLoader
    logger.info("Loading dataset...")
    dataset_class = dataset_type_to_class(config.dataset.dataset_type)
    dataset_path = dataset_type_to_path(config.dataset.dataset_type)
    full_dataset = get_dataset(config, dataset_class, dataset_path, processor)
    _, _, test_loader = full_dataset.split_dataset(config)
    has_super_labels = dataset_labels_are_super_labels(config.dataset.dataset_type)

    # Get number of classes
    num_classes = full_dataset.get_num_classes()

    # Initialize probes
    probes = {
        "embedding_probe": LinearProbe(tokenizer_max_length * hidden_size, num_classes).to(device),
    }

    # Load best checkpoints for probes
    for probe_name in probes:
        load_best_checkpoint(probe_name, probes[probe_name], None, config.checkpoint_dir)

    # Define criterion
    criterion = dataset_type_to_criterion(config.dataset.dataset_type)

    # Evaluate probes
    logger.info("Starting evaluation...")
    evaluate_probes(
        model,
        processor,
        probes,
        test_loader,
        criterion,
        device,
        plot_misclassifications=True,
        output_dir=config.output_dir,
        has_super_labels=has_super_labels,
    )
    logger.success("Evaluation complete.")


if __name__ == "__main__":
    main()
