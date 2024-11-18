
# Evaluating Text Embeddings in CLIP Models for Visual-Language Tasks

![Project Teaser](assets/teaser_prompt.png)

This repository focuses on evaluating text embeddings in CLIP models for visual-language tasks. By probing the capabilities of CLIP embeddings, the project explores various visual-language tasks and benchmarks the performance of text embeddings.

---

## Features
- Training and evaluation scripts for benchmarking CLIP models.
- Configuration-driven setup for flexible experimentation.
- Results are stored in organized directories for easy access.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/kariander1/clip-probing
   cd clip-probing
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Project Structure
- `config/`: Contains all configuration files required for training and evaluation.
- `checkpoints/`: Stores the checkpoints and trained models.
- `output/`: Contains evaluation results, including metrics and visualizations.
- `data/`: Contains the datasets generated for the benchmarks.
- `src/`: Source code for implementing probing and dataset handling.
  - `data/`: Contains scripts for dataset generation and preprocessing.
- `scripts/`: Core scripts for training and evaluation.
  - `run_training.py`: Script for training.
  - `run_evaluation.py`: Script for evaluation.

---

## Training

To train a model, run the following command:

```bash
python -m scripts.run_training --config CONFIG_FILE.yaml
```

- Replace `CONFIG_FILE.yaml` with the path to the desired configuration file located in the `config/` directory.
- The training checkpoint and the saved model will be stored in the `checkpoints/` directory.

---

## Evaluation

To evaluate a model, run the following command:

```bash
python -m scripts.run_evaluation.py --config CONFIG_FILE.yaml
```

- Replace `CONFIG_FILE.yaml` with the path to the desired configuration file located in the `config/` directory.
- The evaluation results, including metrics, will be saved in the `output/` directory.

---

## Data Generation

To generate datasets for relational tasks, use the scripts provided in the `src/data` directory:
- **Relational Positional Dataset**: Generates positional datasets for relational reasoning.
- **Relational Positional LLM Dataset**: Creates relational datasets using large language models (LLMs).
- **GNOME Dataset Preprocessing**: Contains code to preprocess the GNOME dataset for relation prediction tasks.

### Example Command:
Navigate to the `src/data` directory and run:
```bash
python generate_relational_positional_dataset.py
python generate_relational_positional_llm_dataset.py
python preprocess_gnome_dataset.py
```

The generated datasets will be stored in the `data/` directory.

---

## Example Usage

### Training Example:
```bash
python -m scripts.run_training --config config/config_RELATIONAL_POSITIONAL_LLM_CLIP_VIT_B_16.yaml
```

### Evaluation Example:
```bash
python -m scripts.run_evaluation.py --config config/config_RELATIONAL_POSITIONAL_LLM_CLIP_VIT_B_16.yaml
```

---

## Acknowledgments
This project builds upon CLIP and related visual-language tasks research. We extend our gratitude to the authors of [CLIP](https://github.com/openai/CLIP) and associated tools used in this project.
