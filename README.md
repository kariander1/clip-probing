# Text Classification with Frozen CLIP Embeddings

This project uses a pre-trained CLIP model to extract frozen text embeddings and trains a linear probe classifier on top of these embeddings.

## Project Structure
- `src/` contains all source code for dataset handling, model definition, training, and evaluation.
- `scripts/` contains entry-point scripts to run training and evaluation.
- `config.py` stores hyperparameters and configuration values.

## How to Run
1. **Train the Model**
   ```bash
   python scripts/run_training.py
