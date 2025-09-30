# Examples

This directory contains example scripts demonstrating different aspects of the transformers stack.

## Training Examples

### 1. LoRA Fine-tuning (`train_lora.py`)

Fine-tune models using LoRA (Low-Rank Adaptation) with Accelerate for efficient, parameter-efficient training.

```bash
python examples/train_lora.py
```

**Features:**
- LoRA for parameter-efficient fine-tuning
- Accelerate for distributed training
- Minimal example on IMDB dataset

### 2. Trainer API (`train_with_trainer.py`)

Use the Hugging Face Trainer API for supervised fine-tuning with built-in features.

```bash
python examples/train_with_trainer.py
```

**Features:**
- DataCollatorWithPadding for efficient batching
- Automatic checkpointing and evaluation
- Built-in metrics computation
- Easy integration with W&B/MLflow (set `report_to`)

## Evaluation

### Model Evaluation (`evaluate_model.py`)

Evaluate a trained model with comprehensive metrics.

```bash
python examples/evaluate_model.py \
    --model_path ./output/trainer/best_model \
    --dataset imdb \
    --split "test[:10%]" \
    --batch_size 16 \
    --output results.json
```

**Metrics computed:**
- Accuracy
- F1 Score
- Precision
- Recall

## Tips

### Using with Hydra Config

You can integrate these examples with the Hydra configs in `conf/`:

```python
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    # Your training code using cfg.model, cfg.data, etc.
    pass
```

### Experiment Tracking

To enable W&B tracking in `train_with_trainer.py`:

```python
training_args = TrainingArguments(
    ...,
    report_to="wandb",  # or "mlflow"
)
```

Make sure to set your API key:
```bash
export WANDB_API_KEY=your_key_here
```

### GPU/MPS Support

All scripts automatically detect and use available hardware:
- CUDA (NVIDIA GPUs)
- MPS (Apple Silicon)
- CPU (fallback)

### Distributed Training

For multi-GPU training with Accelerate:

```bash
accelerate config  # One-time setup
accelerate launch examples/train_lora.py
```
