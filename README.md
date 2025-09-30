# Transformers Stack

[![CI](https://github.com/evalops/stack/actions/workflows/ci.yml/badge.svg)](https://github.com/evalops/stack/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A production-ready Transformers stack built on PyTorch with uv for reproducible dependency management. This stack covers model definition, data loading, training/inference infrastructure, optimization/quantization, evaluation, deployment, and environment management.

**Features:**
- 🚀 **Fast Setup**: Ultra-fast dependency management with `uv`
- 🧪 **Tested**: Comprehensive test suite with 13+ unit tests
- 🐳 **Docker Ready**: CPU and CUDA Dockerfiles included
- 📊 **Serving**: Production FastAPI server with Prometheus metrics
- 🎯 **CI/CD**: GitHub Actions workflows included
- 🔧 **Configurable**: Hydra configs for reproducible experiments
- 📚 **Examples**: Multiple training and evaluation scripts

## Core Components

| Layer | Packages & Rationale | Stable Version(s) |
|-------|---------------------|-------------------|
| **Base framework** | `torch` provides tensors, autograd and GPU support | `torch==2.8.0` |
| **Model definitions** | `transformers[torch]` supplies ~100 model architectures plus ready‑made pipelines | `transformers[torch]==4.56.2` |
| **Datasets & evaluation** | `datasets` offers fast dataset loading & streaming; `evaluate` and `scikit‑learn` provide metrics and classical ML utilities | `datasets==4.1.1`, `evaluate>=0.4.1`, `scikit-learn>=1.3` |
| **Environment management** | `uv` acts as a drop‑in replacement for pip/pip-tools with ultra-fast installation | `uv` (installed via script or pip) |

## Performance & Fine-tuning Extensions

| Purpose | Package(s) | Notes |
|---------|-----------|-------|
| **Multi‑GPU & distributed** | `accelerate==1.10.1` | Simplifies multi‑GPU, multi‑node and mixed‑precision training |
| **Parameter‑efficient fine‑tuning** | `peft==0.17.1` | Provides LoRA/P‑Tuning/etc. |
| **Quantization & 8‑bit ops** | `bitsandbytes==0.47.0` | Adds 8‑bit optimizers and int8 matmul; requires CUDA |
| **Memory‑efficient attention** | `flash-attn==2.8.3`, `xformers==0.0.32.post2` | FlashAttention and alternative efficient attention kernels |
| **High‑throughput inference** | `vllm==0.10.2` | Serves large‑language models with continuous batching |
| **Logging & monitoring** | `wandb` or `mlflow` | For experiment tracking (add as needed) |

## Quick Start

### Installation

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone the repository
git clone https://github.com/evalops/stack.git
cd stack

# 3. Create virtual environment
uv venv --python=python3.11 .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Install dependencies
uv pip compile pyproject.toml -o requirements.txt
uv pip sync requirements.txt

# 5. Optional: Install additional features
# For serving:
uv pip sync requirements-serve.txt

# For development:
uv pip sync requirements-dev.txt
```

### Running Tests

```bash
pytest tests/ -v --cov=src
```

### Starting the Inference Server

```bash
python serving/app.py
# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
# Metrics at http://localhost:8000/metrics
```

### Training a Model

```bash
# Using LoRA
python examples/train_lora.py

# Using Trainer API
python examples/train_with_trainer.py
```

## Example Usage

### Train a model with LoRA and evaluate

```python
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import LoraConfig, get_peft_model
import torch
from accelerate import Accelerator

acc = Accelerator()
ds = load_dataset("imdb", split="train[:1%]")  # small subset for demo

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
enc = ds.map(lambda ex: tokenizer(ex["text"], truncation=True, padding=True), batched=True)
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16))

model, optimizer, _, dataloader = acc.prepare(
    model,
    torch.optim.Adam(model.parameters(), lr=2e-5),
    None,
    torch.utils.data.DataLoader(enc.with_format("torch"), batch_size=8, shuffle=True),
)
model.train()
for batch in dataloader:
    with acc.accumulate(model):
        outputs = model(**{k: batch[k] for k in ["input_ids", "attention_mask"]}, labels=batch["label"])
        loss = outputs.loss
        acc.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
```

This example shows the interaction between `datasets`, `transformers`, `peft`, and `accelerate`. Swap in `bitsandbytes` optimizers or `flash-attn` kernels as your hardware allows.

## Docker Deployment

### Build and Run with Docker

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t stack:cpu .

# Run container
docker run -p 8000:8000 stack:cpu

# Or use Docker Compose
docker-compose up inference-cpu
```

### CUDA/GPU Deployment

```bash
# Build CUDA image
docker build -f Dockerfile.cuda -t stack:cuda .

# Run with GPU access
docker run --gpus all -p 8000:8000 stack:cuda
```

### API Endpoints

Once running, the server exposes:
- `GET /health` - Health check
- `GET /ready` - Readiness probe
- `POST /predict` - Run inference
- `GET /metrics` - Prometheus metrics
- `GET /docs` - Interactive API documentation

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!", "return_all_scores": false}'
```

## Configuration with Hydra

The stack uses Hydra for configuration management. Configs are in `conf/`:

```yaml
# conf/config.yaml
defaults:
  - model: bert_base
  - data: imdb
  - train: default
  - eval: default
  - system: auto

task: seq_cls
seed: 42
output_dir: outputs/${now:%Y%m%d-%H%M%S}
```

Override configs from command line:

```bash
python train.py model=bert_base data=imdb train.epochs=5 train.batch_size=16
```

## Important Notes & Hazards

### CUDA-only modules

**⚠️ bitsandbytes, flash-attn and xformers require NVIDIA GPUs** and won't work on Apple Silicon or CPU‑only setups. If you're on macOS, omit them or use CPU/Metal‐accelerated alternatives (e.g., skip `bitsandbytes` and rely on full‑precision training).

### vLLM vs. Transformers inference

The Hugging Face `pipeline` API is fine for small tests, but for high‑throughput evaluation or serving you'll want `vllm`, which uses continuous batching. Ensure your GPU has enough memory.

### Stay pinned

Periodically check for new releases, update your `pyproject.toml` versions, run `uv pip compile` again, and sync. This keeps your stack consistent while still benefiting from improvements.

## Project Structure

```
stack/
├── pyproject.toml              # Project configuration and dependencies
├── README.md                   # This file
├── .github/
│   └── workflows/
│       └── ci.yml             # GitHub Actions CI workflow
├── conf/                       # Hydra configuration files
│   ├── config.yaml            # Main config
│   ├── model/                 # Model configs
│   ├── data/                  # Dataset configs
│   ├── train/                 # Training configs
│   ├── eval/                  # Evaluation configs
│   └── system/                # System/hardware configs
├── examples/                   # Example training scripts
│   ├── train_lora.py          # LoRA fine-tuning
│   ├── train_with_trainer.py  # Trainer API example
│   ├── evaluate_model.py      # Model evaluation
│   └── README.md              # Examples documentation
├── serving/                    # Production inference server
│   ├── app.py                 # FastAPI application
│   └── test_server.py         # Server tests
├── src/
│   └── transformers_stack/    # Main package
├── templates/                  # Documentation templates
│   ├── model_card.md          # Model card template
│   └── dataset_card.md        # Dataset card template
├── tests/                      # Test suite
│   ├── test_model.py          # Model tests
│   ├── test_peft.py           # LoRA/PEFT tests
│   └── test_tokenization.py   # Tokenization tests
├── Dockerfile.cpu              # CPU inference Docker image
├── Dockerfile.cuda             # CUDA inference Docker image
├── docker-compose.yml          # Docker Compose configuration
├── requirements.txt            # Core dependencies (locked)
├── requirements-serve.txt      # Serving dependencies (locked)
├── requirements-dev.txt        # Dev dependencies (locked)
└── .pre-commit-config.yaml    # Pre-commit hooks
```

## Development

### Install Development Tools

```bash
uv pip sync requirements-dev.txt
```

This includes:
- `ruff` - Fast Python linter
- `black` - Code formatter
- `pytest` + `pytest-cov` + `pytest-xdist` - Testing framework with coverage and parallel execution
- `mypy` - Static type checker
- `pre-commit` - Git hooks
- `mkdocs-material` - Documentation site generator
- `ipykernel` - Jupyter notebook support

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=term --cov-report=html

# Run in parallel
pytest tests/ -n auto
```

### Code Quality

```bash
# Lint
ruff check .

# Format
black .

# Type check
mypy src/
```

## License

MIT

## Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

## Resources

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [uv Documentation](https://docs.astral.sh/uv/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)
- [Accelerate Documentation](https://huggingface.co/docs/accelerate/)
