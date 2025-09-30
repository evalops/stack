"""
Test script to verify the transformers stack installation
"""

print("Testing imports...")

try:
    import torch

    print(f"✓ PyTorch {torch.__version__}")
    print(f"  - CUDA available: {torch.cuda.is_available()}")
    print(f"  - MPS available: {torch.backends.mps.is_available()}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    exit(1)

try:
    import transformers

    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers import failed: {e}")
    exit(1)

try:
    import datasets

    print(f"✓ Datasets {datasets.__version__}")
except ImportError as e:
    print(f"✗ Datasets import failed: {e}")
    exit(1)

try:
    import peft

    print(f"✓ PEFT {peft.__version__}")
except ImportError as e:
    print(f"✗ PEFT import failed: {e}")
    exit(1)

try:
    import accelerate

    print(f"✓ Accelerate {accelerate.__version__}")
except ImportError as e:
    print(f"✗ Accelerate import failed: {e}")
    exit(1)

try:
    import evaluate

    print(f"✓ Evaluate {evaluate.__version__}")
except ImportError as e:
    print(f"✗ Evaluate import failed: {e}")
    exit(1)

try:
    import sklearn

    print(f"✓ Scikit-learn {sklearn.__version__}")
except ImportError as e:
    print(f"✗ Scikit-learn import failed: {e}")
    exit(1)

print("\nAll core packages installed successfully!")

print("\nTesting basic functionality...")

try:
    from transformers import pipeline

    print("✓ Creating a simple sentiment analysis pipeline...")
    classifier = pipeline(
        "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=-1
    )
    result = classifier("I love this stack!")[0]
    print(f"  - Test result: {result['label']} (score: {result['score']:.4f})")
    print("✓ Pipeline test successful!")
except Exception as e:
    print(f"✗ Pipeline test failed: {e}")
    exit(1)

print("\n✅ Installation verified successfully!")
