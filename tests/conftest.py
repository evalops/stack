"""Pytest configuration and fixtures"""

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_model_name():
    """Use a tiny model for fast testing"""
    return "hf-internal-testing/tiny-random-bert"


@pytest.fixture(scope="session")
def device():
    """Get available device for testing"""
    if torch.cuda.is_available():
        return "cuda"

    # Some PyTorch builds (e.g. Linux CPU wheels) do not expose the MPS backend
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


@pytest.fixture(scope="session")
def synthetic_dataset():
    """Create a small synthetic dataset for testing"""
    return {
        "train": [
            {"text": "This is a positive example", "label": 1},
            {"text": "This is a negative example", "label": 0},
            {"text": "Another positive text", "label": 1},
            {"text": "Another negative text", "label": 0},
        ],
        "test": [
            {"text": "Test positive", "label": 1},
            {"text": "Test negative", "label": 0},
        ],
    }
