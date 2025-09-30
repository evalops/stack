"""Optional installation verification tests.

These tests validate that key libraries import correctly and that a simple
transformers pipeline can run end-to-end. They are skipped by default because
running the pipeline requires downloading public model weights.
"""

from __future__ import annotations

import os

import pytest

RUN_INSTALLATION_TESTS = os.getenv("RUN_INSTALLATION_TESTS") == "1"

if not RUN_INSTALLATION_TESTS:
    pytest.skip(
        "Set RUN_INSTALLATION_TESTS=1 to run installation verification.",
        allow_module_level=True,
    )


def test_core_libraries_importable():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")
    pytest.importorskip("datasets")
    pytest.importorskip("evaluate")
    pytest.importorskip("peft")

    assert torch.__version__
    assert transformers.__version__


@pytest.mark.slow
@pytest.mark.timeout(120)
def test_sentiment_pipeline_executes():
    from transformers import pipeline

    classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1,
    )

    result = classifier("I love this stack!")[0]
    assert result["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0 <= result["score"] <= 1
