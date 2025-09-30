"""Test model loading and basic operations"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_model_load(tiny_model_name):
    """Test that model loads correctly"""
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)
    assert model is not None
    assert hasattr(model, "forward")


def test_forward_pass(tiny_model_name):
    """Test a single forward pass"""
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

    text = "This is a test"
    inputs = tokenizer(text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (1, 2)  # batch_size=1, num_labels=2


def test_training_step(tiny_model_name):
    """Test a single training step (forward + backward)"""
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

    text = "Training example"
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([1])

    model.train()
    outputs = model(**inputs, labels=labels)

    assert hasattr(outputs, "loss")
    assert outputs.loss.requires_grad

    # Test backward pass
    outputs.loss.backward()

    # Check that gradients exist
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            break


def test_batch_forward(tiny_model_name):
    """Test batch forward pass"""
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

    texts = ["Text one", "Text two", "Text three"]
    inputs = tokenizer(texts, padding=True, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    assert outputs.logits.shape == (3, 2)  # batch_size=3, num_labels=2


def test_model_save_load(tiny_model_name, tmp_path):
    """Test model save and load"""
    model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)

    save_path = tmp_path / "model"
    model.save_pretrained(save_path)

    # Load the saved model
    loaded_model = AutoModelForSequenceClassification.from_pretrained(save_path)
    assert loaded_model is not None

    # Compare outputs
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    text = "Test text"
    inputs = tokenizer(text, return_tensors="pt")

    model.eval()
    loaded_model.eval()

    with torch.no_grad():
        outputs1 = model(**inputs)
        outputs2 = loaded_model(**inputs)

    assert torch.allclose(outputs1.logits, outputs2.logits, atol=1e-5)
