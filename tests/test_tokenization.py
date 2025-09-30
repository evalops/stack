"""Test tokenization and data preprocessing"""

from transformers import AutoTokenizer


def test_tokenizer_load(tiny_model_name):
    """Test that tokenizer loads correctly"""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    assert tokenizer is not None
    assert hasattr(tokenizer, "encode")
    assert hasattr(tokenizer, "decode")


def test_tokenization(tiny_model_name):
    """Test basic tokenization"""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    text = "Hello, world!"

    encoded = tokenizer(text, return_tensors="pt")
    assert "input_ids" in encoded
    assert "attention_mask" in encoded
    assert encoded["input_ids"].shape[0] == 1  # batch size


def test_batch_tokenization(tiny_model_name):
    """Test batch tokenization with padding"""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    texts = ["Short text", "This is a much longer text"]

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

    assert encoded["input_ids"].shape[0] == 2  # batch size
    assert encoded["attention_mask"].shape == encoded["input_ids"].shape


def test_tokenization_max_length(tiny_model_name):
    """Test truncation works correctly"""
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)
    long_text = "word " * 1000

    encoded = tokenizer(long_text, max_length=32, truncation=True, return_tensors="pt")

    assert encoded["input_ids"].shape[1] <= 32
