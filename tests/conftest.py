"""Pytest configuration and fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForSequenceClassification, BertConfig


class TinyTokenizer:
    """Minimal whitespace tokenizer for tests."""

    def __init__(self) -> None:
        base_vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self.vocab: dict[str, int] = base_vocab.copy()
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self._next_id = len(self.vocab)
        self.max_vocab_size = 512

        common_tokens = {
            "this",
            "is",
            "a",
            "test",
            "training",
            "example",
            "text",
            "one",
            "two",
            "three",
            "hello,",
            "world!",
            "short",
            "much",
            "longer",
            "word",
            "another",
            "positive",
            "negative",
            "with",
            "lora",
            "batch",
            "model",
            "save",
            "load",
        }

        for token in common_tokens:
            self._get_token_id(token)

    @property
    def vocab_size(self) -> int:
        return self.max_vocab_size

    def encode(
        self, text: str, max_length: int | None = None, truncation: bool = False
    ) -> list[int]:
        token_ids = [self._get_token_id(token) for token in self._tokenize(text)]
        token_ids = [self.vocab["[CLS]"]] + token_ids + [self.vocab["[SEP]"]]

        if truncation and max_length is not None:
            token_ids = token_ids[:max_length]

        return token_ids

    def __call__(
        self,
        texts: str | Iterable[str],
        *,
        return_tensors: str | None = None,
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
    ) -> dict[str, torch.Tensor] | dict[str, list[list[int]]]:
        is_single = isinstance(texts, str)
        inputs = [texts] if is_single else list(texts)

        encoded = [
            self.encode(text, max_length=max_length, truncation=truncation) for text in inputs
        ]
        max_len = max(len(ids) for ids in encoded)

        should_pad = bool(padding) or (return_tensors == "pt" and len(encoded) > 1)

        if should_pad:
            padded = [self._pad(ids, max_len) for ids in encoded]
            masks = [self._mask(ids, max_len) for ids in encoded]
        else:
            padded = encoded
            masks = [self._mask(ids, len(ids)) for ids in encoded]

        if return_tensors == "pt":
            input_ids = torch.tensor(padded, dtype=torch.long)
            attention_mask = torch.tensor(masks, dtype=torch.long)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

        return {"input_ids": padded, "attention_mask": masks}

    def decode(self, token_ids: Iterable[int]) -> str:
        tokens = [self.id_to_token.get(idx, "[UNK]") for idx in token_ids]
        filtered = [tok for tok in tokens if tok not in {"[PAD]", "[CLS]", "[SEP]"}]
        return " ".join(filtered)

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def _get_token_id(self, token: str) -> int:
        if token not in self.vocab:
            if self._next_id >= self.max_vocab_size:
                raise ValueError("Tokenizer vocabulary exceeded capacity")
            self.vocab[token] = self._next_id
            self.id_to_token[self._next_id] = token
            self._next_id += 1
        return self.vocab[token]

    def _pad(self, ids: list[int], max_len: int) -> list[int]:
        padding = [self.vocab["[PAD]"]] * (max_len - len(ids))
        return ids + padding

    def _mask(self, ids: list[int], max_len: int) -> list[int]:
        valid_length = min(len(ids), max_len)
        return [1] * valid_length + [0] * (max_len - valid_length)


@pytest.fixture(scope="session", autouse=True)
def stub_transformers():
    """Stub Hugging Face auto-loaders to avoid network downloads."""

    tokenizer_instance = TinyTokenizer()
    patch = pytest.MonkeyPatch()

    original_model_from_pretrained = AutoModelForSequenceClassification.from_pretrained

    def _fake_tokenizer_from_pretrained(*_: object, **__: object) -> TinyTokenizer:
        return tokenizer_instance

    def _fake_model_from_pretrained(
        identifier: str | os.PathLike[str],
        *,
        num_labels: int = 2,
        **kwargs: object,
    ):
        model_path = Path(identifier)
        if model_path.is_dir() and (model_path / "config.json").exists():
            return original_model_from_pretrained(identifier, num_labels=num_labels, **kwargs)

        config = BertConfig(
            vocab_size=tokenizer_instance.vocab_size,
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=256,
            num_labels=num_labels,
        )
        return AutoModelForSequenceClassification.from_config(config)

    patch.setattr("transformers.AutoTokenizer.from_pretrained", _fake_tokenizer_from_pretrained)
    patch.setattr(
        "transformers.AutoModelForSequenceClassification.from_pretrained",
        _fake_model_from_pretrained,
    )

    yield

    patch.undo()


@pytest.fixture(scope="session")
def tiny_model_name():
    """Provide an identifier for the stubbed tiny model."""
    return "tiny-sequence-classifier"


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
