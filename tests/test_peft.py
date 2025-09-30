"""Test PEFT (LoRA) integration"""

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test_lora_model_creation(tiny_model_name):
    """Test LoRA model wrapping"""
    base_model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)
    assert model is not None


def test_lora_trainable_parameters(tiny_model_name):
    """Test that LoRA reduces trainable parameters"""
    base_model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)

    total_params_before = sum(p.numel() for p in base_model.parameters())
    trainable_params_before = sum(p.numel() for p in base_model.parameters() if p.requires_grad)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)

    trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # LoRA should significantly reduce trainable parameters
    assert trainable_params_after < trainable_params_before
    assert trainable_params_after < total_params_before * 0.1  # <10% trainable


def test_lora_forward_pass(tiny_model_name):
    """Test forward pass with LoRA model"""
    base_model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

    text = "Test with LoRA"
    inputs = tokenizer(text, return_tensors="pt")

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    assert hasattr(outputs, "logits")
    assert outputs.logits.shape == (1, 2)


def test_lora_training_step(tiny_model_name):
    """Test training step with LoRA"""
    base_model = AutoModelForSequenceClassification.from_pretrained(tiny_model_name, num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )

    model = get_peft_model(base_model, lora_config)
    tokenizer = AutoTokenizer.from_pretrained(tiny_model_name)

    text = "Training with LoRA"
    inputs = tokenizer(text, return_tensors="pt")
    labels = torch.tensor([1])

    model.train()
    outputs = model(**inputs, labels=labels)

    assert hasattr(outputs, "loss")
    outputs.loss.backward()

    # Verify only LoRA parameters have gradients
    has_lora_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None
            if "lora" in name.lower():
                has_lora_grad = True

    assert has_lora_grad, "LoRA parameters should have gradients"
