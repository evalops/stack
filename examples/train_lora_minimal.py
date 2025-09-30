"""
Minimal LoRA training for testing (very small dataset, few steps)
"""

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    set_seed,
)

set_seed(42)

print("Initializing...")
acc = Accelerator()

# Use tiny dataset for quick test
print("Loading tiny dataset...")
ds = load_dataset("imdb", split="train[:20]")

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

print("Tokenizing...")
enc = ds.map(
    lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=128),
    batched=True,
)

print("Setting up LoRA...")
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    target_modules=["q_lin", "v_lin"],
    lora_dropout=0.1,
    bias="none",
    task_type="SEQ_CLS",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
dataloader = torch.utils.data.DataLoader(enc.with_format("torch"), batch_size=4, shuffle=True)

model, optimizer, dataloader = acc.prepare(model, optimizer, dataloader)

print("Training for 1 epoch (5 steps max)...")
model.train()
total_loss = 0

for i, batch in enumerate(dataloader):
    if i >= 5:  # Only train for 5 steps
        break

    with acc.accumulate(model):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        loss = outputs.loss
        total_loss += loss.item()

        acc.backward(loss)
        optimizer.step()
        optimizer.zero_grad()

        print(f"Step {i+1}/5 | Loss: {loss.item():.4f}")

avg_loss = total_loss / min(len(dataloader), 5)
print(f"Average Loss: {avg_loss:.4f}")
print("✅ Training test completed successfully!")
