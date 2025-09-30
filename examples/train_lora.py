"""
LoRA Fine-tuning Example

This script demonstrates how to fine-tune a transformer model using LoRA
(Low-Rank Adaptation) with the Accelerate library for distributed training.
"""

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    acc = Accelerator()

    if acc.is_main_process:
        print("Loading dataset...")
    ds = load_dataset("imdb", split="train[:1%]")

    if acc.is_main_process:
        print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if acc.is_main_process:
        print("Tokenizing dataset...")
    enc = ds.map(
        lambda ex: tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)

    if acc.is_main_process:
        model.print_trainable_parameters()

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    dataloader = torch.utils.data.DataLoader(enc.with_format("torch"), batch_size=8, shuffle=True)

    model, optimizer, dataloader = acc.prepare(model, optimizer, dataloader)

    num_epochs = 3
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
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

                if i % 10 == 0 and acc.is_main_process:
                    print(f"Epoch {epoch + 1}/{num_epochs} | Step {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        if acc.is_main_process:
            print(f"Epoch {epoch + 1}/{num_epochs} | Average Loss: {avg_loss:.4f}")

    if acc.is_main_process:
        print("Training complete!")
        print("Saving model...")
        acc.unwrap_model(model).save_pretrained("./output/lora_model")
        tokenizer.save_pretrained("./output/lora_model")
        print("Model saved to ./output/lora_model")


if __name__ == "__main__":
    main()
