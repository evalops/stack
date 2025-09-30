"""
Fine-tuning with Hugging Face Trainer API

This example shows how to use the Trainer class for supervised fine-tuning
with built-in features like checkpointing, metrics, and evaluation.
"""

import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ACCURACY_METRIC = evaluate.load("accuracy")
F1_METRIC = evaluate.load("f1")


def compute_metrics(eval_pred):
    """Compute metrics for evaluation"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = ACCURACY_METRIC.compute(predictions=predictions, references=labels)
    f1 = F1_METRIC.compute(predictions=predictions, references=labels, average="binary")

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
    }


def main():
    # Set seed for reproducibility
    set_seed(42)

    model_name = "distilbert-base-uncased"

    print("Loading dataset...")
    dataset = load_dataset("imdb", split="train[:5%]")
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    print("Tokenizing dataset...")

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir="./output/trainer",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=10,
        save_total_limit=2,
        seed=42,
        report_to="none",  # Change to "wandb" or "mlflow" for tracking
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("\nFinal evaluation:")
    results = trainer.evaluate()
    print(results)

    print("\nSaving model...")
    trainer.save_model("./output/trainer/best_model")
    tokenizer.save_pretrained("./output/trainer/best_model")
    print("Training complete!")


if __name__ == "__main__":
    main()
