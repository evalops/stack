"""
Model Evaluation Script

Evaluate a fine-tuned model on a test set with multiple metrics.
"""

import argparse
import json
from pathlib import Path

import evaluate
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def evaluate_model(model_path, dataset_name="imdb", split="test[:10%]", batch_size=16):
    """Evaluate model on dataset"""

    print(f"Loading model from {model_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Loading dataset: {dataset_name} ({split})...")
    dataset = load_dataset(dataset_name, split=split)

    print("Tokenizing...")

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    dataloader = DataLoader(tokenized_dataset, batch_size=batch_size)

    print("Running evaluation...")
    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

            all_predictions.extend(predictions)
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities)

    print("\nComputing metrics...")
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")

    results = {
        "accuracy": accuracy_metric.compute(predictions=all_predictions, references=all_labels)[
            "accuracy"
        ],
        "f1": f1_metric.compute(predictions=all_predictions, references=all_labels)["f1"],
        "precision": precision_metric.compute(predictions=all_predictions, references=all_labels)[
            "precision"
        ],
        "recall": recall_metric.compute(predictions=all_predictions, references=all_labels)[
            "recall"
        ],
        "num_samples": len(all_predictions),
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name (default: imdb)")
    parser.add_argument(
        "--split", type=str, default="test[:10%]", help="Dataset split (default: test[:10%])"
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    results = evaluate_model(
        model_path=args.model_path,
        dataset_name=args.dataset,
        split=args.split,
        batch_size=args.batch_size,
    )

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, value in results.items():
        if metric != "num_samples":
            print(f"{metric.upper():20s}: {value:.4f}")
        else:
            print(f"{metric.upper():20s}: {value}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
