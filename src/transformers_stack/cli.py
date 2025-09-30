"""Command-line interface for orchestrating training runs."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from pathlib import Path

import evaluate
from datasets import load_dataset
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and execute the training workflow."""

    parser = argparse.ArgumentParser(
        description="Run fine-tuning jobs using the Transformers Stack Hydra configs.",
    )
    parser.add_argument(
        "-c",
        "--config",
        default="config",
        help="Name of the Hydra config to load (default: %(default)s).",
    )
    parser.add_argument(
        "--config-dir",
        default="conf",
        help="Path to the Hydra configuration directory (default: %(default)s).",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        default=[],
        metavar="OVERRIDE",
        help="Hydra override, e.g. model.name=distilbert-base-uncased. Can be repeated.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration and exit without training.",
    )

    args = parser.parse_args(argv)

    config_dir = Path(args.config_dir).expanduser().resolve()
    if not config_dir.exists():
        parser.error(f"Config directory '{config_dir}' does not exist")

    cfg = _load_config(
        config_dir=config_dir,
        config_name=args.config,
        overrides=args.override,
    )

    print("Resolved configuration:\n" + OmegaConf.to_yaml(cfg, resolve=True))

    if args.dry_run:
        print("Dry-run requested; exiting without training")
        return

    _run_training(cfg)


def _load_config(config_dir: Path, config_name: str, overrides: Iterable[str]) -> DictConfig:
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return cfg


def _run_training(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting training run")

    set_seed(cfg.seed)

    model_cfg = cfg.model
    data_cfg = cfg.data
    train_cfg = cfg.train
    eval_cfg = cfg.eval

    tokenizer = AutoTokenizer.from_pretrained(model_cfg.name)

    text_field = getattr(data_cfg, "text_field", "text")
    label_field = getattr(data_cfg, "label_field", "label")

    dataset_config_name = getattr(data_cfg, "dataset_config_name", None)
    train_load_args = {
        "path": data_cfg.dataset_name,
        "split": data_cfg.train_split,
    }
    eval_load_args = {
        "path": data_cfg.dataset_name,
        "split": data_cfg.eval_split,
    }
    if dataset_config_name:
        train_load_args["name"] = dataset_config_name
        eval_load_args["name"] = dataset_config_name

    logger.info("Loading dataset '%s'", data_cfg.dataset_name)
    train_dataset = load_dataset(**train_load_args)
    eval_dataset = load_dataset(**eval_load_args)

    padding = data_cfg.padding
    truncation = bool(data_cfg.truncation)
    max_length = data_cfg.max_length

    def preprocess_batch(batch: dict[str, list[str]]):
        texts = batch[text_field]
        return tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

    remove_columns_train = [c for c in train_dataset.column_names if c != label_field]
    remove_columns_eval = [c for c in eval_dataset.column_names if c != label_field]

    logger.info("Tokenizing train dataset")
    tokenized_train = train_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=remove_columns_train,
    )
    logger.info("Tokenizing eval dataset")
    tokenized_eval = eval_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=remove_columns_eval,
    )

    if label_field in tokenized_train.column_names:
        tokenized_train = tokenized_train.rename_column(label_field, "labels")
    if label_field in tokenized_eval.column_names:
        tokenized_eval = tokenized_eval.rename_column(label_field, "labels")

    tokenized_train.set_format("torch")
    tokenized_eval.set_format("torch")

    logger.info("Loading model '%s'", model_cfg.name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.name,
        num_labels=model_cfg.num_labels,
    )

    if getattr(model_cfg, "use_peft", False):
        peft_cfg = model_cfg.peft
        lora_config = LoraConfig(
            r=peft_cfg.r,
            lora_alpha=peft_cfg.lora_alpha,
            target_modules=list(peft_cfg.target_modules),
            lora_dropout=peft_cfg.lora_dropout,
            bias=peft_cfg.bias,
            task_type="SEQ_CLS",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if train_cfg.mixed_precision in {"fp16", "bf16"} else None,
    )

    metrics = {metric_name: evaluate.load(metric_name) for metric_name in eval_cfg.metrics}

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)

        results = {}
        for metric_name, metric in metrics.items():
            kwargs = {}
            if metric_name in {"precision", "recall", "f1"}:
                kwargs["average"] = "binary"
            metric_result = metric.compute(
                predictions=predictions,
                references=labels,
                **kwargs,
            )
            results.update(metric_result)
        return results

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = "steps" if eval_cfg.eval_steps and eval_cfg.eval_steps > 0 else "no"
    save_strategy = "steps" if train_cfg.save_steps and train_cfg.save_steps > 0 else "no"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        num_train_epochs=train_cfg.epochs,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=eval_cfg.batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        eval_strategy=eval_strategy,
        eval_steps=eval_cfg.eval_steps if eval_strategy == "steps" else None,
        logging_steps=max(1, eval_cfg.eval_steps // 2) if eval_strategy == "steps" else 10,
        save_strategy=save_strategy,
        save_steps=train_cfg.save_steps if save_strategy == "steps" else None,
        save_total_limit=train_cfg.save_total_limit,
        load_best_model_at_end=train_cfg.save_best,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        warmup_steps=train_cfg.warmup_steps,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        gradient_checkpointing=model_cfg.gradient_checkpointing,
        fp16=train_cfg.mixed_precision == "fp16",
        bf16=train_cfg.mixed_precision == "bf16",
        max_grad_norm=train_cfg.max_grad_norm,
        report_to=None,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    logger.info("Commencing training")
    trainer.train()

    logger.info("Final evaluation")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        logger.info("%s = %s", key, value)

    logger.info("Saving artifacts to %s", output_dir)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def entrypoint() -> None:
    """Console-script entrypoint."""

    main()


if __name__ == "__main__":
    entrypoint()
