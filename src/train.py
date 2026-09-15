"""End-to-end training script for the AI Request Dispatcher.

Mirrors the logic in the original notebook's helper_utils.py exactly
(torchmetrics macro-F1 + confusion matrix, AdamW with weight_decay=0.01,
metrics reported from the final epoch) so results match what the notebook
would produce, while adding CLI args, logging, and artifact saving suited to
a repeatable production run.

Usage:
    python -m src.train --data-path data/databricks-dolly-15k.csv
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.utils.class_weight import compute_class_weight
from tqdm.auto import tqdm

from src.config import settings
from src.data import (
    InstructionDataset,
    create_data_collator,
    create_dataloaders,
    create_dataset_splits,
    load_and_prepare_data,
)
from src.model import download_base_model, load_bert, partially_freeze_bert_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def calculate_class_weights(train_dataset, device) -> torch.Tensor:
    """Computes balanced class weights from the training subset's labels."""
    train_labels_list = [train_dataset.dataset.labels[i] for i in train_dataset.indices]
    classes = np.unique(train_labels_list)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=train_labels_list
    )
    return torch.tensor(weights, dtype=torch.float).to(device)


def training_loop(model, train_loader, val_loader, loss_function, learning_rate, num_epochs, device):
    """Trains and validates `model`, returning it plus the final epoch's metrics.

    Matches the reference implementation: AdamW(weight_decay=0.01) over all
    model parameters, macro-averaged F1, and a confusion matrix computed over
    the full validation set on the last epoch. No checkpoint selection --
    the returned model is whatever the last epoch produced, same as the
    original training_loop.
    """
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    num_classes = model.config.num_labels
    val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    val_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    history = []
    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_loop:
        # --- Training phase ---
        model.train()
        train_loss_epoch = 0.0
        train_inner_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Training", leave=False)
        for batch in train_inner_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)

            train_loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            train_inner_loop.set_postfix(loss=loss.item())

        train_loss_epoch /= len(train_loader)

        # --- Validation phase ---
        model.eval()
        val_loss_epoch = 0.0
        val_inner_loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} Validation", leave=False)
        with torch.no_grad():
            for batch in val_inner_loop:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_function(outputs.logits, labels)
                val_loss_epoch += loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                val_accuracy.update(preds, labels)
                val_f1.update(preds, labels)
                val_cm.update(preds, labels)

        val_loss_epoch /= len(val_loader)

        epoch_acc = val_accuracy.compute()
        epoch_f1 = val_f1.compute()
        val_accuracy.reset()
        val_f1.reset()

        epoch_loop.set_postfix(
            train_loss=f"{train_loss_epoch:.4f}",
            val_loss=f"{val_loss_epoch:.4f}",
            val_acc=f"{epoch_acc:.4f}",
        )
        tqdm.write(
            f"Epoch {epoch + 1} Metrics -> "
            f"Train Loss: {train_loss_epoch:.4f}, "
            f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {epoch_acc:.4f}, Val F1: {epoch_f1:.4f}"
        )
        history.append({
            "epoch": epoch + 1, "train_loss": train_loss_epoch,
            "val_loss": val_loss_epoch, "val_accuracy": epoch_acc.item(), "val_f1": epoch_f1.item(),
        })

    logger.info("--- Training complete ---")
    final_confusion_matrix = val_cm.compute()

    final_results = {
        "val_loss": val_loss_epoch,
        "val_accuracy": epoch_acc.item(),
        "val_f1": epoch_f1.item(),
        "confusion_matrix": final_confusion_matrix.cpu(),
        "history": history,
    }
    return model, final_results


def evaluate_on_test_set(model, test_loader, loss_function, device):
    """Runs a single evaluation pass over the held-out test set.

    Only ever called once, after training and checkpoint selection are done
    -- never used to pick epochs/hyperparameters. That's what val is for.
    """
    model.to(device)
    model.eval()

    num_classes = model.config.num_labels
    test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes).to(device)
    test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro").to(device)
    test_cm = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=num_classes).to(device)

    test_loss_total = 0.0
    test_loop = tqdm(test_loader, desc="Final Test Evaluation", leave=False)
    with torch.no_grad():
        for batch in test_loop:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_function(outputs.logits, labels)
            test_loss_total += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            test_accuracy.update(preds, labels)
            test_f1.update(preds, labels)
            test_cm.update(preds, labels)

    return {
        "test_loss": test_loss_total / len(test_loader),
        "test_accuracy": test_accuracy.compute().item(),
        "test_f1": test_f1.compute().item(),
        "confusion_matrix": test_cm.compute().cpu(),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train the AI Request Dispatcher")
    parser.add_argument("--data-path", required=True, help="Path to the augmented Dolly-15k CSV")
    parser.add_argument("--output-dir", default="model_artifacts")
    parser.add_argument("--base-model", default=settings.base_model_name)
    parser.add_argument(
        "--base-model-cache-dir", default="./distilbert-local-base",
        help="Local cache dir for the downloaded base model (avoids re-hitting the Hub on every run)",
    )
    parser.add_argument("--layers-to-train", type=int, default=settings.default_layers_to_train)
    parser.add_argument("--learning-rate", type=float, default=settings.default_learning_rate)
    parser.add_argument("--num-epochs", type=int, default=settings.default_num_epochs)
    parser.add_argument("--batch-size", type=int, default=settings.default_batch_size)
    parser.add_argument("--train-split", type=float, default=settings.train_split_percentage)
    parser.add_argument(
        "--val-split", type=float,
        default=getattr(settings, "val_split_percentage", 0.1),
        help="Fraction held out for validation; remainder after --train-split "
             "and --val-split becomes the held-out test set",
    )
    parser.add_argument("--seed", type=int, default=settings.random_seed)
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    logger.info("Loading data from %s", args.data_path)
    df, cat2id, id2cat = load_and_prepare_data(args.data_path)
    texts, labels = df["instruction"].tolist(), df["label"].tolist()
    num_classes = len(cat2id)
    logger.info("Loaded %d samples across %d classes: %s", len(texts), num_classes, cat2id)

    download_base_model(args.base_model, args.base_model_cache_dir)
    model, tokenizer = load_bert(args.base_model_cache_dir, num_classes=num_classes)

    full_dataset = InstructionDataset(texts, labels, tokenizer)
    train_dataset, val_dataset, test_dataset = create_dataset_splits(
        full_dataset, args.train_split, args.val_split, seed=args.seed
    )
    logger.info(
        "Train samples: %d | Val samples: %d | Test samples: %d",
        len(train_dataset), len(val_dataset), len(test_dataset),
    )

    collator = create_data_collator(tokenizer)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, args.batch_size, collator
    )

    class_weights = calculate_class_weights(train_dataset, device)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)

    model = partially_freeze_bert_layers(model, args.layers_to_train)

    trained_model, results = training_loop(
        model, train_loader, val_loader, loss_function,
        args.learning_rate, args.num_epochs, device,
    )
    print("\nFinal Validation Metrics")
    print(f"  Loss:     {results['val_loss']:.4f}")
    print(f"  Accuracy: {results['val_accuracy']:.4f}")
    print(f"  F1:       {results['val_f1']:.4f}\n")

    logger.info("Running final evaluation on held-out test set...")
    test_results = evaluate_on_test_set(trained_model, test_loader, loss_function, device)
    print("Final Test Metrics (held out, never seen during training or tuning)")
    print(f"  Loss:     {test_results['test_loss']:.4f}")
    print(f"  Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  F1:       {test_results['test_f1']:.4f}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    trained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "id2cat.json"), "w") as f:
        json.dump(id2cat, f, indent=2)

    metrics_out = {k: v for k, v in results.items() if k not in ("confusion_matrix",)}
    metrics_out["confusion_matrix"] = results["confusion_matrix"].tolist()
    metrics_out["seed"] = args.seed
    metrics_out["train_split"] = args.train_split
    metrics_out["val_split"] = args.val_split
    metrics_out["test_loss"] = test_results["test_loss"]
    metrics_out["test_accuracy"] = test_results["test_accuracy"]
    metrics_out["test_f1"] = test_results["test_f1"]
    metrics_out["test_confusion_matrix"] = test_results["confusion_matrix"].tolist()
    with open(os.path.join(args.output_dir, "training_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)

    logger.info("Artifacts saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
