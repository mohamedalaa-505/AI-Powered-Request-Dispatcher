"""Post-training evaluation/reporting.

Reproduces the notebook's `analyze_and_plot_results` (per-class accuracy table
+ confusion matrix heatmap) and `predict_category` qualitative smoke test as a
standalone script you can run in CI or after a training job, against the
artifacts a training run produced.

Usage:
    python -m src.evaluate --model-dir model_artifacts
"""
import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# A small default set of qualitative smoke-test prompts, one per category.
# Override with --test-examples to point at your own JSON file of the same shape.
DEFAULT_TEST_EXAMPLES = {
    "q_and_a": [
        {"instruction": "Explain the main stages of the water cycle.", "expected": "q_and_a"},
    ],
    "classification": [
        {"instruction": "Is a tomato a fruit or a vegetable?", "expected": "classification"},
    ],
    "information_distillation": [
        {"instruction": "Summarize the main arguments of this article on climate change.", "expected": "information_distillation"},
    ],
    "brainstorming": [
        {"instruction": "Brainstorm five ideas for a birthday party theme.", "expected": "brainstorming"},
    ],
    "creative_writing": [
        {"instruction": "Write a short story about a dragon who is afraid of fire.", "expected": "creative_writing"},
    ],
}


def predict_category(model, tokenizer, text, device, id2cat):
    """Single-instance inference, matching the reference predict_category."""
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_id = torch.argmax(outputs.logits, dim=-1).item()
    return id2cat[predicted_id]


def render_confusion_matrix(confusion_matrix, id2cat, output_path):
    """Saves a seaborn heatmap of the confusion matrix, matching analyze_and_plot_results."""
    class_names = [name for _, name in sorted(id2cat.items())]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title("Confusion Matrix for Dispatcher Model")
    plt.xlabel("Predicted Category")
    plt.ylabel("True Category")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to %s", output_path)


def print_per_class_accuracy(confusion_matrix, id2cat):
    """Prints the per-class accuracy table (diagonal / row sum), same math as the reference."""
    correct = confusion_matrix.diagonal()
    totals = confusion_matrix.sum(axis=1)
    accuracy = correct / (totals + 1e-9)

    print("\nPer-Class Accuracy")
    print(f"{'Category':<28}{'Accuracy':>10}")
    for class_id, acc in enumerate(accuracy):
        print(f"{id2cat[class_id]:<28}{acc:>9.2%}")
    print()


def run_qualitative_tests(model, tokenizer, device, id2cat, test_examples):
    print("--- Qualitative Test: Dispatcher on Held-Out Prompts ---\n")
    correct, total = 0, 0
    for category, examples in test_examples.items():
        print(f"--- Category: {category} ---")
        for example in examples:
            predicted = predict_category(model, tokenizer, example["instruction"], device, id2cat)
            expected = example["expected"]
            total += 1
            correct += int(predicted == expected)
            mark = "✓" if predicted == expected else "✗"
            print(f"  {mark} '{example['instruction'][:60]}...' -> predicted={predicted}, expected={expected}")
    print(f"\nQualitative accuracy: {correct}/{total}\n")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained AI Request Dispatcher")
    parser.add_argument("--model-dir", default="model_artifacts")
    parser.add_argument("--test-examples", default=None, help="Optional path to a JSON file shaped like DEFAULT_TEST_EXAMPLES")
    parser.add_argument("--output-plot", default=None, help="Where to save the confusion matrix PNG (defaults to <model-dir>/confusion_matrix.png)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.model_dir, "id2cat.json")) as f:
        id2cat = {int(k): v for k, v in json.load(f).items()}

    metrics_path = os.path.join(args.model_dir, "training_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        cm = torch.tensor(metrics["confusion_matrix"]).numpy()
        print(f"Val Loss: {metrics['val_loss']:.4f}  Val Accuracy: {metrics['val_accuracy']:.4f}  Val F1: {metrics['val_f1']:.4f}")
        print_per_class_accuracy(cm, id2cat)
        output_plot = args.output_plot or os.path.join(args.model_dir, "confusion_matrix.png")
        render_confusion_matrix(cm, id2cat, output_plot)
    else:
        logger.warning("No training_metrics.json found in %s -- skipping confusion matrix report", args.model_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)

    if args.test_examples:
        with open(args.test_examples) as f:
            test_examples = json.load(f)
    else:
        test_examples = DEFAULT_TEST_EXAMPLES

    run_qualitative_tests(model, tokenizer, device, id2cat, test_examples)


if __name__ == "__main__":
    main()
