"""Model loading and partial fine-tuning (parameter-efficient) utilities."""
import logging
import os
from typing import Tuple

from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertForSequenceClassification,
)

logger = logging.getLogger(__name__)

NUM_TRANSFORMER_LAYERS = 6  # DistilBERT has 6 transformer layers


def download_base_model(model_name: str = "distilbert-base-uncased", local_path: str = "./distilbert-local-base") -> None:
    """Downloads a base transformer model + tokenizer (no classification head)
    from the Hub and caches it locally, skipping the download if already cached.

    Mirrors the original notebook's `download_bert` — separating "pull the base
    weights once" from "attach a fresh classification head" makes training runs
    reproducible and avoids re-hitting the Hub (or failing outright) once the
    model is cached, which matters for CI runners and air-gapped training jobs.
    """
    if os.path.isdir(local_path) and os.listdir(local_path):
        logger.info("Base model '%s' already available at %s", model_name, local_path)
        return

    logger.info("Downloading base model '%s' to %s...", model_name, local_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)
    logger.info("Base model downloaded and saved successfully.")


def load_bert(local_path: str, num_classes: int) -> Tuple[DistilBertForSequenceClassification, AutoTokenizer]:
    """Loads a base transformer model from a local directory and attaches a
    fresh sequence classification head with `num_classes` outputs."""
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_path, num_labels=num_classes
    )
    return model, tokenizer


def partially_freeze_bert_layers(
    model: DistilBertForSequenceClassification, layers_to_train: int = 3
) -> DistilBertForSequenceClassification:
    """Freezes all but the last N transformer layers + classification head.

    Args:
        model: a DistilBertForSequenceClassification instance.
        layers_to_train: number of final transformer layers to leave trainable
            (clamped to [0, NUM_TRANSFORMER_LAYERS]).

    Returns:
        The same model instance, modified in place.
    """
    layers_to_train = max(0, min(layers_to_train, NUM_TRANSFORMER_LAYERS))

    # Freeze everything first.
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the last `layers_to_train` transformer layers.
    first_trainable_layer = NUM_TRANSFORMER_LAYERS - layers_to_train
    for name, param in model.named_parameters():
        if name.startswith("distilbert.transformer.layer."):
            layer_idx = int(name.split(".")[3])
            if layer_idx >= first_trainable_layer:
                param.requires_grad = True

    # Always train the classification head.
    for name, param in model.named_parameters():
        if name.startswith("pre_classifier") or name.startswith("classifier"):
            param.requires_grad = True

    return model
