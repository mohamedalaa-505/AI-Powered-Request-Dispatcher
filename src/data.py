"""Data pipeline: CSV -> cleaned DataFrame -> PyTorch Dataset/DataLoaders."""
from typing import Dict, List, Tuple

import pandas as pd
import torch
import transformers
from torch.utils.data import DataLoader, Dataset, random_split

# Consolidates the raw Dolly-15k categories into the dispatcher's target classes.
CATEGORY_MAP = {
    "general_qa": "q_and_a",
    "open_qa": "q_and_a",
    "closed_qa": "q_and_a",
    "information_extraction": "information_distillation",
    "summarization": "information_distillation",
}


def load_and_prepare_data(csv_path: str) -> Tuple[pd.DataFrame, Dict[str, int], Dict[int, str]]:
    """Loads the dataset, merges categories, and builds label mappings.

    Args:
        csv_path: path to the augmented Dolly-15k CSV. Must contain
            `instruction` and `category` columns.

    Returns:
        (df, cat2id, id2cat)
    """
    df = pd.read_csv(csv_path).dropna().reset_index(drop=True)

    if "instruction" not in df.columns or "category" not in df.columns:
        raise ValueError(
            f"Expected 'instruction' and 'category' columns, got {list(df.columns)}"
        )

    df["category"] = df["category"].replace(CATEGORY_MAP)

    unique_categories = df["category"].unique()
    cat2id = {category: i for i, category in enumerate(sorted(unique_categories))}
    id2cat = {i: category for category, i in cat2id.items()}

    df["label"] = df["category"].map(cat2id)

    return df, cat2id, id2cat


class InstructionDataset(Dataset):
    """Custom PyTorch Dataset for text classification.

    Stores raw texts and integer labels, tokenizing on the fly per sample so
    that padding can be deferred to the collator (dynamic padding). Matches
    the reference implementation: truncates at 128 tokens.
    """

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(text, truncation=True, max_length=self.max_length)
        item = {key: torch.tensor(val) for key, val in encoding.items()}
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def create_dataset_splits(
    full_dataset: Dataset,
    train_split_percentage: float = 0.8,
    val_split_percentage: float = 0.1,
    seed: int = 42,
):
    """Splits a dataset into train/val/test subsets with a fixed seed for reproducibility.

    `train_split_percentage` and `val_split_percentage` set the first two
    partitions; whatever's left (1 - train - val) becomes the held-out test
    set. Test is only ever used post-training, for a final unbiased read on
    model quality -- it must never influence checkpoint selection or
    hyperparameter choices the way val does.

    Args:
        full_dataset: the full InstructionDataset to split.
        train_split_percentage: fraction of data for training.
        val_split_percentage: fraction of data for validation.
        seed: seed for the split generator, for reproducibility.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if train_split_percentage + val_split_percentage >= 1.0:
        raise ValueError(
            "train_split_percentage + val_split_percentage must be < 1.0 "
            f"to leave room for a test split (got {train_split_percentage} + "
            f"{val_split_percentage})"
        )

    train_size = int(train_split_percentage * len(full_dataset))
    val_size = int(val_split_percentage * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )


def create_data_collator(tokenizer):
    """Returns a HF data collator that dynamically pads each batch."""
    return transformers.DataCollatorWithPadding(tokenizer)


def create_dataloaders(
    train_dataset, val_dataset, test_dataset, batch_size: int, collate_fn
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Builds train (shuffled) and val/test (unshuffled) DataLoaders."""
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return train_loader, val_loader, test_loader
