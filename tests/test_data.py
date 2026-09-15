import pandas as pd
import pytest
import torch

from src.data import (
    CATEGORY_MAP,
    InstructionDataset,
    create_data_collator,
    create_dataloaders,
    create_dataset_splits,
    load_and_prepare_data,
)


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame({
        "instruction": [
            "What is the capital of France?",
            "Explain quantum computing to a 5 year old",
            "Is a tomato a fruit or vegetable?",
            "Summarize this article about climate change",
            "Extract all dates from this document",
            "Brainstorm ideas for a birthday party",
            "Write a short story about a dragon",
            "Who wrote Romeo and Juliet?",
        ],
        "category": [
            "open_qa", "general_qa", "classification", "summarization",
            "information_extraction", "brainstorming", "creative_writing", "closed_qa",
        ],
    })
    path = tmp_path / "sample.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_and_prepare_data_merges_categories(sample_csv):
    df, cat2id, id2cat = load_and_prepare_data(sample_csv)

    # None of the merged-away category names should remain.
    remaining = set(df["category"].unique())
    for old_name in CATEGORY_MAP:
        assert old_name not in remaining

    assert "q_and_a" in remaining
    assert "information_distillation" in remaining
    assert len(cat2id) == len(id2cat)
    assert all(id2cat[cat2id[c]] == c for c in cat2id)


def test_load_and_prepare_data_requires_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1, 2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        load_and_prepare_data(str(bad_csv))


# A tiny, fast tokenizer built from the standard BERT vocab isn't available
# offline, so these tests use a hand-rolled fake tokenizer with the same
# interface subset InstructionDataset relies on.
class FakeTokenizer:
    def __call__(self, text, truncation=True, max_length=128):
        ids = [ord(c) % 100 for c in text[:max_length]]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, encoded_inputs, return_tensors=None):
        max_len = max(len(e["input_ids"]) for e in encoded_inputs)
        input_ids, attn = [], []
        for e in encoded_inputs:
            pad_len = max_len - len(e["input_ids"])
            input_ids.append(e["input_ids"] + [0] * pad_len)
            attn.append(e["attention_mask"] + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attn),
        }


def test_instruction_dataset_len_and_getitem():
    texts = ["hello world", "a much longer sentence here"]
    labels = [0, 1]
    ds = InstructionDataset(texts, labels, FakeTokenizer())

    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item and "attention_mask" in item
    assert item["labels"].item() == 0


def test_dataset_split_sizes():
    texts = [f"instruction {i}" for i in range(100)]
    labels = [i % 3 for i in range(100)]
    ds = InstructionDataset(texts, labels, FakeTokenizer())

    train_ds, val_ds = create_dataset_splits(ds, train_split_percentage=0.8, seed=42)
    assert len(train_ds) == 80
    assert len(val_ds) == 20
    assert len(train_ds) + len(val_ds) == len(ds)
