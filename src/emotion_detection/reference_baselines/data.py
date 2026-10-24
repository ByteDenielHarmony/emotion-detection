"""Validation and tokenization of already-loaded official-format data.

This module deliberately performs no downloads, file reads, or split preparation.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

EMOTIONS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")


def validate_label_names(label_names: Sequence[str]) -> tuple[str, ...]:
    names = tuple(label_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Emotion names must be nonempty and unique")
    if not set(names).issubset(EMOTIONS):
        raise ValueError(f"Emotion names must be selected from {EMOTIONS}")
    return names


def infer_label_names(frame: pd.DataFrame) -> tuple[str, ...]:
    """Use known emotion columns, never arbitrary metadata columns."""
    return validate_label_names([name for name in EMOTIONS if name in frame])


def validate_frame(
    frame: pd.DataFrame,
    label_names: Sequence[str],
    *,
    require_labels: bool = True,
    allow_empty: bool = False,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Expected an already-loaded pandas DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("DataFrame columns must be unique")
    if frame.empty and not allow_empty:
        raise ValueError("Training and evaluation data must not be empty")
    required = {"id", "text"}
    if require_labels:
        required.update(label_names)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if frame["id"].isna().any() or not frame["id"].is_unique:
        raise ValueError("IDs must be non-null and unique within a split")
    if (
        frame["id"]
        .map(lambda value: isinstance(value, str) and not value.strip())
        .any()
    ):
        raise ValueError("IDs must not be blank")
    if (
        not frame["text"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError("Texts must be nonempty strings")
    if require_labels and not frame[list(label_names)].isin([0, 1]).all().all():
        raise ValueError("Track A labels must be present and binary (0 or 1)")


def validate_partition_separation(
    train: pd.DataFrame, validation: pd.DataFrame
) -> None:
    """Reject overlaps, but leave cleaning and split construction to the caller."""
    if set(train["id"]) & set(validation["id"]):
        raise ValueError("Training and validation IDs overlap")
    if set(train["text"].str.lower()) & set(validation["text"].str.lower()):
        raise ValueError("Training and validation texts overlap ignoring case")


class TokenizedEmotionDataset(Dataset):
    """An in-memory adapter reusable with binary or future intensity targets.

    Target validation belongs to the track-specific caller. No missing target is
    imputed. Tokenization preserves text case and pads dynamically in the collator.
    """

    def __init__(self, texts, tokenizer, *, labels=None, max_length=256):
        self.texts = tuple(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = None
        if labels is not None:
            targets = np.array(labels, dtype=np.float32, copy=True)
            if targets.ndim != 2 or len(targets) != len(self.texts):
                raise ValueError("Targets must have shape (number of texts, emotions)")
            self.labels = torch.from_numpy(targets)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        item = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        if self.labels is not None:
            item["labels"] = self.labels[index]
        return item


class EmotionDataCollator:
    """Pad token sequences separately from fixed-width emotion target vectors."""

    def __init__(self, tokenizer):
        self.padding = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        labeled = ["labels" in item for item in features]
        if any(labeled) and not all(labeled):
            raise ValueError("A batch cannot mix labeled and unlabeled examples")
        batch = self.padding(
            [
                {key: value for key, value in item.items() if key != "labels"}
                for item in features
            ]
        )
        if all(labeled):
            batch["labels"] = torch.stack([item["labels"] for item in features])
        return batch
