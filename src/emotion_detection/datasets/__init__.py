"""Data preparation APIs; importing them never loads or trains a model."""

from emotion_detection.datasets.reference import (
    load_prepared_reference_dataset,
    load_reference_dataset,
    prepare_reference_dataset,
)
from emotion_detection.datasets.reference_types import (
    CrosslingualReferenceDataset,
    ReferenceDataset,
)
from emotion_detection.datasets.sources import available_reference_languages

__all__ = [
    "CrosslingualReferenceDataset",
    "ReferenceDataset",
    "available_reference_languages",
    "load_prepared_reference_dataset",
    "load_reference_dataset",
    "prepare_reference_dataset",
]
