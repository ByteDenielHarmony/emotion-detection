"""Data preparation APIs; importing them never loads or trains a model."""

from emotion_detection.datasets.reference import load_reference_dataset
from emotion_detection.datasets.reference_types import ReferenceDataset
from emotion_detection.datasets.sources import available_reference_languages

__all__ = [
    "ReferenceDataset",
    "available_reference_languages",
    "load_reference_dataset",
]
