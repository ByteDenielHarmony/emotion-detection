"""Data preparation APIs; importing them never loads or trains a model."""

from emotion_detection.datasets.reference import load_reference_dataset
from emotion_detection.datasets.reference_types import ReferenceDataset

__all__ = ["ReferenceDataset", "load_reference_dataset"]
