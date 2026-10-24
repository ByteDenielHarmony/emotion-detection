"""Prepared reference data, independent of acquisition records and model code."""

import hashlib
from dataclasses import dataclass

import pandas as pd


@dataclass
class ReferenceDataset:
    train: pd.DataFrame
    dev: pd.DataFrame
    test: pd.DataFrame
    label_names: tuple[str, ...]
    metadata: dict


class ReferencePreparationError(ValueError):
    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


@dataclass
class CrosslingualReferenceDataset:
    """Four distinct roles; target_dev must never masquerade as source validation.

    Frames use model_label_names plus boolean known_<emotion> masks. label_names
    is the target's native evaluation schema; unknown cells remain nullable.
    """

    train: pd.DataFrame
    source_dev: pd.DataFrame
    target_dev: pd.DataFrame
    test: pd.DataFrame
    label_names: tuple[str, ...]
    model_label_names: tuple[str, ...]
    metadata: dict


def prepared_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint canonical values, column order, and row order, not file encoding."""
    payload = frame.to_json(orient="split", index=False, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
