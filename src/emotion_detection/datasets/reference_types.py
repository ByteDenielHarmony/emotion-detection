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


def prepared_fingerprint(frame: pd.DataFrame) -> str:
    """Fingerprint canonical values, column order, and row order, not file encoding."""
    payload = frame.to_json(orient="split", index=False, force_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
