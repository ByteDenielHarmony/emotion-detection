"""Prepared reference data, independent of acquisition records and model code."""

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
