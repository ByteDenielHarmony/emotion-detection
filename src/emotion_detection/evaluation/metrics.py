"""Metrics shared by multi-label emotion classifiers."""

import numpy as np
from sklearn.metrics import precision_recall_fscore_support


def multilabel_metrics(y_true, y_pred, label_names):
    """Macro-F1 and per-emotion precision/recall/F1; undefined scores are zero."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if (
        y_true.ndim != 2
        or y_true.shape != y_pred.shape
        or not len(y_true)
        or y_true.shape[1] != len(label_names)
        or not len(label_names)
        or len(set(label_names)) != len(label_names)
    ):
        raise ValueError("Expected matching nonempty (examples, emotions) arrays")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Classification targets and predictions must be binary")
    metrics = {}
    scores = []
    for index, name in enumerate(label_names):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true[:, index], y_pred[:, index], average="binary", zero_division=0
        )
        metrics[f"precision_{name}"] = float(precision)
        metrics[f"recall_{name}"] = float(recall)
        metrics[f"f1_{name}"] = float(f1)
        scores.append(f1)
    metrics["macro_f1"] = float(np.mean(scores))
    return metrics
