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


def multilabel_diagnostics(y_true, y_pred, label_names):
    """Scalar metrics and binary confusion matrices, including absent classes.

    Matrix rows are actual [absent, present]; columns are predicted
    [absent, present], i.e. [[TN, FP], [FN, TP]].
    """
    metrics = multilabel_metrics(y_true, y_pred, label_names)
    truth, prediction = np.asarray(y_true, dtype=bool), np.asarray(y_pred, dtype=bool)
    tp = (truth & prediction).sum(axis=0)
    fp = (~truth & prediction).sum(axis=0)
    fn = (truth & ~prediction).sum(axis=0)
    tn = (~truth & ~prediction).sum(axis=0)
    total_tp, total_fp, total_fn = int(tp.sum()), int(fp.sum()), int(fn.sum())
    denominator = 2 * total_tp + total_fp + total_fn
    metrics["micro_f1"] = 2 * total_tp / denominator if denominator else 0.0
    return {
        "metrics": metrics,
        "label_names": list(label_names),
        "support": {name: int(tp[i] + fn[i]) for i, name in enumerate(label_names)},
        "confusion_matrices": {
            name: [[int(tn[i]), int(fp[i])], [int(fn[i]), int(tp[i])]]
            for i, name in enumerate(label_names)
        },
        "matrix_axes": {
            "rows": ["actual_absent", "actual_present"],
            "columns": ["predicted_absent", "predicted_present"],
        },
    }


def masked_multilabel_diagnostics(y_true, y_pred, label_names, known):
    truth, prediction, known = (
        np.asarray(y_true),
        np.asarray(y_pred),
        np.asarray(known, dtype=bool),
    )
    if (
        truth.shape != prediction.shape
        or truth.shape != known.shape
        or truth.ndim != 2
        or truth.shape[1] != len(label_names)
        or not len(label_names)
        or len(set(label_names)) != len(label_names)
    ):
        raise ValueError("Masked targets, predictions, and annotation masks must align")
    metrics, matrices, support, coverage, undefined = {}, {}, {}, {}, []
    scores, total_tp, total_fp, total_fn = [], 0, 0, 0
    for i, name in enumerate(label_names):
        mask = known[:, i]
        coverage[name] = int(mask.sum())
        if not mask.any():
            undefined.append(name)
            matrices[name], support[name] = [[0, 0], [0, 0]], 0
            metrics.update(
                {f"{metric}_{name}": None for metric in ("precision", "recall", "f1")}
            )
            continue
        result = multilabel_diagnostics(
            truth[mask, i : i + 1], prediction[mask, i : i + 1], [name]
        )
        metrics.update(
            {
                key: value
                for key, value in result["metrics"].items()
                if key not in {"macro_f1", "micro_f1"}
            }
        )
        matrices[name], support[name] = (
            result["confusion_matrices"][name],
            result["support"][name],
        )
        (_, fp), (fn, tp) = matrices[name]
        total_tp, total_fp, total_fn = total_tp + tp, total_fp + fp, total_fn + fn
        scores.append(result["metrics"]["macro_f1"])
    if not scores:
        raise ValueError("No observed labels for classification evaluation")
    denominator = 2 * total_tp + total_fp + total_fn
    metrics.update(
        macro_f1=float(np.mean(scores)),
        micro_f1=2 * total_tp / denominator if denominator else 0.0,
    )
    return {
        "metrics": metrics,
        "label_names": list(label_names),
        "support": support,
        "known_counts": coverage,
        "undefined_emotions": undefined,
        "confusion_matrices": matrices,
        "matrix_axes": {
            "rows": ["actual_absent", "actual_present"],
            "columns": ["predicted_absent", "predicted_present"],
        },
    }


def intensity_diagnostics(y_true, y_pred, label_names):
    truth, prediction = np.asarray(y_true), np.asarray(y_pred)
    if (
        truth.ndim != 2
        or truth.shape != prediction.shape
        or not len(truth)
        or truth.shape[1] != len(label_names)
        or not len(label_names)
        or len(set(label_names)) != len(label_names)
    ):
        raise ValueError("Intensity targets and predictions must align")
    if not np.isin(truth, range(4)).all() or not np.isin(prediction, range(4)).all():
        raise ValueError("Intensity classes must be integers 0-3")
    metrics, matrices, undefined = {}, {}, []
    for i, name in enumerate(label_names):
        if len(truth) < 2 or np.ptp(truth[:, i]) == 0 or np.ptp(prediction[:, i]) == 0:
            value = 0.0
            undefined.append(name)
        else:
            value = float(np.corrcoef(truth[:, i], prediction[:, i])[0, 1])
        metrics[f"pearson_{name}"] = value
        metrics[f"mae_{name}"] = float(np.abs(truth[:, i] - prediction[:, i]).mean())
        metrics[f"accuracy_{name}"] = float((truth[:, i] == prediction[:, i]).mean())
        matrix = np.zeros((4, 4), dtype=int)
        np.add.at(matrix, (truth[:, i].astype(int), prediction[:, i].astype(int)), 1)
        matrices[name] = matrix.tolist()
    metrics["pearson"] = float(
        np.mean([metrics[f"pearson_{name}"] for name in label_names])
    )
    metrics["mae"] = float(np.abs(truth - prediction).mean())
    return {
        "metrics": metrics,
        "label_names": list(label_names),
        "support": {name: len(truth) for name in label_names},
        "confusion_matrices": matrices,
        "undefined_correlations": undefined,
        "correlation_policy": "Constant gold/prediction or fewer than two observations: r=0, flagged undefined (BRIGHTER convention)",
        "matrix_axes": {"rows": [0, 1, 2, 3], "columns": [0, 1, 2, 3]},
    }
