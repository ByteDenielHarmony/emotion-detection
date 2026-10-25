"""One-shot held-out evaluation after a reference model has been selected/saved.

Receives data in memory; dataset acquisition and loading are deliberately absent.
Test outcomes never update thresholds, model weights, or checkpoint selection.
"""

import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from emotion_detection.evaluation.reporting import format_classification_report
from emotion_detection.reference_baselines.tasks import task_for_track
from emotion_detection.reference_baselines.transformer import resolve_precision


def evaluate_reference_test(classifier, test_data, *, batch_size=2):
    """Evaluate the selected model once and save separate, non-overwriting files."""
    run = Path(classifier.run_dir)
    checkpoint = run / "checkpoint"
    if not (checkpoint / "config.json").is_file() or not any(
        (checkpoint / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json")
    ):
        raise ValueError("Final test evaluation requires an exported best checkpoint")
    saved = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    track = saved.get("track", "a")
    task = task_for_track(track)
    names = tuple(classifier.label_names)
    if (
        saved["label_names"] != list(names)
        or saved["threshold"] != classifier.threshold
    ):
        raise ValueError(
            "Use the saved checkpoint's label order and threshold for testing"
        )
    artifacts = [
        run / name
        for name in (
            "test_metrics.json",
            "test_report.json",
            "test_predictions.csv",
        )
    ]
    if any(path.exists() for path in artifacts):
        raise FileExistsError(
            "Final test artifacts already exist; refusing to overwrite or rescore"
        )
    task.validate(test_data, names)
    fingerprint = hashlib.sha256(
        test_data.to_json(orient="split", index=False, force_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()
    expected = saved["data"]["provenance"].get("prepared_fingerprints", {}).get("test")
    if not expected or fingerprint != expected:
        raise ValueError(
            "Test data does not match the selected run's recorded fingerprint"
        )

    training = saved["training"]
    weights = None
    if track != "b":
        all_names = saved.get("model_label_names", list(names))
        all_weights = np.asarray(training["positive_class_weights"], dtype=np.float32)
        if (
            all_weights.shape != (len(all_names),)
            or not np.isfinite(all_weights).all()
            or (all_weights < 0).any()
        ):
            raise ValueError("Invalid training class weights in checkpoint metadata")
        weights = torch.tensor(all_weights[[all_names.index(name) for name in names]])
    device = next(classifier.model.parameters()).device.type
    training_precision = training["mixed_precision"]
    precision = (
        resolve_precision(training_precision, "cuda") if device == "cuda" else "no"
    )
    context = nullcontext()
    if precision != "no":
        context = torch.autocast(
            "cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16
        )
    with context:
        logits_frame = classifier.predict_logits(test_data, batch_size=batch_size)
    if logits_frame["id"].tolist() != test_data["id"].tolist():
        raise ValueError("Test predictions do not preserve input IDs/order")
    logits = torch.tensor(
        logits_frame[list(task.output_names(names))].to_numpy(), dtype=torch.float32
    )
    if not torch.isfinite(logits).all():
        raise ValueError("Test logits contain non-finite values")
    predicted = task.decode(logits.numpy(), len(names), saved["threshold"])
    targets = task.targets(test_data, names)
    report = task.report(targets, predicted, names)
    # Same task objective as validation, evaluated stably from logits once.
    report["metrics"]["loss"] = float(
        task.loss(logits, torch.from_numpy(targets), weights)
    )
    best_checkpoint = training.get("best_checkpoint")
    best_step = (
        int(Path(best_checkpoint).name.split("-")[-1]) if best_checkpoint else None
    )
    report.update(
        {
            "schema_version": 1,
            "status": "completed",
            "split": "test",
            "checkpoint": "checkpoint",
            "epoch": training.get("best_epoch"),
            "global_step": best_step,
            "monitor": training.get("monitor", "macro_f1"),
            "direction": "minimize"
            if training.get("monitor") == "loss"
            else "maximize",
            "selection_split": saved.get("data", {}).get("validation_role", "dev"),
            "track": track,
            "threshold": saved["threshold"],
            "rows": len(test_data),
            "data_fingerprint": fingerprint,
            "device": device,
            "inference_precision": precision,
            "training_precision": training_precision,
        }
    )
    predictions = test_data[["id"]].copy()
    for index, name in enumerate(names):
        predictions[name] = predicted[:, index]
    metrics_json = json.dumps(report["metrics"], indent=2, allow_nan=False) + "\n"
    report_json = json.dumps(report, indent=2, allow_nan=False) + "\n"
    # x mode protects existing artifacts even if another process writes first.
    with artifacts[0].open("x", encoding="utf-8") as stream:
        stream.write(metrics_json)
    with artifacts[1].open("x", encoding="utf-8") as stream:
        stream.write(report_json)
    predictions.to_csv(artifacts[2], index=False, mode="x", lineterminator="\n")
    print(
        format_classification_report(
            report, heading="Held-out test metrics - selected checkpoint"
        ),
        flush=True,
    )
    return report
