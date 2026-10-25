"""Training controls and epoch diagnostics, reusable by reference classifiers."""

import json
import math
from pathlib import Path

from transformers import TrainerCallback

from emotion_detection.evaluation.reporting import format_development_report
from emotion_detection.reference_baselines.tasks import BinaryEmotionTask


def validate_crosslingual_checkpoint(model_name, target_language):
    """Reject known supervised project warm starts, including direct model API use.

    Arbitrary external checkpoints still require a caller provenance review.
    """
    metadata = Path(model_name) / "metadata.json"
    if metadata.is_file():
        previous = json.loads(metadata.read_text())
        provenance = previous.get("data", {}).get("provenance", {})
        supervised = (
            provenance.get("source_languages", [])
            if previous.get("track") == "c"
            else [provenance.get("language")]
        )
        if target_language in supervised:
            raise ValueError(
                "Track C cannot warm-start from target-language supervised training"
            )


def validate_training_control(
    monitor, patience, min_delta, *, primary_metric="macro_f1"
):
    if monitor not in {primary_metric, "loss"}:
        raise ValueError(f"monitor must be '{primary_metric}' or 'loss' for this task")
    if patience is not None and (
        not isinstance(patience, int) or isinstance(patience, bool) or patience < 1
    ):
        raise ValueError("early_stopping_patience must be a positive integer or None")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("early_stopping_min_delta must be finite and nonnegative")
    if patience is None and min_delta != 0:
        raise ValueError("early_stopping_min_delta requires early_stopping_patience")


class DevelopmentReportingCallback(TrainerCallback):
    """Reuse evaluation predictions; persist one diagnostic record per epoch.

    Checkpoint retention is delegated to Trainer's safe rotation in the current
    run directory. This callback never deletes checkpoints or runs inference.
    """

    def __init__(
        self,
        run_dir,
        label_names,
        *,
        monitor,
        threshold,
        task=None,
        validation_role="dev",
    ):
        self.run_dir = Path(run_dir)
        self.label_names = tuple(label_names)
        self.monitor = monitor
        self.threshold = threshold
        self.task = task or BinaryEmotionTask()
        self.validation_role = validation_role
        self.latest = None
        self.history = []
        self.epoch_reporting = True

    def capture(self, labels, predictions):
        self.latest = self.task.report(labels, predictions, self.label_names)
        # Trainer may rename/prefix the returned metric keys in place.
        return dict(self.latest["metrics"])

    def record(self, *, epoch, global_step, metrics):
        if self.latest is None:
            raise RuntimeError("Development diagnostics require evaluation predictions")
        scalars = dict(self.latest["metrics"])
        scalars.update(
            {
                key.removeprefix("eval_"): value
                for key, value in metrics.items()
                if key.startswith("eval_")
            }
        )
        value = scalars[self.monitor]
        if not math.isfinite(value):
            raise ValueError(f"Development {self.monitor} is not finite")
        return {
            **self.latest,
            "metrics": scalars,
            "epoch": float(epoch),
            "global_step": int(global_step),
            "split": self.validation_role,
            "threshold": self.threshold,
            "monitor": self.monitor,
            "monitor_value": float(value),
            "direction": "minimize" if self.monitor == "loss" else "maximize",
        }

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if not self.epoch_reporting or not state.is_world_process_zero:
            return
        record = self.record(
            epoch=state.epoch, global_step=state.global_step, metrics=metrics
        )
        record["stop_requested"] = bool(control.should_training_stop)
        with (self.run_dir / "epoch_history.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.history.append(record)
        print(format_development_report(record), flush=True)

    def selected_report(self, *, epoch, global_step, metrics):
        """Final best-model evaluation, kept separate from chronological epochs."""
        record = self.record(epoch=epoch, global_step=global_step, metrics=metrics)
        (self.run_dir / "validation_report.json").write_text(
            json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        print(format_development_report(record, selected=True), flush=True)
