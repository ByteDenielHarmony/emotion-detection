"""Modular Track A reference baseline; all training data is supplied in memory.

Modeling reference (not an exact reproduction of the reported task scores):
https://aclanthology.org/2025.semeval-1.327/, section 4.3.
Upstream implementation:
https://github.com/emotion-analysis-project/BRIGHTER/blob/
08d4a59ad950ad78265322cf3d3dbf41a5888f90/regular_lms_track_ab.py
"""

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)

from emotion_detection.evaluation.metrics import multilabel_metrics
from emotion_detection.reference_baselines.data import (
    EmotionDataCollator,
    TokenizedEmotionDataset,
    infer_label_names,
    validate_frame,
    validate_label_names,
    validate_partition_separation,
)
from emotion_detection.reference_baselines.training import (
    DevelopmentReportingCallback,
    validate_training_control,
)
from emotion_detection.reference_baselines.transformer import (
    DEFAULT_MODEL_NAME,
    WeightedMultilabelTrainer,
    load_sequence_classifier,
    positive_class_weights,
    resolve_device,
    resolve_precision,
)


def _fingerprint(frame, label_names):
    payload = frame[["id", "text", *label_names]].to_json(
        orient="split", index=False, force_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_metadata():
    root = Path(__file__).resolve().parents[3]
    metadata = {"git_revision": None, "git_dirty": None, "uv_lock_sha256": None}
    try:
        metadata["git_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        metadata["git_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    lock = root / "uv.lock"
    if lock.is_file():
        metadata["uv_lock_sha256"] = hashlib.sha256(lock.read_bytes()).hexdigest()
    return metadata


class TrackAClassifier:
    """Fine-tune any compatible RoBERTa-family sequence-classification checkpoint.

    Inputs are already-loaded DataFrames with id, text, and named binary emotion
    columns. Prediction also accepts frames without targets. No dataset file I/O
    or persistent loading pipeline is implemented here.
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        *,
        revision=None,
        label_names=None,
        max_length=256,
        threshold=0.5,
        seed=42,
        device="auto",
        local_files_only=False,
    ):
        if not isinstance(max_length, int) or max_length < 1:
            raise ValueError("max_length must be a positive integer")
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if not isinstance(seed, int) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer between 0 and 2**32 - 1")
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
        self.model_name = str(model_name)
        self.revision = revision
        self.label_names = (
            None if label_names is None else validate_label_names(label_names)
        )
        self.max_length = max_length
        self.threshold = threshold
        self.seed = seed
        self.device = device
        self.local_files_only = local_files_only
        self.model = None
        self.tokenizer = None
        self.run_dir = None
        self.metadata = {}
        self._fitted = False

    def fit(
        self,
        train_data: pd.DataFrame,
        validation_data: pd.DataFrame,
        *,
        output_dir="output/runs",
        run_id=None,
        epochs=2,
        learning_rate=1e-5,
        batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        mixed_precision="auto",
        monitor="macro_f1",
        early_stopping_patience=None,
        early_stopping_min_delta=0.0,
        data_metadata=None,
    ):
        """Train from the configured checkpoint; save the best validation model.

        A fresh, non-overwriting run directory is created. This method does not
        accept test data. data_metadata can carry caller-provided dataset revisions
        or provenance; fingerprints of both input partitions are always recorded.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("Expected an already-loaded pandas DataFrame")
        names = self.label_names or infer_label_names(train_data)
        validate_frame(train_data, names)
        validate_frame(validation_data, names)
        validate_partition_separation(train_data, validation_data)
        validate_training_control(
            monitor, early_stopping_patience, early_stopping_min_delta
        )
        if not math.isfinite(epochs) or epochs <= 0:
            raise ValueError("epochs must be positive and finite")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        for name, value in (
            ("batch_size", batch_size),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
        ):
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        provenance = dict(data_metadata or {})
        json.dumps(provenance, allow_nan=False)
        if run_id is None:
            run_id = (
                "reference-track-a-"
                + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
                + "-"
                + uuid4().hex[:8]
            )
        if not isinstance(run_id, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id
        ):
            raise ValueError("run_id must be a simple name, not a path")
        device = resolve_device(self.device)
        precision = resolve_precision(mixed_precision, device)
        run_dir = Path(output_dir) / run_id
        if run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {run_dir}")

        self._fitted = False
        self.model, self.tokenizer, resolved_revision = load_sequence_classifier(
            self.model_name,
            names,
            revision=self.revision,
            seed=self.seed,
            local_files_only=self.local_files_only,
        )
        if self.max_length > self.tokenizer.model_max_length:
            raise ValueError("max_length exceeds the tokenizer's supported context")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("The checkpoint tokenizer must define a padding token")
        self.label_names = names
        targets = train_data[list(names)].to_numpy(dtype=np.float32)
        weights = positive_class_weights(targets)
        train_dataset = TokenizedEmotionDataset(
            train_data["text"],
            self.tokenizer,
            labels=targets,
            max_length=self.max_length,
        )
        validation_dataset = TokenizedEmotionDataset(
            validation_data["text"],
            self.tokenizer,
            labels=validation_data[list(names)].to_numpy(dtype=np.float32),
            max_length=self.max_length,
        )
        run_dir.mkdir(parents=True, exist_ok=False)
        self.run_dir = run_dir
        reporting = DevelopmentReportingCallback(
            run_dir, names, monitor=monitor, threshold=self.threshold
        )
        self.metadata = {
            "schema_version": 1,
            "track": "a",
            "model_name": self.model_name,
            "model_revision": resolved_revision,
            "label_names": list(names),
            "threshold": self.threshold,
            "max_length": self.max_length,
            "seed": self.seed,
            "source": _source_metadata(),
            "environment": {
                "python": platform.python_version(),
                "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                **{
                    name: version(name)
                    for name in (
                        "torch",
                        "transformers",
                        "accelerate",
                        "numpy",
                        "pandas",
                        "scikit-learn",
                    )
                },
            },
            "data": {
                "train_rows": len(train_data),
                "validation_rows": len(validation_data),
                "train_sha256": _fingerprint(train_data, names),
                "validation_sha256": _fingerprint(validation_data, names),
                "provenance": provenance,
            },
            "training": {
                "epochs": epochs,
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "gradient_checkpointing": gradient_checkpointing,
                "mixed_precision": precision,
                "device": device,
                "positive_class_weights": weights.tolist(),
                "optimizer": "adamw_torch",
                "weight_decay": 0.0,
                "monitor": monitor,
                "monitor_direction": "minimize" if monitor == "loss" else "maximize",
                "early_stopping_patience": early_stopping_patience,
                "early_stopping_min_delta": early_stopping_min_delta,
                "checkpoint_retention": "at_most_two_protecting_best_and_latest",
            },
            "reference": {
                "paper": "https://aclanthology.org/2025.semeval-1.327/",
                "repository": "https://github.com/emotion-analysis-project/BRIGHTER",
                "revision": "08d4a59ad950ad78265322cf3d3dbf41a5888f90",
                "deviations": [
                    "Development partition used for validation, never the test set",
                    f"Best development {monitor} checkpoint retained",
                    "Microbatching and gradient checkpointing for limited VRAM",
                    "Checkpoint is configurable; exact published scores not claimed",
                ],
            },
        }
        if early_stopping_patience is not None:
            self.metadata["reference"]["deviations"].append(
                "Optional development-based early stopping enabled"
            )

        def compute_metrics(prediction):
            # sigmoid(logit) >= threshold, with stable sigmoid for large logits.
            probabilities = torch.from_numpy(prediction.predictions).sigmoid().numpy()
            return reporting.capture(
                prediction.label_ids, probabilities >= self.threshold
            )

        args = TrainingArguments(
            output_dir=str(run_dir / "training"),
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gradient_checkpointing=gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model=monitor,
            greater_is_better=monitor != "loss",
            optim="adamw_torch",
            weight_decay=0.0,
            fp16=precision == "fp16",
            bf16=precision == "bf16",
            use_cpu=device == "cpu",
            dataloader_pin_memory=device == "cuda",
            seed=self.seed,
            data_seed=self.seed,
            report_to="none",
            logging_strategy="epoch",
        )
        self.metadata["training_arguments"] = args.to_dict()
        (run_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        early_stopping = None
        callbacks = []
        if early_stopping_patience is not None:
            early_stopping = EarlyStoppingCallback(
                early_stopping_patience=early_stopping_patience,
                early_stopping_threshold=early_stopping_min_delta,
            )
            callbacks.append(early_stopping)
        callbacks.append(reporting)
        trainer = WeightedMultilabelTrainer(
            model=self.model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=self.tokenizer,
            data_collator=EmotionDataCollator(self.tokenizer),
            compute_metrics=compute_metrics,
            pos_weights=weights,
            callbacks=callbacks,
        )
        training_result = trainer.train()
        # Trainer has restored the selected best weights. Do not count this final
        # evaluation as another epoch or change the early-stopping counter.
        reporting.epoch_reporting = False
        if early_stopping is not None:
            trainer.remove_callback(early_stopping)
        selected_metrics = trainer.evaluate()
        metrics = {**training_result.metrics, **selected_metrics}
        self.model = trainer.model
        self.model.eval()
        if trainer.state.best_model_checkpoint is None:
            raise RuntimeError(
                "Training completed without a selected validation checkpoint"
            )
        best_step = int(Path(trainer.state.best_model_checkpoint).name.split("-")[-1])
        best_epoch = next(
            record["epoch"]
            for record in reporting.history
            if record["global_step"] == best_step
        )
        stopped_early = (
            early_stopping is not None
            and early_stopping.early_stopping_patience_counter
            >= early_stopping_patience
            and trainer.state.global_step < trainer.state.max_steps
        )
        self.metadata["training"].update(
            {
                "best_metric": trainer.state.best_metric,
                "best_macro_f1": max(
                    record["metrics"]["macro_f1"] for record in reporting.history
                ),
                "selected_checkpoint_macro_f1": selected_metrics["eval_macro_f1"],
                "selected_checkpoint_loss": selected_metrics["eval_loss"],
                "best_epoch": best_epoch,
                "epochs_completed": trainer.state.epoch,
                "stopped_early": stopped_early,
                "stopping_reason": "early_stopping" if stopped_early else "epoch_limit",
            }
        )
        self.metadata["training"]["best_checkpoint"] = (
            trainer.state.best_model_checkpoint
        )
        self.metadata["training"]["global_step"] = trainer.state.global_step
        reporting.selected_report(
            epoch=best_epoch, global_step=best_step, metrics=selected_metrics
        )
        self._fitted = True
        self.save(run_dir / "checkpoint")
        (run_dir / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        return self

    def _require_fitted(self):
        if not self._fitted:
            raise RuntimeError("Fit the classifier or load a saved checkpoint first")

    def predict_proba(self, data: pd.DataFrame, *, batch_size=8) -> pd.DataFrame:
        """Return id and named probabilities in exactly the input row order."""
        return self._predict_values(data, batch_size=batch_size, probabilities=True)

    def predict_logits(self, data: pd.DataFrame, *, batch_size=8) -> pd.DataFrame:
        """Return raw logits for stable loss calculation without a second pass."""
        return self._predict_values(data, batch_size=batch_size, probabilities=False)

    def _predict_values(self, data, *, batch_size, probabilities):
        self._require_fitted()
        validate_frame(data, self.label_names, require_labels=False, allow_empty=True)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        result = data[["id"]].copy()
        if data.empty:
            for name in self.label_names:
                result[name] = pd.Series(index=data.index, dtype=float)
            return result
        dataset = TokenizedEmotionDataset(
            data["text"], self.tokenizer, max_length=self.max_length
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=EmotionDataCollator(self.tokenizer),
        )
        device = next(self.model.parameters()).device
        self.model.eval()
        batches = []
        with torch.inference_mode():
            for batch in loader:
                batch = {name: value.to(device) for name, value in batch.items()}
                logits = self.model(**batch).logits.float()
                values = logits.sigmoid() if probabilities else logits
                batches.append(values.cpu().numpy())
        values = np.concatenate(batches)
        for index, name in enumerate(self.label_names):
            result[name] = values[:, index]
        return result

    def predict(self, data: pd.DataFrame, *, batch_size=8) -> pd.DataFrame:
        result = self.predict_proba(data, batch_size=batch_size)
        for name in self.label_names:
            result[name] = (result[name] >= self.threshold).astype("int64")
        return result

    def evaluate(self, data: pd.DataFrame, *, batch_size=8) -> dict[str, float]:
        self._require_fitted()
        validate_frame(data, self.label_names)
        prediction = self.predict(data, batch_size=batch_size)
        return multilabel_metrics(
            data[list(self.label_names)].to_numpy(),
            prediction[list(self.label_names)].to_numpy(),
            self.label_names,
        )

    def save(self, checkpoint_dir):
        """Save a self-contained inference checkpoint, never overwrite artifacts."""
        self._require_fitted()
        path = Path(checkpoint_dir)
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Checkpoint directory is not empty: {path}")
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        # Preserve a caller's validation-selected threshold on subsequent saves.
        self.metadata["threshold"] = self.threshold
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path, safe_serialization=True)
        self.tokenizer.save_pretrained(path)
        (path / "metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, checkpoint_dir, *, device="auto"):
        """Restore weights and tokenizer entirely locally, without dataset loading."""
        path = Path(checkpoint_dir)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("schema_version") != 1 or metadata.get("track") != "a":
            raise ValueError("Unsupported checkpoint metadata or task")
        classifier = cls(
            model_name=metadata["model_name"],
            revision=metadata["model_revision"],
            label_names=metadata["label_names"],
            max_length=metadata["max_length"],
            threshold=metadata["threshold"],
            seed=metadata["seed"],
            device=device,
            local_files_only=True,
        )
        classifier.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True
        )
        classifier.model = AutoModelForSequenceClassification.from_pretrained(
            path, local_files_only=True
        )
        expected_mapping = dict(enumerate(classifier.label_names))
        if classifier.model.config.id2label != expected_mapping:
            raise ValueError("Saved model and metadata disagree on emotion label order")
        classifier.model.to(resolve_device(device))
        classifier.model.eval()
        classifier.metadata = metadata
        classifier.run_dir = path.parent
        classifier._fitted = True
        return classifier
