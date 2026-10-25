"""Reference experiment orchestration; dataset I/O stays in the dataset package."""

import gc
import json
import math
import re
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import torch

from emotion_detection.datasets import (
    load_prepared_reference_dataset,
    prepare_reference_dataset,
)
from emotion_detection.datasets.acquisition import AcquisitionError
from emotion_detection.datasets.sources import DEFAULT_RELEASE, get_reference_sources
from emotion_detection.evaluation.reference import evaluate_reference_test
from emotion_detection.reference_baselines import (
    TrackAClassifier,
    TrackBClassifier,
    TrackCClassifier,
)
from emotion_detection.reference_baselines.training import (
    validate_crosslingual_checkpoint,
    validate_training_control,
)
from emotion_detection.reference_baselines.transformer import (
    DEFAULT_MODEL_NAME,
    resolve_device,
    resolve_precision,
)


@dataclass(frozen=True)
class ReferenceTrainingOptions:
    """In-memory command options, not an external configuration framework."""

    project_root: Path = field(default_factory=Path.cwd)
    track: str = "all"
    language: str = "eng"
    release: str = DEFAULT_RELEASE
    model_name: str | None = None
    revision: str | None = None
    device: str = "auto"
    epochs: float = 2
    learning_rate: float = 1e-5
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    mixed_precision: str = "auto"
    monitor: str = "auto"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_length: int = 256
    seed: int = 42
    threshold: float | None = None
    run_id: str | None = None
    local_files_only: bool = False
    prepare_data: bool = False
    non_interactive: bool = False
    skip_test: bool = False
    protocol: str | None = None


def _monitor(options):
    primary = "pearson" if options.track == "b" else "macro_f1"
    return primary if options.monitor == "auto" else options.monitor


def _protocol(options):
    return (
        {"protocol": options.protocol}
        if options.track == "c" and options.protocol
        else {}
    )


def _new_model(options, data=None):
    sources = get_reference_sources(options.track, options.language, options.release)
    model_class = {"a": TrackAClassifier, "b": TrackBClassifier, "c": TrackCClassifier}[
        options.track
    ]
    return model_class(
        model_name=options.model_name or DEFAULT_MODEL_NAME,
        revision=options.revision,
        label_names=data.label_names if data is not None else sources[-1].label_names,
        device=options.device,
        max_length=options.max_length,
        seed=options.seed,
        threshold=options.threshold if options.track != "b" else None,
        local_files_only=options.local_files_only,
        **(
            {"model_label_names": data.model_label_names}
            if options.track == "c" and data is not None
            else {}
        ),
    )


def _validate_options(options):
    if options.track not in {"a", "b", "c"}:
        raise ValueError("Select track a, b, c, or all")
    get_reference_sources(options.track, options.language, options.release)
    if options.track == "c":
        from emotion_detection.datasets.sources_track_c import (
            DEFAULT_C_PROTOCOL,
            get_crosslingual_sources,
        )

        get_crosslingual_sources(
            options.language, options.release, options.protocol or DEFAULT_C_PROTOCOL
        )
        validate_crosslingual_checkpoint(
            options.model_name or DEFAULT_MODEL_NAME, options.language
        )
    validate_training_control(
        _monitor(options),
        options.early_stopping_patience,
        options.early_stopping_min_delta,
        primary_metric="pearson" if options.track == "b" else "macro_f1",
    )
    root = Path(options.project_root).resolve()
    for name in ("epochs", "learning_rate"):
        value = getattr(options, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    for name in ("batch_size", "gradient_accumulation_steps"):
        value = getattr(options, name)
        if not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if options.run_id is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", options.run_id):
            raise ValueError("run_id must be a simple name, not a path")
        if (root / "output/runs" / options.run_id).exists():
            raise FileExistsError(
                "Requested run already exists; choose another --run-id"
            )

    _new_model(options)  # Constructor validates settings without loading weights.
    device = resolve_device(options.device)
    resolve_precision(options.mixed_precision, device)


def _train_one(options, data, *, output_dir=None):
    model = _new_model(options, data)
    validation = data.source_dev if options.track == "c" else data.dev
    role = "source_dev" if options.track == "c" else "dev"
    print(
        f"Training reference {options.track.upper()} / {options.language}: {len(data.train)} train, {len(validation)} {role}; monitor={_monitor(options)}",
        flush=True,
    )
    try:
        model.fit(
            data.train,
            validation,
            output_dir=output_dir
            or Path(options.project_root).resolve() / "output/runs",
            run_id=options.run_id,
            epochs=options.epochs,
            learning_rate=options.learning_rate,
            batch_size=options.batch_size,
            gradient_accumulation_steps=options.gradient_accumulation_steps,
            gradient_checkpointing=options.gradient_checkpointing,
            mixed_precision=options.mixed_precision,
            monitor=_monitor(options),
            early_stopping_patience=options.early_stopping_patience,
            early_stopping_min_delta=options.early_stopping_min_delta,
            data_metadata=data.metadata,
        )
        if not options.skip_test:
            try:
                evaluate_reference_test(model, data.test, batch_size=options.batch_size)
            except (OSError, ValueError, RuntimeError) as error:
                raise RuntimeError(
                    f"Training completed; checkpoint saved under {model.run_dir}. Final test evaluation failed: {error}"
                ) from error
        return model.run_dir
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def train_reference(options: ReferenceTrainingOptions) -> Path:
    """Preflight all selected tracks, then run independent models sequentially."""
    if options.track not in {"all", "a", "b", "c"}:
        raise ValueError("Select track all, a, b, or c")
    if options.protocol is not None and options.track not in {"all", "c"}:
        raise ValueError("--protocol applies only to Track C")
    if options.threshold is not None and options.track == "b":
        raise ValueError("Track B uses argmax intensity classes, not --threshold")
    tracks = ("a", "b", "c") if options.track == "all" else (options.track,)
    selected = [replace(options, track=track) for track in tracks]
    for choice in selected:
        _validate_options(choice)
    # All acquisitions/readiness checks happen before the first model is loaded.
    if options.prepare_data:
        for choice in selected:
            try:
                prepare_reference_dataset(
                    choice.track,
                    choice.language,
                    release=choice.release,
                    project_root=choice.project_root,
                    interactive=False if choice.non_interactive else None,
                    **_protocol(choice),
                )
            except AcquisitionError as error:
                raise RuntimeError(str(error)) from error
    datasets = {
        choice.track: load_prepared_reference_dataset(
            choice.track,
            choice.language,
            release=choice.release,
            project_root=choice.project_root,
            **_protocol(choice),
        )
        for choice in selected
    }
    for choice in selected:
        data = datasets[choice.track]
        if choice.track == "c":
            for frame in (data.train, data.source_dev):
                if (
                    frame.source_language.eq(choice.language).any()
                    or not frame.source_track.eq("a").all()
                ):
                    raise ValueError(
                        "Track C requires other-language source supervision only"
                    )
            if any(not data.train[f"known_{name}"].any() for name in data.label_names):
                raise ValueError(
                    "A target emotion has no observed source training annotations"
                )
        print(
            f"Plan {choice.track.upper()}: {len(data.train)} train; selection={_monitor(choice)}",
            flush=True,
        )
    if options.track != "all":
        return _train_one(selected[0], datasets[options.track])
    root = Path(options.project_root).resolve()
    suite_id = options.run_id or (
        "reference-suite-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        + "-"
        + uuid4().hex[:8]
    )
    suite = root / "output/runs" / suite_id
    suite.mkdir(parents=True, exist_ok=False)
    summary = {
        "schema_version": 1,
        "language": options.language,
        "status": "running",
        "tracks": {track: {"status": "pending"} for track in tracks},
    }

    def save_summary():
        (suite / "summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    save_summary()
    for choice in selected:
        entry = summary["tracks"][choice.track]
        entry.update(
            status="running",
            metric=_monitor(choice),
            selection_split="source_dev" if choice.track == "c" else "dev",
            checkpoint=f"{choice.track}/checkpoint",
            test_skipped=choice.skip_test,
        )
        save_summary()
        try:
            path = _train_one(
                replace(choice, run_id=choice.track),
                datasets[choice.track],
                output_dir=suite,
            )
            entry.update(status="completed", run=choice.track)
            metrics = json.loads((path / "metrics.json").read_text())
            entry["development_score"] = metrics[f"eval_{_monitor(choice)}"]
            if not choice.skip_test:
                entry["test_metrics"] = json.loads(
                    (path / "test_metrics.json").read_text()
                )
        except (
            OSError,
            ValueError,
            RuntimeError,
            ImportError,
            KeyboardInterrupt,
        ) as error:
            status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            entry.update(status=status, error=str(error) or type(error).__name__)
            summary["status"] = status
            save_summary()
            raise
        save_summary()
    summary["status"] = "completed"
    save_summary()
    return suite
