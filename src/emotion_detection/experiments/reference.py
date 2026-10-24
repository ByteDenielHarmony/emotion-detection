"""Reference experiment orchestration; dataset I/O stays in the dataset package."""

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from emotion_detection.datasets import (
    load_prepared_reference_dataset,
    prepare_reference_dataset,
)
from emotion_detection.datasets.acquisition import AcquisitionError
from emotion_detection.datasets.sources import DEFAULT_RELEASE, get_reference_sources
from emotion_detection.evaluation.reference import evaluate_reference_test
from emotion_detection.reference_baselines import TrackAClassifier
from emotion_detection.reference_baselines.training import validate_training_control
from emotion_detection.reference_baselines.transformer import (
    DEFAULT_MODEL_NAME,
    resolve_device,
    resolve_precision,
)


@dataclass(frozen=True)
class ReferenceTrainingOptions:
    """In-memory command options, not an external configuration framework."""

    project_root: Path = field(default_factory=Path.cwd)
    track: str = "a"
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
    monitor: str = "macro_f1"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    max_length: int = 256
    seed: int = 42
    threshold: float = 0.5
    run_id: str | None = None
    local_files_only: bool = False
    prepare_data: bool = False
    non_interactive: bool = False
    skip_test: bool = False


def train_reference(options: ReferenceTrainingOptions) -> Path:
    """Train Track A on train/dev; optionally test the exported best checkpoint."""
    if options.track != "a":
        raise ValueError(
            "Reference training implements Track A only; B and C training are not implemented"
        )
    sources = get_reference_sources(options.track, options.language, options.release)
    validate_training_control(
        options.monitor,
        options.early_stopping_patience,
        options.early_stopping_min_delta,
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

    # Constructor and device checks do not load pretrained weights. Validate all
    # requested settings before allowing --prepare-data to write anything.
    model = TrackAClassifier(
        model_name=options.model_name or DEFAULT_MODEL_NAME,
        revision=options.revision,
        label_names=sources[0].label_names,
        device=options.device,
        max_length=options.max_length,
        seed=options.seed,
        threshold=options.threshold,
        local_files_only=options.local_files_only,
    )
    device = resolve_device(options.device)
    precision = resolve_precision(options.mixed_precision, device)
    if options.prepare_data:
        try:
            prepare_reference_dataset(
                options.track,
                options.language,
                release=options.release,
                project_root=root,
                interactive=False if options.non_interactive else None,
            )
        except AcquisitionError as error:
            raise RuntimeError(str(error)) from error
    data = load_prepared_reference_dataset(
        options.track, options.language, release=options.release, project_root=root
    )
    print(
        f"Training reference Track {options.track.upper()} / {options.language}: "
        f"{len(data.train)} train, {len(data.dev)} dev; "
        f"checkpoint={model.model_name}, device={device}, precision={precision}",
        flush=True,
    )
    model.fit(
        data.train,
        data.dev,
        output_dir=root / "output/runs",
        run_id=options.run_id,
        epochs=options.epochs,
        learning_rate=options.learning_rate,
        batch_size=options.batch_size,
        gradient_accumulation_steps=options.gradient_accumulation_steps,
        gradient_checkpointing=options.gradient_checkpointing,
        mixed_precision=options.mixed_precision,
        monitor=options.monitor,
        early_stopping_patience=options.early_stopping_patience,
        early_stopping_min_delta=options.early_stopping_min_delta,
        data_metadata=data.metadata,
    )
    if not options.skip_test:
        try:
            evaluate_reference_test(model, data.test, batch_size=options.batch_size)
        except (OSError, ValueError, RuntimeError) as error:
            raise RuntimeError(
                f"Training completed; the selected checkpoint is saved under {model.run_dir}. "
                f"Final test evaluation failed: {error}"
            ) from error
    return model.run_dir
