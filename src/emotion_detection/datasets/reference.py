"""Independent reference dataset acquisition, preparation and in-memory loading."""

import platform
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow

from emotion_detection.datasets.acquisition import acquire_sources, read_source
from emotion_detection.datasets.reference_track_a import prepare_track_a
from emotion_detection.datasets.reference_types import ReferencePreparationError
from emotion_detection.datasets.sources import DEFAULT_RELEASE, get_reference_sources

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _flatten(item, f"{prefix}.{index}")
    else:
        yield {"metric": prefix, "value": value}


def _write_audits(directory, sources, diagnostics, metadata=None):
    directory.mkdir(parents=True, exist_ok=True)
    records = []
    for source in sources:
        record = source.metadata()
        record["label_names"] = "|".join(source.label_names)
        records.append(record)
    pd.DataFrame(records).to_csv(
        directory / "source_manifest.csv", index=False, lineterminator="\n"
    )
    for name, frame in diagnostics.items():
        frame.to_csv(directory / f"{name}.csv", index=False, lineterminator="\n")
    pd.DataFrame(
        _flatten(metadata or {"status": "failed"}), columns=["metric", "value"]
    ).to_csv(
        directory / "summary.csv",
        index=False,
        lineterminator="\n",
    )


def load_reference_dataset(
    track="a",
    language="eng",
    *,
    release=DEFAULT_RELEASE,
    project_root=PROJECT_ROOT,
    interactive=None,
):
    """Acquire selected official inputs and regenerate independent canonical outputs.

    Raw files remain unchanged. Outputs preserve source order, with lower-priority
    leakage removals explicitly recorded. This never reads the legacy joint data.
    """
    sources = get_reference_sources(track, language, release)
    root = Path(project_root).resolve()
    raw = root / "data/raw/reference_baselines" / release
    output = root / "data/processed/reference_baselines" / f"track_{track}" / language
    analysis = (
        root / "output/analysis/reference_baselines" / f"track_{track}" / language
    )
    paths = [path.resolve() for path in (raw, output, analysis)]
    for i, path in enumerate(paths):
        for other in paths[i + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(
                    "Reference raw, processed and analysis directories must not overlap"
                )
    if (
        {source.split for source in sources} != {"train", "dev", "test"}
        or len(sources) != 3
        or any(
            (s.track, s.language, s.release) != (track, language, release)
            for s in sources
        )
        or len({s.label_names for s in sources}) != 1
        or len({(s.repository, s.revision) for s in sources}) != 1
    ):
        raise ValueError(
            "Reference manifest must declare one consistent track/language/release and all three splits"
        )
    acquire_sources(sources, raw, interactive=interactive)
    frames = {
        source.split: read_source(raw / source.relative_path, source)
        for source in sources
    }
    try:
        dataset, diagnostics = prepare_track_a(
            frames,
            label_names=sources[0].label_names,
            language=language,
            source_metadata=[source.metadata() for source in sources],
        )
    except ReferencePreparationError as error:
        _write_audits(
            analysis,
            sources,
            error.diagnostics,
            {"status": "failed", "reason": str(error)},
        )
        raise
    dataset.metadata["release"] = release
    dataset.metadata["environment"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }
    dataset.metadata["code_revision"] = None
    dataset.metadata["code_dirty"] = None
    try:
        dataset.metadata["code_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dataset.metadata["code_dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        getattr(dataset, split).to_parquet(output / f"{split}.parquet", index=False)
    _write_audits(analysis, sources, diagnostics, dataset.metadata)
    return dataset
