"""Independent reference dataset acquisition, preparation and in-memory loading."""

import json
import platform
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow

from emotion_detection.datasets.acquisition import acquire_sources, read_source
from emotion_detection.datasets.reference_labeled import (
    PREPARATION_VARIANTS,
    TARGET_VALUES,
)
from emotion_detection.datasets.reference_track_a import prepare_track_a
from emotion_detection.datasets.reference_track_b import prepare_track_b
from emotion_detection.datasets.reference_types import (
    ReferenceDataset,
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import (
    DEFAULT_RELEASE,
    SIX_LABELS,
    SPLITS,
    get_reference_sources,
)

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
    # CSV remains the dataset artifact format. JSON-encoded cells retain types
    # and nesting for read-only loading; summary.csv is the human-readable view.
    pd.DataFrame(
        [
            {"key": key, "value": json.dumps(value, allow_nan=False)}
            for key, value in (metadata or {"status": "failed"}).items()
        ],
        columns=["key", "value"],
    ).to_csv(directory / "metadata.csv", index=False, lineterminator="\n")


def prepare_reference_dataset(
    track="a",
    language="eng",
    *,
    release=DEFAULT_RELEASE,
    project_root=PROJECT_ROOT,
    interactive=None,
    protocol=None,
):
    """Acquire selected official inputs and regenerate independent canonical outputs.

    Raw files remain unchanged. Outputs preserve source order, with lower-priority
    leakage removals explicitly recorded. This never reads the legacy joint data.
    """
    if track == "c":
        from emotion_detection.datasets.reference_crosslingual import (
            prepare_crosslingual_dataset,
        )
        from emotion_detection.datasets.sources_track_c import DEFAULT_C_PROTOCOL

        return prepare_crosslingual_dataset(
            language,
            release=release,
            project_root=project_root,
            interactive=interactive,
            protocol=protocol or DEFAULT_C_PROTOCOL,
        )
    if protocol is not None:
        raise ValueError("--protocol is applicable to Track C only")
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
        prepare = {"a": prepare_track_a, "b": prepare_track_b}[track]
        dataset, diagnostics = prepare(
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


def iter_prepare_reference_datasets(
    language="eng",
    *,
    release=DEFAULT_RELEASE,
    project_root=PROJECT_ROOT,
    interactive=None,
    protocol=None,
):
    """Yield prepared A, B, C datasets for one language, without training.

    Iteration validates all manifests and the C source protocol before acquisition
    or writes. Failures stop iteration without removing completed track outputs.
    Protocol overrides apply only to C; single-track APIs remain unchanged.
    """
    from emotion_detection.datasets.sources_track_c import (
        DEFAULT_C_PROTOCOL,
        get_crosslingual_sources,
    )

    for track in ("a", "b"):
        get_reference_sources(track, language, release)
    get_crosslingual_sources(language, release, protocol or DEFAULT_C_PROTOCOL)

    for track in ("a", "b", "c"):
        yield prepare_reference_dataset(
            track,
            language,
            release=release,
            project_root=project_root,
            interactive=interactive,
            **({"protocol": protocol} if track == "c" and protocol is not None else {}),
        )


def load_reference_dataset(
    track="a",
    language="eng",
    *,
    release=DEFAULT_RELEASE,
    project_root=PROJECT_ROOT,
    interactive=None,
    protocol=None,
):
    """Compatibility API: acquire and prepare, as before.

    New callers should explicitly choose prepare_reference_dataset for writes or
    load_prepared_reference_dataset for strictly read-only loading.
    """
    return prepare_reference_dataset(
        track,
        language,
        release=release,
        project_root=project_root,
        interactive=interactive,
        protocol=protocol,
    )


def load_prepared_reference_dataset(
    track="a",
    language="eng",
    *,
    release=DEFAULT_RELEASE,
    project_root=PROJECT_ROOT,
    protocol=None,
):
    """Read and validate saved data without acquisition, preparation, or writes.

    Verify the selected release/schema, source manifest, canonical fingerprints,
    row counts, targets, provenance columns, and separation of all partitions.
    Missing or stale artifacts are errors, never a reason to regenerate silently.
    """
    if track == "c":
        from emotion_detection.datasets.reference_crosslingual import (
            load_prepared_crosslingual_dataset,
        )
        from emotion_detection.datasets.sources_track_c import DEFAULT_C_PROTOCOL

        return load_prepared_crosslingual_dataset(
            language,
            release=release,
            project_root=project_root,
            protocol=protocol or DEFAULT_C_PROTOCOL,
        )
    if protocol is not None:
        raise ValueError("--protocol is applicable to Track C only")
    sources = get_reference_sources(track, language, release)
    root = Path(project_root).resolve()
    output = root / "data/processed/reference_baselines" / f"track_{track}" / language
    analysis = (
        root / "output/analysis/reference_baselines" / f"track_{track}" / language
    )

    def fail(reason):
        raise ReferencePreparationError(
            f"{reason}. Run 'emotion data prepare --dataset reference --track {track} "
            f"--language {language}' with the same --project-root and --release"
        )

    try:
        records = pd.read_csv(
            analysis / "metadata.csv", dtype=str, keep_default_na=False
        )
        if list(records.columns) != ["key", "value"] or not records["key"].is_unique:
            fail("Prepared metadata has invalid or duplicate fields")
        metadata = {row.key: json.loads(row.value) for row in records.itertuples()}
        json.dumps(metadata, allow_nan=False)
        labels = sources[0].label_names
        if (
            metadata.get("schema_version") != 1
            or metadata.get("track") != track
            or metadata.get("language") != language
            or metadata.get("release") != release
            or metadata.get("label_names") != list(labels)
            or metadata.get("preparation_variant") != PREPARATION_VARIANTS[track]
            or (
                track == "b"
                and metadata.get("target_values") != list(TARGET_VALUES[track])
            )
            or metadata.get("test_ids_and_order_preserved") is not True
            or metadata.get("source_manifest")
            != [source.metadata() for source in sources]
        ):
            fail(
                "Prepared metadata does not match the requested dataset/release/policy"
            )

        manifest = pd.read_csv(analysis / "source_manifest.csv", keep_default_na=False)
        expected = []
        for source in sources:
            record = source.metadata()
            record["label_names"] = "|".join(source.label_names)
            expected.append(record)
        if manifest.to_dict("records") != expected:
            fail("Prepared source manifest does not match the pinned inputs")

        frames = {
            split: pd.read_parquet(output / f"{split}.parquet") for split in SPLITS
        }
        for split, frame in frames.items():
            required = {
                "id",
                "text",
                "source_row",
                "source_language",
                "source_split",
                *labels,
            }
            if (
                frame.empty
                or not frame.columns.is_unique
                or not required.issubset(frame.columns)
                or set(frame.columns) & set(SIX_LABELS) != set(labels)
            ):
                fail(f"{split}: invalid prepared columns or empty partition")
            if (
                not frame["id"].is_unique
                or not frame["id"]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
                or not frame["text"]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
                or not frame[list(labels)].isin(TARGET_VALUES[track]).all().all()
                or not frame["source_language"].eq(language).all()
                or not frame["source_split"].eq(split).all()
            ):
                fail(f"{split}: invalid IDs, text, annotations, or provenance")
            if (
                len(frame) != metadata["split_counts"][split]["output_rows"]
                or prepared_fingerprint(frame)
                != metadata["prepared_fingerprints"][split]
            ):
                fail(f"{split}: prepared row count or fingerprint mismatch")
        for index, first in enumerate(SPLITS):
            for second in SPLITS[index + 1 :]:
                if set(frames[first]["id"]) & set(frames[second]["id"]) or set(
                    frames[first]["text"].str.lower()
                ) & set(frames[second]["text"].str.lower()):
                    fail(f"Prepared {first}/{second} partitions overlap")
    except ReferencePreparationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, pyarrow.ArrowException) as error:
        fail(f"Prepared reference data is missing or unreadable: {error}")
    return ReferenceDataset(
        train=frames["train"],
        dev=frames["dev"],
        test=frames["test"],
        label_names=labels,
        metadata=metadata,
    )
