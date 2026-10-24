"""Track C persistence and strict read-only loading, separate from A/B triplets."""

import json
import platform
import subprocess
from pathlib import Path

import pandas as pd
import pyarrow

from emotion_detection.datasets.acquisition import acquire_sources, read_source
from emotion_detection.datasets.reference_track_c import (
    C_COLUMNS,
    C_DEVIATIONS,
    C_POLICIES,
    C_ROLES,
    MASK_COLUMNS,
    PREPARATION_VARIANT,
    _identities,
    _keys,
    prepare_track_c,
    source_key,
)
from emotion_detection.datasets.reference_types import (
    CrosslingualReferenceDataset,
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import SIX_LABELS
from emotion_detection.datasets.sources_track_c import (
    DEFAULT_C_PROTOCOL,
    get_crosslingual_sources,
    track_c_source_languages,
)


def _directories(root, language, release, protocol):
    # Separate raw scope deliberately contains only A train/dev and C dev/test.
    # Existing A test files are neither acquired, validated nor used for cleaning.
    raw = root / "data/raw/reference_baselines" / release / "crosslingual"
    output = root / "data/processed/reference_baselines/track_c" / language / protocol
    analysis = (
        root / "output/analysis/reference_baselines/track_c" / language / protocol
    )
    paths = [p.resolve() for p in (raw, output, analysis)]
    for i, path in enumerate(paths):
        for other in paths[i + 1 :]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(
                    "Reference raw, processed and analysis directories must not overlap"
                )
    return raw, output, analysis


def prepare_crosslingual_dataset(
    language, *, release, project_root, interactive=None, protocol=DEFAULT_C_PROTOCOL
):
    from emotion_detection.datasets.reference import _write_audits

    sources = get_crosslingual_sources(language, release, protocol)
    root = Path(project_root).resolve()
    raw, output, analysis = _directories(root, language, release, protocol)
    acquire_sources(sources, raw, interactive=interactive)
    frames = {source_key(s): read_source(raw / s.relative_path, s) for s in sources}
    try:
        data, diagnostics = prepare_track_c(
            frames,
            sources=sources,
            language=language,
            release=release,
            protocol=protocol,
        )
    except ReferencePreparationError as error:
        _write_audits(
            analysis,
            sources,
            error.diagnostics,
            {"status": "failed", "reason": str(error)},
        )
        raise
    data.metadata["source_row_counts"] = {
        "/".join(source_key(s)): len(frames[source_key(s)]) for s in sources
    }
    data.metadata["environment"] = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "pyarrow": pyarrow.__version__,
    }
    data.metadata["code_revision"] = None
    data.metadata["code_dirty"] = None
    try:
        data.metadata["code_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        data.metadata["code_dirty"] = bool(
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
    for role in C_ROLES:
        getattr(data, role).to_parquet(output / f"{role}.parquet", index=False)
    _write_audits(analysis, sources, diagnostics, data.metadata)
    return data


def load_prepared_crosslingual_dataset(
    language, *, release, project_root, protocol=DEFAULT_C_PROTOCOL
):
    sources = get_crosslingual_sources(language, release, protocol)
    languages = track_c_source_languages(language, protocol)
    _, output, analysis = _directories(
        Path(project_root).resolve(), language, release, protocol
    )

    def fail(reason):
        raise ReferencePreparationError(
            f"{reason}. Run 'emotion data prepare --track c --language {language} "
            f"--protocol {protocol}' with the same --project-root and --release"
        )

    try:
        records = pd.read_csv(
            analysis / "metadata.csv", dtype=str, keep_default_na=False
        )
        if list(records.columns) != ["key", "value"] or not records.key.is_unique:
            fail("Invalid Track C metadata fields")
        metadata = {r.key: json.loads(r.value) for r in records.itertuples()}
        json.dumps(metadata, allow_nan=False)
        expected = {
            "schema_version": 1,
            "track": "c",
            "language": language,
            "release": release,
            "protocol": protocol,
            "source_languages": list(languages),
            "label_names": list(sources[-1].label_names),
            "model_label_names": list(SIX_LABELS),
            "preparation_variant": PREPARATION_VARIANT,
            "target_ids_and_order_preserved": True,
            "policies": C_POLICIES,
            "protocol_deviations": list(C_DEVIATIONS),
            "source_manifest": [s.metadata() for s in sources],
        }
        if any(metadata.get(k) != v for k, v in expected.items()):
            fail(
                "Track C metadata does not match the selected manifest/protocol/schema"
            )
        manifest = pd.read_csv(analysis / "source_manifest.csv", keep_default_na=False)
        expected_manifest = [
            {**s.metadata(), "label_names": "|".join(s.label_names)} for s in sources
        ]
        if manifest.to_dict("records") != expected_manifest:
            fail("Track C source manifest mismatch")
        row_counts = metadata["source_row_counts"]
        if set(row_counts) != {"/".join(source_key(s)) for s in sources} or any(
            not isinstance(n, int) or isinstance(n, bool) or n < 1
            for n in row_counts.values()
        ):
            fail("Track C source row counts mismatch")
        frames = {role: pd.read_parquet(output / f"{role}.parquet") for role in C_ROLES}
        native_schemas = {s.language: s.label_names for s in sources}
        for role, frame in frames.items():
            is_source = role in ("train", "source_dev")
            split = "train" if role == "train" else "test" if role == "test" else "dev"
            track = "a" if is_source else "c"
            permitted = set(languages) if is_source else {language}
            if frame.empty or tuple(frame.columns) != C_COLUMNS:
                fail(f"{role}: invalid columns or empty partition")
            if (
                not frame.source_language.isin(permitted).all()
                or not frame.source_track.eq(track).all()
                or not frame.source_split.eq(split).all()
                or frame[["source_language", "id"]].duplicated().any()
                or not frame.id.map(
                    lambda v: isinstance(v, str) and bool(v.strip())
                ).all()
                or not frame.text.map(
                    lambda v: isinstance(v, str) and bool(v.strip())
                ).all()
                or not pd.api.types.is_integer_dtype(frame.source_row.dtype)
            ):
                fail(f"{role}: invalid identity or provenance")
            for name, mask in zip(SIX_LABELS, MASK_COLUMNS, strict=True):
                if (
                    not pd.api.types.is_bool_dtype(frame[mask].dtype)
                    or frame[mask].isna().any()
                    or not frame[mask].eq(frame[name].notna()).all()
                    or not frame[name].dropna().isin([0, 1]).all()
                ):
                    fail(f"{role}: invalid annotation or known-label mask")
                for source_language in permitted:
                    if (
                        name not in native_schemas[source_language]
                        and frame.loc[
                            frame.source_language.eq(source_language), mask
                        ].any()
                    ):
                        fail(f"{role}: annotations outside the native schema")
            if not frame[list(MASK_COLUMNS)].any().any() or (
                role == "train" and not frame[list(MASK_COLUMNS)].any(axis=1).all()
            ):
                fail(f"{role}: missing supervision")
            for source_language, group in frame.groupby("source_language", sort=False):
                total = row_counts[f"{track}/{source_language}/{split}"]
                positions = group.source_row.tolist()
                if (
                    positions != sorted(set(positions))
                    or min(positions) < 0
                    or max(positions) >= total
                ):
                    fail(f"{role}: invalid source row positions")
                if not is_source and positions != list(range(total)):
                    fail(f"{role}: target population or order changed")
            count = metadata["split_counts"][role]
            input_rows = sum(
                row_counts[f"{track}/{name}/{split}"] for name in permitted
            )
            if (
                count["input_rows"] != input_rows
                or count["output_rows"] != len(frame)
                or input_rows
                != len(frame) + count["invalid_rows"] + count["overlap_rows_removed"]
                or count["unknown_annotation_cells"]
                != int(frame[list(SIX_LABELS)].isna().sum().sum())
                or (
                    not is_source
                    and (count["invalid_rows"] or count["overlap_rows_removed"])
                )
                or metadata["prepared_fingerprints"][role]
                != prepared_fingerprint(frame)
            ):
                fail(f"{role}: count or fingerprint mismatch")
        for first, second in (
            ("train", "source_dev"),
            ("train", "target_dev"),
            ("train", "test"),
            ("source_dev", "target_dev"),
            ("source_dev", "test"),
        ):
            if _identities(frames[first]) & _identities(frames[second]) or _keys(
                frames[first]
            ) & _keys(frames[second]):
                fail(f"Track C {first}/{second} leakage")
        if metadata["target_dev_test_overlap_texts"] != len(
            _keys(frames["target_dev"]) & _keys(frames["test"])
        ) or metadata["target_dev_test_overlap_ids"] != len(
            _identities(frames["target_dev"]) & _identities(frames["test"])
        ):
            fail("Track C target overlap diagnostics mismatch")
    except ReferencePreparationError:
        raise
    except (OSError, ValueError, KeyError, TypeError, pyarrow.ArrowException) as error:
        fail(f"Prepared Track C data is missing or unreadable: {error}")
    return CrosslingualReferenceDataset(
        **frames,
        label_names=tuple(metadata["label_names"]),
        model_label_names=SIX_LABELS,
        metadata=metadata,
    )
