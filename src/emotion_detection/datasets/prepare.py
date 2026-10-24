"""Acquire pinned English BRIGHTER inputs and prepare paired annotations.

Run from the repository root:
    uv run python src/emotion_detection/datasets/prepare.py
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import ssl
import sys
import tempfile
import time
from dataclasses import dataclass
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow
import pyarrow.parquet as pq

EMOTIONS = ("anger", "fear", "joy", "sadness", "surprise")
SPLITS = ("train", "dev", "test")
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
    "a": {
        "dataset": "brighter-dataset/BRIGHTER-emotion-categories",
        "revision": "419566ac8f46f951b51584be36c13e5363008144",
    },
    "b": {
        "dataset": "brighter-dataset/BRIGHTER-emotion-intensities",
        "revision": "08a5d61d3fc33036f97c8c76a61ff8d6f02f5157",
    },
}

INPUT_SHA256 = {
    ("a", "train"): "b6d668f4bc0081c22e5b4be0679498709676acbf6288a2d81037ab01ac1e28f8",
    ("a", "dev"): "82e6a08eebef0c1cf2b6fc30497b746b38566686e1468e1ff03c052c52c037ca",
    ("a", "test"): "d2ffa29d9f55227c4630a6287f1ac76325047abf2c21d7431e04190cc22c3548",
    ("b", "train"): "7764415133c7ed8edf1ec322af26e0e23849f5ff49ac4b834a4fcf881dfcf723",
    ("b", "dev"): "7b974a684fdb382cdd3f40ca86ea3ab1a36c63850e5ae0a81562e141070ecdd5",
    ("b", "test"): "c7becf513082560b6bb4f881fa7242e619ffc37fc6b3b01c40141b57c435016a",
}


@dataclass(frozen=True)
class SourceFile:
    track: str
    split: str
    sha256: str

    @property
    def label(self) -> str:
        return f"Track {self.track.upper()} / {self.split}"

    @property
    def relative_path(self) -> Path:
        return Path(f"track_{self.track}/eng/{self.split}-00000-of-00001.parquet")

    @property
    def url(self) -> str:
        source = SOURCES[self.track]
        return (
            f"https://huggingface.co/datasets/{source['dataset']}/resolve/"
            f"{source['revision']}/eng/{self.relative_path.name}"
        )


DATASET_FILES = tuple(
    SourceFile(track, split, INPUT_SHA256[track, split])
    for track in ("a", "b")
    for split in SPLITS
)


class DatasetAcquisitionError(Exception):
    """An expected acquisition problem that should not produce a traceback."""


class DatasetFileError(DatasetAcquisitionError):
    """An existing or downloaded file does not meet the pinned input contract."""


def verify_source_file(path: Path, source: SourceFile) -> None:
    """Verify actual content, including manually supplied files, without a network."""
    if not path.is_file():
        raise DatasetFileError(
            f"Required file is missing or is not a regular file: {path}"
        )
    try:
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
    except OSError as error:
        raise DatasetFileError(f"Cannot read {path}: {error}") from error
    if actual != source.sha256:
        raise DatasetFileError(
            f"Checksum mismatch for {source.label}. "
            f"Expected SHA-256 {source.sha256}; got {actual}."
        )
    try:
        with pq.ParquetFile(path) as parquet:
            required = {"id", "text", *EMOTIONS}
            missing = required - set(parquet.schema_arrow.names)
            if missing:
                raise DatasetFileError(f"Missing required columns: {sorted(missing)}")
            parquet.read(columns=["id", "text", *EMOTIONS])
    except (OSError, ValueError, pyarrow.ArrowException) as error:
        raise DatasetFileError(
            f"Cannot read Parquet data at {path}: {error}"
        ) from error


def download_source_file(
    raw_dir: Path,
    source: SourceFile,
    *,
    attempts: int = 3,
    timeout: float = 30,
) -> None:
    """Publish only verified files; never overwrite an existing destination."""
    destination = raw_dir / source.relative_path
    if destination.exists() or destination.is_symlink():
        verify_source_file(destination, source)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    last_reason = "Download failed"
    for attempt in range(1, attempts + 1):
        temporary = None
        retryable = True
        print(
            f"[INFO] Downloading {source.label} (attempt {attempt}/{attempts})",
            flush=True,
        )
        try:
            request = Request(
                source.url, headers={"User-Agent": "emotion-detection/0.1.0"}
            )
            with (
                urlopen(request, timeout=timeout) as response,
                tempfile.NamedTemporaryFile(
                    dir=destination.parent,
                    prefix=f".{destination.name}.",
                    suffix=".part",
                    delete=False,
                ) as stream,
            ):
                temporary = Path(stream.name)
                while chunk := response.read(64 * 1024):
                    stream.write(chunk)
            verify_source_file(temporary, source)
            try:
                # Same-directory hard linking publishes atomically and, unlike
                # rename/replace, cannot overwrite a concurrently supplied file.
                os.link(temporary, destination)
            except FileExistsError:
                verify_source_file(destination, source)
            print(f"[OK] Downloaded and verified {source.label}", flush=True)
            return
        except HTTPError as error:
            retryable = error.code in (408, 429) or 500 <= error.code < 600
            descriptions = {
                401: "authentication required",
                403: "access denied",
                404: "required file or revision not found",
                410: "source is no longer available",
            }
            last_reason = (
                f"HTTP {error.code}: {descriptions.get(error.code, error.reason)}"
            )
            error.close()
        except URLError as error:
            last_reason = f"Network error: {error.reason}"
        except (TimeoutError, ConnectionError, HTTPException, ssl.SSLError) as error:
            last_reason = f"Connection interrupted or timed out: {error}"
        except DatasetFileError as error:
            last_reason = str(error)
        except OSError as error:
            raise DatasetAcquisitionError(
                f"Cannot save {source.label} at {destination}: {error}"
            ) from error
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not retryable or attempt == attempts:
            break
        print(f"[WARNING] {source.label}: {last_reason}. Retrying...", file=sys.stderr)
        time.sleep(attempt)
    raise DatasetAcquisitionError(
        f"Unable to download {source.label}. Reason: {last_reason}"
    )


def validate_input_layout(raw_dir: Path) -> None:
    """Do not feed unverified additional shards into the existing glob-based loader."""
    expected = {source.relative_path for source in DATASET_FILES}
    for track in ("a", "b"):
        for split in SPLITS:
            for path in (raw_dir / f"track_{track}/eng").glob(f"{split}-*.parquet"):
                if path.relative_to(raw_dir) not in expected:
                    raise DatasetAcquisitionError(
                        f"Unexpected input shard: {path}. Move it aside before continuing; "
                        "the pinned English datasets require exactly the six expected filenames."
                    )


def ensure_raw_data(raw_dir: Path, *, interactive: bool | None = None) -> None:
    """Acquire missing inputs, then offer manual recovery without losing progress."""
    raw_dir = raw_dir.resolve()
    if interactive is None:
        interactive = sys.stdin.isatty()
    validate_input_layout(raw_dir)
    pending = []
    for source in DATASET_FILES:
        path = raw_dir / source.relative_path
        try:
            if path.exists() or path.is_symlink():
                verify_source_file(path, source)
                print(f"[OK] Using verified local file: {source.label}", flush=True)
            else:
                download_source_file(raw_dir, source)
        except DatasetAcquisitionError as error:
            pending.append((source, str(error)))

    while pending:
        for source, reason in pending:
            path = raw_dir / source.relative_path
            metadata = SOURCES[source.track]
            print(
                f"\n[ERROR] {source.label}: {reason}\n"
                f"Dataset: {metadata['dataset']}\n"
                f"Required revision: {metadata['revision']}\n"
                f"Source: {source.url}\n"
                f"Required SHA-256: {source.sha256}\n"
                f"Find the required file manually and place it at:\n{path}\n"
                "Existing files have not been overwritten. If an existing file is invalid, "
                "move it aside before supplying the correct file.",
                file=sys.stderr,
            )
        if not interactive:
            raise DatasetAcquisitionError(
                "Input files are not ready and interactive input is unavailable. "
                "Place the files at the paths above and rerun the same command to resume."
            )
        try:
            answer = input(
                "\nPlace the files, then press Enter to check again; "
                "enter q to exit (rerun the command later to resume): "
            )
        except EOFError:
            answer = "q"
        if answer.strip().lower() in {"q", "quit", "exit"}:
            raise DatasetAcquisitionError(
                "Stopped without preparing data. Existing files were kept. "
                "Rerun the same command to resume."
            )
        remaining = []
        for source, _ in pending:
            try:
                verify_source_file(raw_dir / source.relative_path, source)
                print(
                    f"[OK] Verified manually supplied file: {source.label}", flush=True
                )
            except DatasetFileError as error:
                remaining.append((source, str(error)))
        pending = remaining
    # Manual restoration may have introduced additional files after preflight.
    validate_input_layout(raw_dir)


def audit(frame: pd.DataFrame) -> dict:
    """Describe inputs before cleaning; do not silently repair annotations."""
    return {
        "rows": len(frame),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "null_counts": {column: int(n) for column, n in frame.isna().sum().items()},
        "unique_ids": int(frame["id"].nunique()),
        "label_values": {
            emotion: sorted(
                frame[emotion].dropna().unique().tolist(),
                key=lambda value: (type(value).__name__, str(value)),
            )
            for emotion in EMOTIONS
        },
    }


def clean_track(frame: pd.DataFrame, track: str, split: str):
    """Collapse label-identical copies and quarantine conflicting text groups.

    Equality is Python str.lower() only. Original text and every source ID are
    retained. Annotation agreement is used to detect duplicates within a track,
    never to choose a cross-track match.
    """
    required = {"id", "text", *EMOTIONS}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Track {track}/{split}: missing columns {sorted(missing)}")
    if (
        frame["id"].isna().any()
        or not frame["id"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise ValueError(f"Track {track}/{split}: IDs must be nonempty strings")
    if frame["id"].duplicated().any():
        raise ValueError(f"Track {track}/{split}: source IDs are not unique")
    if "disgust" in frame and frame["disgust"].notna().any():
        raise ValueError(f"Track {track}/{split}: unexpected non-null disgust labels")

    frame = frame[["id", "text", *EMOTIONS]].copy()
    diagnostics = []
    invalid_indices = []
    valid_values = range(2 if track == "a" else 4)
    for index, row in frame.iterrows():
        reasons = []
        if not isinstance(row["text"], str) or not row["text"].strip():
            reasons.append("missing_or_blank_text")
        for emotion in EMOTIONS:
            if pd.isna(row[emotion]) or row[emotion] not in valid_values:
                reasons.append(f"invalid_{emotion}")
        if reasons:
            invalid_indices.append(index)
            diagnostics.append(
                {
                    "track": track,
                    "split": split,
                    "reason": ";".join(reasons),
                    **row.to_dict(),
                }
            )
    valid = frame.drop(index=invalid_indices).copy()
    valid["text_key"] = valid["text"].map(str.lower)
    records = []
    conflict_keys = set()
    duplicate_groups = collapsed = 0
    for text_key, group in valid.groupby("text_key", sort=True):
        duplicate_groups += int(len(group) > 1)
        if len(group[list(EMOTIONS)].drop_duplicates()) > 1:
            conflict_keys.add(text_key)
            for _, row in group.iterrows():
                diagnostics.append(
                    {
                        "track": track,
                        "split": split,
                        "reason": "conflicting_annotations_for_text_key",
                        **row.to_dict(),
                    }
                )
            continue
        # Prefer non-Track-C provenance, then lexical ID order. Selection is
        # deterministic and only happens after target annotations agree.
        group = group.assign(
            _track_c=group["id"].str.contains("_track_c_", regex=False)
        ).sort_values(["_track_c", "id"])
        representative = group.iloc[0]
        records.append(
            {
                "text_key": text_key,
                "id": representative["id"],
                "source_ids": sorted(group["id"].tolist()),
                "text": representative["text"],
                **{emotion: int(representative[emotion]) for emotion in EMOTIONS},
            }
        )
        collapsed += len(group) - 1
    columns = ["text_key", "id", "source_ids", "text", *EMOTIONS]
    cleaned = pd.DataFrame(records, columns=columns)
    stats = {
        "input_rows": len(frame),
        "invalid_rows": len(invalid_indices),
        "valid_unique_text_keys": int(valid["text_key"].nunique()),
        "duplicate_text_groups": duplicate_groups,
        "conflicting_text_groups": len(conflict_keys),
        "conflicting_rows": int(valid["text_key"].isin(conflict_keys).sum()),
        "identical_annotation_copies_collapsed": collapsed,
        "retained_keys": len(cleaned),
    }
    assert len(frame) == (
        stats["invalid_rows"] + stats["conflicting_rows"] + collapsed + len(cleaned)
    )
    return cleaned, stats, diagnostics, conflict_keys


def merge_tracks(a: pd.DataFrame, b: pd.DataFrame, split: str):
    """Join unique keys within one split, never by ID or annotation values."""
    merged = (
        a.merge(
            b,
            on="text_key",
            how="inner",
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        .sort_values("text_key")
        .reset_index(drop=True)
    )
    result = pd.DataFrame(
        {
            "split": pd.Series([split] * len(merged), dtype="string"),
            "text_key": merged["text_key"],
            "id_a": merged["id_a"],
            "id_b": merged["id_b"],
            "source_ids_a": merged["source_ids_a"],
            "source_ids_b": merged["source_ids_b"],
            "text": merged["text_b"],
            "text_a": merged["text_a"],
            "text_b": merged["text_b"],
        }
    )
    for emotion in EMOTIONS:
        result[f"{emotion}_present"] = merged[f"{emotion}_a"].astype("int64")
        result[f"{emotion}_intensity"] = merged[f"{emotion}_b"].astype("int64")
    inconsistent = pd.Series(False, index=result.index)
    for emotion in EMOTIONS:
        inconsistent |= result[f"{emotion}_present"].ne(
            result[f"{emotion}_intensity"].gt(0).astype("int64")
        )
    # Do not fabricate agreement or silently discard a valid cross-track pair.
    result["presence_intensity_consistent"] = ~inconsistent
    return result


def exclude_cross_split_overlap(
    frames: dict[str, pd.DataFrame],
    source_keys: dict[str, set[str]],
):
    """Preserve test over dev over train, including unpaired source records."""
    higher_keys = set()
    diagnostics = []
    counts = {}
    for split in reversed(SPLITS):
        frame = frames[split]
        overlap = frame["text_key"].isin(higher_keys)
        for row in frame.loc[overlap].to_dict("records"):
            diagnostics.append(
                {"reason": "text_in_higher_priority_source_split", **row}
            )
        frames[split] = frame.loc[~overlap].reset_index(drop=True)
        counts[split] = int(overlap.sum())
        higher_keys.update(source_keys[split])
    return counts, diagnostics


def flatten_summary(value, prefix=""):
    """Represent nested audit metadata as scalar metric/value CSV rows."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten_summary(item, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from flatten_summary(item, f"{prefix}.{index}")
    else:
        yield {"metric": prefix, "value": value}


def write_diagnostic_csv(records, columns, path: Path) -> None:
    """Keep headers even when empty; serialize provenance ID lists with '|'."""
    frame = pd.DataFrame(records, columns=columns)
    for column in ("source_ids", "source_ids_a", "source_ids_b"):
        if column in frame:
            frame[column] = frame[column].map(lambda ids: "|".join(ids))
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def validate_directories(raw_dir: Path, output_dir: Path, analysis_dir: Path) -> None:
    """Reject unsafe paths before downloading or writing any data."""
    resolved = [p.resolve() for p in (raw_dir, output_dir, analysis_dir)]
    for i, left in enumerate(resolved):
        for right in resolved[i + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    "Raw, processed and analysis directories must not overlap"
                )


def prepare(raw_dir: Path, output_dir: Path, analysis_dir: Path) -> dict:
    """Audit, clean and merge local input files; write deterministic outputs."""
    validate_directories(raw_dir, output_dir, analysis_dir)
    summary = {
        "schema_version": 1,
        "environment": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "expected_sources": SOURCES,
        "text_key_policy": "Python str.lower(); no stripping or whitespace/URL normalization",
        "cross_split_priority": ["test", "dev", "train"],
        "inputs": {},
        "splits": {},
    }
    prepared = {}
    excluded = []
    unmatched = []
    mismatches = []
    # Include keys from all valid source texts, not just successful merged pairs.
    # Thus an unmatched/conflicting evaluation record cannot leak into training.
    source_keys = {split: set() for split in SPLITS}
    for split in SPLITS:
        raw_frames = {}
        cleaned = {}
        conflicts = {}
        split_stats = {}
        for track in ("a", "b"):
            paths = sorted(
                (raw_dir / f"track_{track}" / "eng").glob(f"{split}-*.parquet")
            )
            if not paths:
                raise FileNotFoundError(
                    f"No Track {track}/{split} Parquet files under {raw_dir}"
                )
            frame = pd.concat(
                [pd.read_parquet(path) for path in paths], ignore_index=True
            )
            raw_frames[track] = frame
            required = {"id", "text", *EMOTIONS}
            if not required.issubset(frame.columns):
                raise ValueError(
                    f"Track {track}/{split}: missing columns {sorted(required - set(frame.columns))}"
                )
            summary["inputs"][f"{track}/{split}"] = {
                **audit(frame),
                "files": [
                    {
                        "path": str(path.relative_to(raw_dir)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in paths
                ],
            }
            source_keys[split].update(
                value.lower()
                for value in frame["text"]
                if isinstance(value, str) and value.strip()
            )
            clean, stats, diagnostics, conflict = clean_track(frame, track, split)
            cleaned[track] = clean
            conflicts[track] = conflict
            split_stats[track] = stats
            excluded.extend(diagnostics)

        raw_a, raw_b = raw_frames["a"], raw_frames["b"]
        text_a = {text for text in raw_a["text"] if isinstance(text, str)}
        text_b = {text for text in raw_b["text"] if isinstance(text, str)}
        split_stats["matching_audit"] = {
            "shared_source_ids": len(set(raw_a["id"]) & set(raw_b["id"])),
            "shared_exact_text_keys": len(text_a & text_b),
            "shared_lowercase_text_keys": len(
                {text.lower() for text in text_a} & {text.lower() for text in text_b}
            ),
        }

        for track, other in (("a", "b"), ("b", "a")):
            absent = ~cleaned[track]["text_key"].isin(cleaned[other]["text_key"])
            for row in cleaned[track].loc[absent].to_dict("records"):
                reason = (
                    "counterpart_has_conflicting_annotations"
                    if row["text_key"] in conflicts[other]
                    else "no_counterpart_text_key"
                )
                unmatched.append(
                    {"track": track, "split": split, "reason": reason, **row}
                )
            split_stats[track]["unmatched_retained_keys"] = int(absent.sum())

        paired = merge_tracks(cleaned["a"], cleaned["b"], split)
        mismatches.extend(
            paired.loc[~paired["presence_intensity_consistent"]].to_dict("records")
        )
        split_stats["merged_before_overlap_removal"] = len(paired)
        split_stats["presence_intensity_mismatches"] = int(
            (~paired["presence_intensity_consistent"]).sum()
        )
        prepared[split] = paired
        summary["splits"][split] = split_stats

    # Drop from a lower-priority split if the key appears anywhere in a raw
    # higher-priority split, including an unmatched or quarantined record.
    overlap_counts, overlap_records = exclude_cross_split_overlap(prepared, source_keys)
    for split in SPLITS:
        summary["splits"][split]["cross_split_rows_removed"] = overlap_counts[split]

    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    for split, frame in prepared.items():
        if frame["text_key"].duplicated().any():
            raise AssertionError(f"Duplicate output keys in {split}")
        frame.to_parquet(output_dir / f"{split}.parquet", index=False)
        summary["splits"][split]["final_rows"] = len(frame)
    paired_columns = list(prepared["train"].columns)
    for name, (records, columns) in {
        "excluded_records": (
            excluded,
            ["track", "split", "reason", "id", "text", "text_key", *EMOTIONS],
        ),
        "unmatched_records": (
            unmatched,
            [
                "track",
                "split",
                "reason",
                "id",
                "source_ids",
                "text",
                "text_key",
                *EMOTIONS,
            ],
        ),
        "cross_split_overlap": (overlap_records, ["reason", *paired_columns]),
        "presence_intensity_mismatches": (mismatches, paired_columns),
    }.items():
        write_diagnostic_csv(records, columns, analysis_dir / f"{name}.csv")
        summary[f"{name}_count"] = len(records)
    pd.DataFrame(flatten_summary(summary), columns=["metric", "value"]).to_csv(
        analysis_dir / "summary.csv",
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=REPOSITORY_ROOT / "data/raw")
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "data/processed"
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=REPOSITORY_ROOT / "output/analysis/data_preparation",
    )
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for manual recovery; exit with an error if inputs are unavailable.",
    )
    args = parser.parse_args(argv)
    try:
        validate_directories(args.raw_dir, args.output_dir, args.analysis_dir)
        ensure_raw_data(
            args.raw_dir,
            interactive=False if args.non_interactive else None,
        )
        summary = prepare(args.raw_dir, args.output_dir, args.analysis_dir)
    except (
        DatasetAcquisitionError,
        OSError,
        ValueError,
        pyarrow.ArrowException,
    ) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\n[ERROR] Interrupted. Existing raw files were kept; rerun the same command to resume.",
            file=sys.stderr,
        )
        return 130
    for split in SPLITS:
        stats = summary["splits"][split]
        print(
            f"{split}: {stats['final_rows']} final rows "
            f"({stats['merged_before_overlap_removal']} paired before overlap removal)"
        )
    print(f"Audit and diagnostics: {args.analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
