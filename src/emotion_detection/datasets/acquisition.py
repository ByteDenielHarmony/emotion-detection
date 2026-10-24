"""Language/format-aware acquisition for independent reference datasets.

Uses the same verified-temporary-download/manual-fallback approach as the legacy
joint pipeline, which remains untouched. This module has no model dependency.
"""

import csv
import hashlib
import os
import ssl
import sys
import tempfile
import time
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import pyarrow

from emotion_detection.datasets.sources import SourceSpec


class AcquisitionError(Exception):
    """Expected acquisition failure suitable for a concise CLI message."""


def read_source(path: Path, source: SourceSpec) -> pd.DataFrame:
    if source.file_format == "csv":
        # Preserve literal strings such as NA/NULL and leading zeros in IDs.
        with path.open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream), [])
        if len(header) != len(set(header)):
            raise AcquisitionError(f"Duplicate CSV column names: {path}")
        return pd.read_csv(
            path,
            dtype={"id": str, "text": str},
            keep_default_na=False,
            encoding="utf-8-sig",
        )
    return pd.read_parquet(path)


def verify_source(path: Path, source: SourceSpec) -> None:
    try:
        if not path.is_file():
            raise AcquisitionError(
                f"Required file is missing or not a regular file: {path}"
            )
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != source.sha256:
            raise AcquisitionError(
                f"Checksum mismatch for {source.label}: expected {source.sha256}; got {digest}"
            )
        frame = read_source(path, source)
        missing = {"id", "text", *source.label_names} - set(frame.columns)
        if missing:
            raise AcquisitionError(
                f"Missing required columns in {path}: {sorted(missing)}"
            )
    except (OSError, ValueError, pyarrow.ArrowException) as error:
        raise AcquisitionError(f"Cannot read {path}: {error}") from error


def download_source(source: SourceSpec, destination: Path, *, attempts=3, timeout=30):
    if destination.exists() or destination.is_symlink():
        verify_source(destination, source)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    reason = "download failed"
    for attempt in range(1, attempts + 1):
        temporary = None
        retryable = True
        print(
            f"[INFO] Downloading {source.label} (attempt {attempt}/{attempts})",
            flush=True,
        )
        try:
            with (
                urlopen(
                    Request(
                        source.url, headers={"User-Agent": "emotion-detection/0.1.0"}
                    ),
                    timeout=timeout,
                ) as response,
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
            verify_source(temporary, source)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                verify_source(destination, source)
            print(f"[OK] Downloaded and verified {source.label}", flush=True)
            return
        except HTTPError as error:
            reason = f"HTTP {error.code}: {error.reason}"
            retryable = error.code in (408, 429) or 500 <= error.code < 600
            error.close()
        except URLError as error:
            reason = f"Network error: {error.reason}"
        except (
            TimeoutError,
            ConnectionError,
            HTTPException,
            ssl.SSLError,
            AcquisitionError,
        ) as error:
            reason = str(error)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        if not retryable or attempt == attempts:
            break
        print(f"[WARNING] {source.label}: {reason}. Retrying...", file=sys.stderr)
        time.sleep(attempt)
    raise AcquisitionError(f"Unable to download {source.label}: {reason}")


def validate_layout(sources, root: Path):
    expected = {root / source.relative_path for source in sources}
    for parent in {path.parent for path in expected}:
        if parent.exists():
            for path in parent.iterdir():
                if (
                    path.is_file()
                    and path.suffix in {".csv", ".parquet"}
                    and path not in expected
                ):
                    raise AcquisitionError(
                        f"Unexpected reference input: {path}. Move it aside; releases must not be mixed."
                    )


def acquire_sources(sources, root: Path, *, interactive=None):
    """Verify all selected files, acquiring only missing ones. Never replace raw data."""
    sources = tuple(sources)
    if not sources or len({s.relative_path for s in sources}) != len(sources):
        raise ValueError("Acquisition needs nonempty, unique source destinations")
    if len({s.release for s in sources}) != 1:
        raise ValueError("Do not mix source releases")
    root = Path(root).resolve()
    interactive = sys.stdin.isatty() if interactive is None else interactive
    validate_layout(sources, root)
    pending = []
    for source in sources:
        path = root / source.relative_path
        try:
            if path.exists() or path.is_symlink():
                verify_source(path, source)
                print(f"[OK] Using verified local file: {source.label}", flush=True)
            else:
                download_source(source, path)
        except AcquisitionError as error:
            pending.append((source, str(error)))
    while pending:
        for source, reason in pending:
            print(
                f"\n[ERROR] {source.label}: {reason}\nRepository: {source.repository}\n"
                f"Revision: {source.revision}\nSource: {source.url}\nSHA-256: {source.sha256}\n"
                f"Find this file manually and place it at:\n{root / source.relative_path}\n"
                "Existing files have not been overwritten. Move an invalid file aside before replacing it.",
                file=sys.stderr,
            )
        if not interactive:
            raise AcquisitionError(
                "Inputs are not ready. Place the files at the paths above and rerun the same command to resume."
            )
        try:
            answer = input(
                "Press Enter after placing the files to check again, or q to exit: "
            )
        except EOFError:
            answer = "q"
        if answer.strip().lower() in {"q", "quit", "exit"}:
            raise AcquisitionError(
                "Stopped. Existing files were kept; rerun the same command to resume."
            )
        remaining = []
        for source, _ in pending:
            try:
                verify_source(root / source.relative_path, source)
            except AcquisitionError as error:
                remaining.append((source, str(error)))
        pending = remaining
    validate_layout(sources, root)
