"""CLI for independent reference data, not for model training or joint A+B data."""

import argparse
import sys
from pathlib import Path

import pyarrow

from emotion_detection.datasets.acquisition import AcquisitionError
from emotion_detection.datasets.reference import PROJECT_ROOT, load_reference_dataset
from emotion_detection.datasets.sources import (
    DEFAULT_RELEASE,
    available_reference_languages,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", default="a")
    parser.add_argument("--language", default="eng")
    parser.add_argument("--release", default=DEFAULT_RELEASE)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--non-interactive", action="store_true")
    parser.add_argument(
        "--list-languages",
        action="store_true",
        help="List supported languages for this track/release without downloading data.",
    )
    args = parser.parse_args(argv)
    try:
        if args.list_languages:
            print(" ".join(available_reference_languages(args.track, args.release)))
            return 0
        data = load_reference_dataset(
            track=args.track,
            language=args.language,
            release=args.release,
            project_root=args.project_root,
            interactive=False if args.non_interactive else None,
        )
    except (AcquisitionError, OSError, ValueError, pyarrow.ArrowException) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print(
            "\n[ERROR] Interrupted. Verified raw files were kept; rerun to resume.",
            file=sys.stderr,
        )
        return 130
    for split in ("train", "dev", "test"):
        counts = data.metadata["split_counts"][split]
        print(
            f"{split}: {counts['input_rows']} source rows -> {counts['output_rows']} prepared rows"
        )
    print(f"Labels: {', '.join(data.label_names)}")
    print(f"Preparation variant: {data.metadata['preparation_variant']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
