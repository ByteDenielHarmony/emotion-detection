"""Lightweight argparse entry point; importing this module never loads a model."""

import argparse
import json
import math
import sys
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def _common_options(parser, *, defaults=False):
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd() if defaults else argparse.SUPPRESS,
        help="Project/data root (default: current directory)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False if defaults else argparse.SUPPRESS,
        help="Show a traceback on application errors",
    )


def _positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be positive and finite")
    return number


def _nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def _reference_selection(parser, *, training=False):
    parser.add_argument(
        "--track",
        default="a",
        help="Reference track (training: a only)"
        if training
        else "Reference track (a, b or c)",
    )
    parser.add_argument(
        "--language",
        default="eng",
        help="Registered language code (default eng; see: emotion data languages)",
    )
    parser.add_argument(
        "--release",
        help="Pinned dataset release (default: registered official release)",
    )
    if not training:
        parser.add_argument(
            "--protocol",
            help="Track C source-selection protocol (default: task-paper-family-v1)",
        )


def build_parser():
    parser = argparse.ArgumentParser(
        prog="emotion", description="Emotion detection experiment tools"
    )
    _common_options(parser, defaults=True)
    try:
        application_version = version("emotion-detection")
    except PackageNotFoundError:
        application_version = "development"
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {application_version}"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser(
        "doctor", help="Inspect package versions and compute devices"
    )
    _common_options(doctor)
    doctor.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    doctor.set_defaults(handler=_doctor)

    data = commands.add_parser(
        "data", help="Dataset acquisition, preparation, and readiness"
    )
    _common_options(data)
    data_commands = data.add_subparsers(dest="data_command", required=True)
    languages = data_commands.add_parser(
        "languages", help="List registered reference languages without downloading data"
    )
    _common_options(languages)
    languages.add_argument(
        "--track",
        default="a",
        help="Reference track (a, b or c; C lists targets, sun needs another source protocol)",
    )
    languages.add_argument("--release", help="Registered pinned dataset release")
    languages.set_defaults(handler=_list_languages)
    prepare = data_commands.add_parser(
        "prepare", help="Download, verify, and prepare datasets"
    )
    _common_options(prepare)
    prepare.add_argument(
        "--dataset", choices=("reference", "joint"), default="reference"
    )
    prepare.add_argument(
        "--track", help="Reference track (a, b or c); not applicable to joint data"
    )
    prepare.add_argument(
        "--language", help="Reference language (default eng); joint data is English"
    )
    prepare.add_argument(
        "--release", help="Pinned reference release; not applicable to joint data"
    )
    prepare.add_argument(
        "--protocol",
        help="Track C source-selection protocol (default: task-paper-family-v1)",
    )
    prepare.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt for manual recovery",
    )
    prepare.set_defaults(handler=_prepare_data)

    check = data_commands.add_parser(
        "check", help="Read-only reference dataset readiness checks"
    )
    _common_options(check)
    check.add_argument("--dataset", choices=("reference",), default="reference")
    _reference_selection(check)
    check.set_defaults(handler=_check_data)

    train = commands.add_parser("train", help="Run model training")
    _common_options(train)
    training_commands = train.add_subparsers(dest="training_command", required=True)
    reference = training_commands.add_parser(
        "reference", help="Train the Track A reference model"
    )
    _common_options(reference)
    _reference_selection(reference, training=True)
    reference.add_argument(
        "--model-name",
        help="Hub model ID or local path (default: FacebookAI/xlm-roberta-large)",
    )
    reference.add_argument(
        "--revision", help="Pretrained model revision; overrides known checkpoint pins"
    )
    reference.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    reference.add_argument(
        "--epochs",
        type=_positive_float,
        default=2,
        help="Training epochs (default: %(default)s)",
    )
    reference.add_argument(
        "--learning-rate",
        type=_positive_float,
        default=1e-5,
        help="Learning rate (default: %(default)s)",
    )
    reference.add_argument(
        "--monitor",
        choices=("macro_f1", "loss"),
        default="macro_f1",
        help="Development metric for checkpoint selection and early stopping (default: %(default)s)",
    )
    reference.add_argument(
        "--early-stopping-patience",
        type=_positive_int,
        help="Enable early stopping after this many non-improving epochs (disabled by default)",
    )
    reference.add_argument(
        "--early-stopping-min-delta",
        type=_nonnegative_float,
        default=0.0,
        help="Minimum metric improvement to reset patience; requires --early-stopping-patience (default: %(default)s)",
    )
    reference.add_argument(
        "--batch-size",
        type=_positive_int,
        default=2,
        help="Per-device microbatch size (default: %(default)s)",
    )
    reference.add_argument(
        "--gradient-accumulation-steps",
        type=_positive_int,
        default=4,
        help="Microbatches per optimizer update (default: %(default)s)",
    )
    reference.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Recompute activations to reduce VRAM (enabled by default)",
    )
    reference.add_argument(
        "--mixed-precision",
        choices=("auto", "no", "bf16", "fp16"),
        default="auto",
        help="Automatic CUDA BF16/FP16 or CPU FP32 by default",
    )
    reference.add_argument(
        "--max-length",
        type=_positive_int,
        default=256,
        help="Maximum input tokens (default: %(default)s)",
    )
    reference.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: %(default)s)"
    )
    reference.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Emotion probability threshold (default: %(default)s)",
    )
    reference.add_argument(
        "--run-id", help="Nonexisting run name; otherwise generate a unique ID"
    )
    reference.add_argument(
        "--local-files-only",
        action="store_true",
        help="Forbid model/tokenizer downloads; independent of dataset preparation",
    )
    reference.add_argument(
        "--prepare-data",
        action="store_true",
        help="Explicitly acquire and prepare the selected dataset before training",
    )
    reference.add_argument(
        "--non-interactive",
        action="store_true",
        help="Never prompt during --prepare-data",
    )
    reference.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip final held-out test evaluation after the best checkpoint is saved",
    )
    reference.set_defaults(handler=_train_reference)
    return parser


def _print_dataset(data):
    roles = (
        ("train", "source_dev", "target_dev", "test")
        if data.metadata["track"] == "c"
        else ("train", "dev", "test")
    )
    for split in roles:
        counts = data.metadata["split_counts"][split]
        print(
            f"{split}: {counts['input_rows']} source rows -> {len(getattr(data, split))} prepared rows"
        )
    print(f"Labels: {', '.join(data.label_names)}")
    print(f"Release: {data.metadata['release']}")
    print(f"Preparation variant: {data.metadata['preparation_variant']}")
    if data.metadata["track"] == "c":
        print(f"Protocol: {data.metadata['protocol']}")
        print(f"Source languages: {', '.join(data.metadata['source_languages'])}")
        print("Unknown labels retain nullable values and known_<emotion> masks")


def _doctor(args):
    from emotion_detection.diagnostics import inspect_environment

    print(json.dumps(inspect_environment(args.device), indent=2))
    return 0


def _list_languages(args):
    from emotion_detection.datasets import available_reference_languages
    from emotion_detection.datasets.sources import DEFAULT_RELEASE

    print(
        " ".join(
            available_reference_languages(args.track, args.release or DEFAULT_RELEASE)
        )
    )
    return 0


def _prepare_data(args):
    root = args.project_root.resolve()
    if args.dataset == "joint":
        if (
            args.track is not None
            or args.protocol is not None
            or args.release is not None
            or args.language not in (None, "eng")
        ):
            raise ValueError(
                "Joint preparation is English A+B; --track, --release and --protocol are reference-only"
            )
        from emotion_detection.datasets.prepare import main as prepare_joint

        arguments = [
            "--raw-dir",
            str(root / "data/raw"),
            "--output-dir",
            str(root / "data/processed"),
            "--analysis-dir",
            str(root / "output/analysis/data_preparation"),
        ]
        if args.non_interactive:
            arguments.append("--non-interactive")
        return prepare_joint(arguments)

    from emotion_detection.datasets import prepare_reference_dataset
    from emotion_detection.datasets.acquisition import AcquisitionError
    from emotion_detection.datasets.sources import DEFAULT_RELEASE

    try:
        data = prepare_reference_dataset(
            track=args.track or "a",
            language=args.language or "eng",
            release=args.release or DEFAULT_RELEASE,
            project_root=root,
            interactive=False if args.non_interactive else None,
            **({"protocol": args.protocol} if args.protocol is not None else {}),
        )
    except AcquisitionError as error:
        raise RuntimeError(str(error)) from error
    _print_dataset(data)
    return 0


def _check_data(args):
    from emotion_detection.datasets import load_prepared_reference_dataset
    from emotion_detection.datasets.sources import DEFAULT_RELEASE

    data = load_prepared_reference_dataset(
        track=args.track,
        language=args.language,
        release=args.release or DEFAULT_RELEASE,
        project_root=args.project_root,
        **({"protocol": args.protocol} if args.protocol is not None else {}),
    )
    _print_dataset(data)
    print(
        "Ready: source manifest, fingerprints, label masks, and source/target separation verified"
        if args.track == "c"
        else "Ready: source manifest, fingerprints, schemas, and split separation verified"
    )
    return 0


def _train_reference(args):
    from emotion_detection.datasets.sources import DEFAULT_RELEASE
    from emotion_detection.experiments.reference import (
        ReferenceTrainingOptions,
        train_reference,
    )

    keys = (
        "project_root",
        "track",
        "language",
        "model_name",
        "revision",
        "device",
        "epochs",
        "learning_rate",
        "monitor",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "batch_size",
        "gradient_accumulation_steps",
        "gradient_checkpointing",
        "mixed_precision",
        "max_length",
        "seed",
        "threshold",
        "run_id",
        "local_files_only",
        "prepare_data",
        "non_interactive",
        "skip_test",
    )
    options = ReferenceTrainingOptions(
        **{key: getattr(args, key) for key in keys},
        release=args.release or DEFAULT_RELEASE,
    )
    run = train_reference(options)
    print(f"Run: {run}")
    print(f"Best validation checkpoint: {run / 'checkpoint'}")
    if args.skip_test:
        print("Final test evaluation skipped (--skip-test)")
    else:
        print(f"Final test metrics: {run / 'test_metrics.json'}")
        print(f"Final test report: {run / 'test_report.json'}")
        print(f"Final test predictions: {run / 'test_predictions.csv'}")
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.handler(args)
    except KeyboardInterrupt:
        print("[ERROR] Interrupted; completed artifacts were kept", file=sys.stderr)
        return 130
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        if args.verbose:
            traceback.print_exc()
        print(f"[ERROR] {type(error).__name__}: {error}", file=sys.stderr)
        return 2
