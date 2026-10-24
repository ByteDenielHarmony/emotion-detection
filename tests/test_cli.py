"""CLI and readiness tests use synthetic files and never download model weights."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from emotion_detection.cli import main
from emotion_detection.datasets import available_reference_languages, reference
from emotion_detection.datasets.reference_types import ReferencePreparationError
from emotion_detection.datasets.sources import ENGLISH_LABELS, get_reference_sources


def seed_reference_inputs(tmp_path, monkeypatch, language="eng"):
    specs = []
    for source in get_reference_sources("a", language):
        raw = (
            tmp_path
            / "data/raw/reference_baselines"
            / source.release
            / source.relative_path
        )
        raw.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {
                    "id": f"{source.split}-{index}",
                    "text": f"{source.split} example {index}",
                    **{
                        name: int(name == "joy" and index == 0)
                        for name in source.label_names
                    },
                }
                for index in range(2)
            ]
        ).to_csv(raw, index=False)
        specs.append(
            replace(source, sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
        )
    monkeypatch.setattr(
        reference, "get_reference_sources", lambda *a, **kw: tuple(specs)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("CLI tests must not access the network")

    monkeypatch.setattr("emotion_detection.datasets.acquisition.urlopen", forbidden)
    return tmp_path


@pytest.fixture
def local_reference(tmp_path, monkeypatch):
    return seed_reference_inputs(tmp_path, monkeypatch)


def snapshot(root):
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


@pytest.mark.parametrize(
    "command",
    [[], ["data"], ["data", "languages"], ["data", "prepare"], ["train", "reference"]],
)
def test_help_is_lightweight_and_model_free(command, tmp_path):
    script = (
        "import sys\n"
        "from emotion_detection.cli import main\n"
        "try:\n"
        "    main(sys.argv[1:])\n"
        "except SystemExit as error:\n"
        "    assert error.code == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
        "assert 'pandas' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, *command, "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: emotion" in result.stdout
    assert not list(tmp_path.iterdir())


def test_module_entry_point_from_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "emotion_detection", "--version"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("emotion ")


def test_prepare_and_check_commands(local_reference, capsys):
    root = local_reference
    assert (
        main(
            [
                "data",
                "prepare",
                "--dataset",
                "reference",
                "--project-root",
                str(root),
                "--track",
                "a",
                "--language",
                "eng",
                "--non-interactive",
            ]
        )
        == 0
    )
    assert "2 source rows -> 2 prepared rows" in capsys.readouterr().out
    before = snapshot(root)
    assert (
        main(
            [
                "--project-root",
                str(root),
                "data",
                "check",
                "--track",
                "a",
                "--language",
                "eng",
            ]
        )
        == 0
    )
    assert "Ready:" in capsys.readouterr().out
    assert snapshot(root) == before
    loaded = reference.load_prepared_reference_dataset(project_root=root)
    assert loaded.label_names == ENGLISH_LABELS
    assert loaded.metadata["track"] == "a"
    assert snapshot(root) == before


def test_language_discovery_does_not_load_or_prepare_data(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("Language discovery must not load, download, or prepare data")

    for name in (
        "prepare_reference_dataset",
        "load_reference_dataset",
        "load_prepared_reference_dataset",
    ):
        monkeypatch.setattr(f"emotion_detection.datasets.{name}", forbidden)
    monkeypatch.setattr("emotion_detection.datasets.acquisition.urlopen", forbidden)
    assert (
        main(["data", "languages", "--track", "a", "--project-root", str(tmp_path)])
        == 0
    )
    assert tuple(capsys.readouterr().out.split()) == available_reference_languages()
    assert len(available_reference_languages()) == 28
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("track", ["b", "c"])
def test_language_discovery_rejects_unimplemented_tracks(track, tmp_path, capsys):
    assert (
        main(["data", "languages", "--track", track, "--project-root", str(tmp_path)])
        == 2
    )
    assert "not implemented" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("language", available_reference_languages())
def test_all_registered_languages_support_preparation_and_read_only_checks(
    language,
    tmp_path,
    monkeypatch,
    capsys,
):
    root = seed_reference_inputs(tmp_path, monkeypatch, language)
    expected_labels = get_reference_sources("a", language)[0].label_names
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "a",
                "--language",
                language,
                "--project-root",
                str(root),
                "--non-interactive",
            ]
        )
        == 0
    )
    capsys.readouterr()
    before = snapshot(root)
    assert (
        main(
            [
                "data",
                "check",
                "--track",
                "a",
                "--language",
                language,
                "--project-root",
                str(root),
            ]
        )
        == 0
    )
    assert "Ready:" in capsys.readouterr().out
    data = reference.load_prepared_reference_dataset("a", language, project_root=root)
    assert data.label_names == expected_labels
    assert data.metadata["language"] == language
    assert snapshot(root) == before
    from emotion_detection.reference_baselines.data import validate_frame

    validate_frame(data.train, data.label_names)
    if language == "afr":
        assert "disgust" in data.label_names and "surprise" not in data.label_names
    if language == "eng":
        assert "surprise" in data.label_names and "disgust" not in data.label_names


def test_missing_data_check_fails_without_writes(tmp_path, capsys):
    assert main(["data", "check", "--project-root", str(tmp_path)]) == 2
    error = capsys.readouterr().err
    assert "emotion data prepare" in error
    assert "Traceback" not in error
    assert not list(tmp_path.iterdir())


def test_read_only_loading_does_not_require_raw_files(local_reference):
    root = local_reference
    reference.prepare_reference_dataset(project_root=root, interactive=False)
    for path in (root / "data/raw/reference_baselines").rglob("*.csv"):
        path.unlink()
    before = snapshot(root)
    data = reference.load_prepared_reference_dataset(project_root=root)
    assert len(data.train) == 2
    assert snapshot(root) == before


@pytest.mark.parametrize("tamper", ["frame", "metadata", "manifest", "missing"])
def test_read_only_loader_rejects_stale_or_modified_artifacts(local_reference, tamper):
    root = local_reference
    reference.prepare_reference_dataset(project_root=root, interactive=False)
    analysis = root / "output/analysis/reference_baselines/track_a/eng"
    train = root / "data/processed/reference_baselines/track_a/eng/train.parquet"
    if tamper == "frame":
        frame = pd.read_parquet(train)
        frame.loc[0, "joy"] = 1 - frame.loc[0, "joy"]
        frame.to_parquet(train, index=False)
    elif tamper == "metadata":
        records = pd.read_csv(analysis / "metadata.csv", keep_default_na=False)
        records.loc[records.key == "language", "value"] = json.dumps("deu")
        records.to_csv(analysis / "metadata.csv", index=False)
    elif tamper == "manifest":
        records = pd.read_csv(analysis / "source_manifest.csv", keep_default_na=False)
        records.loc[0, "sha256"] = "0" * 64
        records.to_csv(analysis / "source_manifest.csv", index=False)
    else:
        train.unlink()
    before = snapshot(root)
    with pytest.raises(ReferencePreparationError, match="emotion data prepare"):
        reference.load_prepared_reference_dataset(project_root=root)
    assert snapshot(root) == before


def test_joint_prepare_delegates_to_existing_pipeline(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "emotion_detection.datasets.prepare.main", lambda argv: calls.append(argv) or 0
    )
    assert (
        main(
            [
                "data",
                "prepare",
                "--dataset",
                "joint",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 0
    )
    assert calls == [
        [
            "--raw-dir",
            str(tmp_path / "data/raw"),
            "--output-dir",
            str(tmp_path / "data/processed"),
            "--analysis-dir",
            str(tmp_path / "output/analysis/data_preparation"),
            "--non-interactive",
        ]
    ]


def test_joint_rejects_reference_only_flags(tmp_path, capsys):
    assert (
        main(
            [
                "data",
                "prepare",
                "--dataset",
                "joint",
                "--track",
                "a",
                "--project-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "reference-only" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_cli_preserves_preparation_errors_and_interrupts(monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise ValueError("bad manifest")

    monkeypatch.setattr("emotion_detection.datasets.prepare_reference_dataset", fail)
    assert main(["data", "prepare"]) == 2
    assert "Traceback" not in capsys.readouterr().err
    assert main(["data", "prepare", "--verbose"]) == 2
    assert "Traceback" in capsys.readouterr().err

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        "emotion_detection.datasets.prepare_reference_dataset", interrupt
    )
    assert main(["data", "prepare"]) == 130


def test_training_defaults_and_no_implicit_preparation(tmp_path, monkeypatch, capsys):
    from emotion_detection.experiments import reference as experiment

    calls = []

    def run(options):
        calls.append(options)
        return Path(options.project_root) / "output/runs/example"

    monkeypatch.setattr(experiment, "train_reference", run)
    assert main(["train", "reference", "--project-root", str(tmp_path)]) == 0
    options = calls[0]
    assert options.batch_size == 2
    assert options.gradient_accumulation_steps == 4
    assert options.gradient_checkpointing is True
    assert options.prepare_data is False
    assert options.monitor == "macro_f1"
    assert options.early_stopping_patience is None
    assert options.early_stopping_min_delta == 0
    assert "Final test evaluation was not run" in capsys.readouterr().out


def test_training_flags_are_forwarded(tmp_path, monkeypatch):
    from emotion_detection.experiments import reference as experiment

    calls = []
    monkeypatch.setattr(
        experiment, "train_reference", lambda options: calls.append(options) or tmp_path
    )
    assert (
        main(
            [
                "train",
                "reference",
                "--model-name",
                "local-model",
                "--revision",
                "fixed",
                "--device",
                "cpu",
                "--epochs",
                "1",
                "--learning-rate",
                "0.001",
                "--batch-size",
                "1",
                "--gradient-accumulation-steps",
                "8",
                "--no-gradient-checkpointing",
                "--mixed-precision",
                "no",
                "--max-length",
                "32",
                "--seed",
                "7",
                "--threshold",
                "0.4",
                "--run-id",
                "example",
                "--local-files-only",
                "--prepare-data",
                "--non-interactive",
                "--monitor",
                "loss",
                "--early-stopping-patience",
                "3",
                "--early-stopping-min-delta",
                "0.001",
            ]
        )
        == 0
    )
    options = calls[0]
    assert options.model_name == "local-model" and options.revision == "fixed"
    assert options.device == "cpu" and options.max_length == 32 and options.seed == 7
    assert options.gradient_accumulation_steps == 8 and options.batch_size == 1
    assert options.gradient_checkpointing is False
    assert options.prepare_data and options.local_files_only and options.non_interactive
    assert options.monitor == "loss" and options.early_stopping_patience == 3
    assert options.early_stopping_min_delta == 0.001


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--epochs", "0"),
        ("--learning-rate", "nan"),
        ("--batch-size", "-1"),
        ("--early-stopping-patience", "0"),
        ("--early-stopping-min-delta", "-1"),
        ("--early-stopping-min-delta", "nan"),
    ],
)
def test_invalid_training_arguments_are_parser_errors(flag, value):
    with pytest.raises(SystemExit) as error:
        main(["train", "reference", flag, value])
    assert error.value.code == 2


def test_doctor_cpu_and_cuda_validation(monkeypatch, capsys):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert main(["doctor", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["selected_device"] == "cpu"
    assert main(["doctor", "--device", "cuda"]) == 2
    assert "CUDA was requested" in capsys.readouterr().err


def test_training_missing_data_never_initializes_pretrained_weights(
    tmp_path, monkeypatch, capsys
):
    def forbidden(*args, **kwargs):
        pytest.fail("Missing data must fail before downloading/initializing weights")

    monkeypatch.setattr(
        "emotion_detection.reference_baselines.track_a.load_sequence_classifier",
        forbidden,
    )
    assert (
        main(["train", "reference", "--device", "cpu", "--project-root", str(tmp_path)])
        == 2
    )
    assert "emotion data prepare" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_prepare_data_training_flag_orders_stages(local_reference, monkeypatch):
    from emotion_detection.experiments import reference as experiment

    calls = []
    prepare = experiment.prepare_reference_dataset
    load = experiment.load_prepared_reference_dataset

    def prepare_data(*args, **kwargs):
        calls.append("prepare")
        assert kwargs["interactive"] is False
        return prepare(*args, **kwargs)

    def load_data(*args, **kwargs):
        calls.append("load")
        return load(*args, **kwargs)

    class Model:
        def __init__(self, **kwargs):
            self.model_name = kwargs["model_name"]

        def fit(self, train, dev, **kwargs):
            calls.append("fit")
            assert len(train) == len(dev) == 2
            assert kwargs["data_metadata"]["language"] == "eng"
            self.run_dir = kwargs["output_dir"] / "fake-run"

    monkeypatch.setattr(experiment, "TrackAClassifier", Model)
    monkeypatch.setattr(experiment, "prepare_reference_dataset", prepare_data)
    monkeypatch.setattr(experiment, "load_prepared_reference_dataset", load_data)
    assert (
        main(
            [
                "train",
                "reference",
                "--project-root",
                str(local_reference),
                "--device",
                "cpu",
                "--prepare-data",
                "--non-interactive",
            ]
        )
        == 0
    )
    assert calls == ["prepare", "load", "fit"]


@pytest.mark.parametrize("track", ["b", "c"])
def test_unimplemented_tracks_fail_before_creating_files(tmp_path, track, capsys):
    assert (
        main(["data", "prepare", "--track", track, "--project-root", str(tmp_path)])
        == 2
    )
    assert "not implemented" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_invalid_run_fails_before_explicit_preparation(tmp_path, monkeypatch):
    from emotion_detection.experiments import reference as experiment

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid run IDs must fail before data preparation")

    monkeypatch.setattr(experiment, "prepare_reference_dataset", forbidden)
    assert (
        main(
            [
                "train",
                "reference",
                "--project-root",
                str(tmp_path),
                "--prepare-data",
                "--run-id",
                "../escape",
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert not list(tmp_path.iterdir())


def test_invalid_early_stopping_fails_before_explicit_preparation(
    tmp_path, monkeypatch, capsys
):
    from emotion_detection.experiments import reference as experiment

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid early stopping must fail before preparing data")

    monkeypatch.setattr(experiment, "prepare_reference_dataset", forbidden)
    assert (
        main(
            [
                "train",
                "reference",
                "--project-root",
                str(tmp_path),
                "--prepare-data",
                "--device",
                "cpu",
                "--early-stopping-min-delta",
                "0.1",
            ]
        )
        == 2
    )
    assert "requires early_stopping_patience" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
