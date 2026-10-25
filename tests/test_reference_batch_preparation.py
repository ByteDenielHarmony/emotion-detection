"""Standalone batch preparation uses synthetic inputs, without models or network."""

import hashlib
import subprocess
import sys
from dataclasses import replace

import pandas as pd
import pytest

from emotion_detection.cli import main
from emotion_detection.datasets import (
    iter_prepare_reference_datasets,
    reference,
    reference_crosslingual,
)
from emotion_detection.datasets.acquisition import AcquisitionError
from emotion_detection.datasets.sources import DEFAULT_RELEASE, get_reference_sources
from emotion_detection.datasets.sources_track_c import (
    DEFAULT_C_PROTOCOL,
    get_crosslingual_sources,
)


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


@pytest.fixture
def batch_inputs(tmp_path, monkeypatch):
    def seed(sources, crosslingual=False):
        selected = []
        for source in sources:
            raw = tmp_path / "data/raw/reference_baselines" / source.release
            if crosslingual:
                raw /= "crosslingual"
            raw /= source.relative_path
            raw.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [
                    {
                        "id": f"{source.split}-{index}",
                        "text": f"{source.track} {source.language} {source.split} text {index}",
                        **{
                            name: (index % 4 if source.track == "b" else index % 2)
                            for name in source.label_names
                        },
                    }
                    for index in range(4)
                ]
            ).to_csv(raw, index=False)
            selected.append(
                replace(source, sha256=hashlib.sha256(raw.read_bytes()).hexdigest())
            )
        return tuple(selected)

    labeled = {track: seed(get_reference_sources(track)) for track in ("a", "b")}
    crosslingual = seed(get_crosslingual_sources("eng"), crosslingual=True)
    monkeypatch.setattr(
        reference, "get_reference_sources", lambda track, *args: labeled[track]
    )
    monkeypatch.setattr(
        reference_crosslingual,
        "get_crosslingual_sources",
        lambda *args: crosslingual,
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Batch preparation must not access the network")

    monkeypatch.setattr("emotion_detection.datasets.acquisition.urlopen", forbidden)
    return tmp_path


def test_batch_preflights_then_prepares_in_order(tmp_path, monkeypatch):
    events = []

    def labeled(track, language, release):
        events.append(("preflight", track, language, release))

    def crosslingual(language, release, protocol):
        events.append(("preflight", "c", language, release, protocol))

    def prepare(track, language, **options):
        events.append(("prepare", track, language, options))
        return track

    monkeypatch.setattr(reference, "get_reference_sources", labeled)
    monkeypatch.setattr(
        "emotion_detection.datasets.sources_track_c.get_crosslingual_sources",
        crosslingual,
    )
    monkeypatch.setattr(reference, "prepare_reference_dataset", prepare)
    options = {
        "release": DEFAULT_RELEASE,
        "project_root": tmp_path,
        "interactive": False,
    }
    assert list(
        iter_prepare_reference_datasets("eng", protocol=DEFAULT_C_PROTOCOL, **options)
    ) == ["a", "b", "c"]
    assert events == [
        ("preflight", "a", "eng", DEFAULT_RELEASE),
        ("preflight", "b", "eng", DEFAULT_RELEASE),
        ("preflight", "c", "eng", DEFAULT_RELEASE, DEFAULT_C_PROTOCOL),
        ("prepare", "a", "eng", options),
        ("prepare", "b", "eng", options),
        ("prepare", "c", "eng", {**options, "protocol": DEFAULT_C_PROTOCOL}),
    ]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("protocol", [None, DEFAULT_C_PROTOCOL])
def test_cli_prepares_all_and_outputs_remain_readable(batch_inputs, capsys, protocol):
    root = batch_inputs
    raw_before = snapshot(root / "data/raw")
    arguments = ["--protocol", protocol] if protocol is not None else []
    assert (
        main(
            [
                "data",
                "prepare",
                "--dataset",
                "reference",
                "--track",
                "all",
                "--language",
                "eng",
                "--project-root",
                str(root),
                "--non-interactive",
                *arguments,
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert (
        output.index("Prepared reference Track A")
        < output.index("Prepared reference Track B")
        < output.index("Prepared reference Track C")
    )
    assert "Preparation completed: A, B, C for eng; no training started" in output
    assert "source_dev:" in output and "target_dev:" in output
    for track in ("a", "b", "c"):
        data = reference.load_prepared_reference_dataset(track, project_root=root)
        assert len(data.test) == 4
        if track == "b":
            assert data.train["joy"].tolist() == [0, 1, 2, 3]
        if track == "c":
            assert data.test["disgust"].isna().all()
            assert not data.test["known_disgust"].any()
    assert snapshot(root / "data/raw") == raw_before
    assert not (root / "output/runs").exists()
    assert not (root / "data/processed/train.parquet").exists()


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--language", "ron"], "no Track B train/ron.csv"),
        (["--language", "ind"], "Unsupported Track A language"),
        (["--language", "unknown"], "Unsupported Track A language"),
        (["--release", "unknown"], "Unsupported release"),
        (["--protocol", "unknown"], "Unsupported Track C protocol"),
    ],
)
def test_batch_invalid_selection_fails_before_preparation(
    tmp_path, monkeypatch, capsys, arguments, message
):
    def forbidden(*args, **kwargs):
        pytest.fail("All selections must be preflighted before any preparation")

    monkeypatch.setattr(reference, "prepare_reference_dataset", forbidden)
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "all",
                "--project-root",
                str(tmp_path),
                *arguments,
            ]
        )
        == 2
    )
    assert message in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_batch_checks_c_sources_not_just_target_manifest(tmp_path, monkeypatch):
    # Let A/B validation pass to isolate C's source-readiness check for sun.
    monkeypatch.setattr(reference, "get_reference_sources", lambda *args: ())

    def forbidden(*args, **kwargs):
        pytest.fail("Unavailable C sources must fail before preparing A or B")

    monkeypatch.setattr(reference, "prepare_reference_dataset", forbidden)
    with pytest.raises(ValueError, match="no available other Austronesian"):
        list(iter_prepare_reference_datasets("sun", project_root=tmp_path))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("non_interactive,expected", [(False, None), (True, False)])
def test_batch_cli_forwards_interaction_mode(
    tmp_path, monkeypatch, non_interactive, expected
):
    calls = []

    def prepare(**options):
        calls.append(options)
        return iter(())

    monkeypatch.setattr(
        "emotion_detection.datasets.iter_prepare_reference_datasets", prepare
    )
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "all",
                "--project-root",
                str(tmp_path),
                *(["--non-interactive"] if non_interactive else []),
            ]
        )
        == 0
    )
    assert calls == [
        {
            "language": "eng",
            "release": DEFAULT_RELEASE,
            "project_root": tmp_path.resolve(),
            "interactive": expected,
        }
    ]


@pytest.mark.parametrize("failed_track", ["b", "c"])
@pytest.mark.parametrize(
    "error_type,exit_code",
    [(AcquisitionError, 2), (ValueError, 2), (KeyboardInterrupt, 130)],
)
def test_batch_failure_keeps_completed_outputs(
    batch_inputs, monkeypatch, capsys, failed_track, error_type, exit_code
):
    root = batch_inputs
    original = reference.prepare_reference_dataset
    calls = []
    completed = {}
    before = snapshot(root)

    def prepare(track, *args, **options):
        calls.append(track)
        if track == failed_track:
            raise error_type("synthetic batch failure")
        data = original(track, *args, **options)
        completed.update(snapshot(root))
        return data

    monkeypatch.setattr(reference, "prepare_reference_dataset", prepare)
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "all",
                "--project-root",
                str(root),
                "--non-interactive",
            ]
        )
        == exit_code
    )
    assert calls == (["a", "b"] if failed_track == "b" else ["a", "b", "c"])
    after = snapshot(root)
    assert all(after[path] == content for path, content in before.items())
    assert after == completed
    for track in calls[:-1]:
        reference.load_prepared_reference_dataset(track, project_root=root)
    output = capsys.readouterr()
    assert "Prepared reference Track A" in output.out
    assert "Preparation completed" not in output.out
    assert (
        "Interrupted" in output.err
        if exit_code == 130
        else "synthetic batch failure" in output.err
    )


@pytest.mark.parametrize("track", [None, "a", "b", "c"])
def test_single_track_and_default_still_prepare_only_one(batch_inputs, capsys, track):
    assert (
        main(
            [
                "data",
                "prepare",
                "--project-root",
                str(batch_inputs),
                "--non-interactive",
                *(["--track", track] if track is not None else []),
            ]
        )
        == 0
    )
    selected = track or "a"
    assert {
        p.name for p in (batch_inputs / "data/processed/reference_baselines").iterdir()
    } == {f"track_{selected}"}
    assert "Preparation completed: A, B, C" not in capsys.readouterr().out


@pytest.mark.parametrize(
    "dataset,track",
    [("reference", "a"), ("reference", "b"), ("joint", None), ("joint", "all")],
)
def test_protocol_stays_invalid_outside_c_or_reference_batch(
    tmp_path, capsys, dataset, track
):
    assert (
        main(
            [
                "data",
                "prepare",
                "--dataset",
                dataset,
                "--protocol",
                DEFAULT_C_PROTOCOL,
                "--project-root",
                str(tmp_path),
                *(["--track", track] if track else []),
            ]
        )
        == 2
    )
    assert "--protocol" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_dataset_all_is_not_added():
    with pytest.raises(SystemExit) as error:
        main(["data", "prepare", "--dataset", "all"])
    assert error.value.code == 2


def test_real_batch_preparation_does_not_import_models(batch_inputs):
    script = """
import hashlib
import sys
from dataclasses import replace
from pathlib import Path
from emotion_detection.cli import main
from emotion_detection.datasets import reference, reference_crosslingual, acquisition

root = Path(sys.argv[1])
labeled = reference.get_reference_sources
crosslingual = reference_crosslingual.get_crosslingual_sources

def select(sources, cross=False):
    raw = root / "data/raw/reference_baselines" / sources[0].release
    if cross:
        raw /= "crosslingual"
    return tuple(replace(s, sha256=hashlib.sha256(
        (raw / s.relative_path).read_bytes()).hexdigest()) for s in sources)

reference.get_reference_sources = lambda *args: select(labeled(*args))
reference_crosslingual.get_crosslingual_sources = lambda *args: select(crosslingual(*args), True)
def forbidden(*args, **kwargs):
    raise AssertionError("No network access allowed")
acquisition.urlopen = forbidden
assert main(["data", "prepare", "--track", "all", "--project-root", str(root),
             "--non-interactive"]) == 0
for name in sys.modules:
    assert name.split(".")[0] not in {"torch", "transformers"}, name
    assert not name.startswith(("emotion_detection.experiments", "emotion_detection.reference_baselines")), name
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(batch_inputs)],
        cwd=batch_inputs,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Preparation completed: A, B, C" in result.stdout
