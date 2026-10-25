"""Synthetic, network-free tests for read-only reference-suite figures."""

import json
import struct
import subprocess
import sys

import pytest

from emotion_detection.cli import main
from emotion_detection.evaluation.reference_figures import (
    FIGURE_NAMES,
    comparison_table,
    load_reference_suite,
    write_reference_figures,
)


def _metrics(track, labels, score):
    if track == "b":
        metrics = {"pearson": score, "mae": 0.2, "loss": 1.5}
        for name in labels:
            metrics[f"pearson_{name}"] = score - 0.01
            metrics[f"mae_{name}"] = 0.2
            metrics[f"accuracy_{name}"] = 0.5
        return metrics, "pearson"
    metrics = {"macro_f1": score, "micro_f1": score - 0.01, "loss": 1.5}
    for name in labels:
        metrics[f"f1_{name}"] = score - 0.02
        metrics[f"precision_{name}"] = score
        metrics[f"recall_{name}"] = score
    return metrics, "macro_f1"


def _record(track, epoch, labels, score, *, split, stop):
    metrics, monitor = _metrics(track, labels, score)
    return {
        "metrics": metrics,
        "label_names": list(labels),
        "epoch": float(epoch),
        "split": split,
        "monitor": monitor,
        "monitor_value": score,
        "stop_requested": stop,
    }


def build_suite(path, *, language="eng", labels=("surprise", "ira"), early_stop=False):
    path.mkdir(parents=True)
    splits = {"a": "dev", "b": "dev", "c": "source_dev"}
    series = {"a": [0.4, 0.7, 0.55], "b": [0.1, 0.8], "c": [0.2, 0.5, 0.45]}
    test_scores = {"a": 0.5, "b": 0.3, "c": 0.25}
    if early_stop:
        series["a"] = [0.4, 0.7, 0.55, 0.5]
    summary_tracks = {}
    for track, values in series.items():
        history_labels = (*labels, "extra") if track == "c" else labels
        selected_epoch = 2
        selected_score = values[selected_epoch - 1]
        stopped = early_stop and track == "a"
        records = [
            _record(
                track,
                epoch,
                history_labels,
                score,
                split=splits[track],
                stop=stopped and epoch == len(values),
            )
            for epoch, score in enumerate(values, start=1)
        ]
        child = path / track
        child.mkdir()
        (child / "epoch_history.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        selected = next(record for record in records if record["epoch"] == selected_epoch)
        validation = {
            key: value for key, value in selected.items() if key != "stop_requested"
        }
        (child / "validation_report.json").write_text(
            json.dumps(validation), encoding="utf-8"
        )
        test_metrics, monitor = _metrics(track, labels, test_scores[track])
        (child / "test_report.json").write_text(
            json.dumps(
                {
                    "metrics": test_metrics,
                    "label_names": list(labels),
                    "split": "test",
                    "track": track,
                    "epoch": selected_epoch,
                }
            ),
            encoding="utf-8",
        )
        (child / "metadata.json").write_text(
            json.dumps(
                {
                    "track": track,
                    "data": {"provenance": {"language": language}},
                    "training": {
                        "best_metric": selected_score,
                        "best_epoch": selected_epoch,
                        "stopped_early": stopped,
                        "stopping_reason": "early_stopping" if stopped else "epoch_limit",
                    },
                }
            ),
            encoding="utf-8",
        )
        summary_tracks[track] = {
            "status": "completed",
            "development_score": selected_score,
            "test_skipped": False,
            "test_metrics": {monitor: test_scores[track]},
        }
    (path / "summary.json").write_text(
        json.dumps(
            {"language": language, "status": "completed", "tracks": summary_tracks}
        ),
        encoding="utf-8",
    )
    return path


def snapshot(root):
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def png_size(path):
    data = path.read_bytes()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    return struct.unpack(">II", data[16:24])


def test_loader_uses_artifact_names_and_unsmoothed_epochs(tmp_path):
    suite = load_reference_suite(
        build_suite(tmp_path / "suite", labels=("surprise", "ira"))
    )
    track_a = suite.tracks[0]
    assert suite.language == "eng"
    assert [point.value for point in track_a.points] == [0.4, 0.7, 0.55]
    assert track_a.selected_epoch == 2
    assert track_a.selected_score == 0.7
    assert track_a.stopped_early is False
    assert track_a.test_labels == ("surprise", "ira")
    assert suite.tracks[2].development_split == "source_dev"
    assert "extra" not in suite.tracks[2].test_labels
    assert suite.tracks[2].test_labels == ("surprise", "ira")


def test_epoch_limit_can_request_stop_without_early_stopping(tmp_path):
    run = build_suite(tmp_path / "suite")
    history_path = run / "b" / "epoch_history.jsonl"
    lines = history_path.read_text(encoding="utf-8").splitlines()
    last = json.loads(lines[-1])
    last["stop_requested"] = True
    lines[-1] = json.dumps(last)
    history_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    suite = load_reference_suite(run)
    assert suite.tracks[1].stopped_early is False


def test_early_stopping_is_taken_from_the_saved_run(tmp_path):
    suite = load_reference_suite(build_suite(tmp_path / "suite", early_stop=True))
    assert suite.tracks[0].stopped_early is True
    assert [point.epoch for point in suite.tracks[0].points] == [1, 2, 3, 4]
    assert suite.tracks[1].stopped_early is False
    assert suite.tracks[2].stopped_early is False


def test_english_table_states_the_comparison_limits(tmp_path):
    table = comparison_table(load_reference_suite(build_suite(tmp_path / "suite")))
    assert table.baselines_applied is True
    notes = " ".join(table.notes)
    assert "Paper development scores are unavailable." in notes
    assert "not directly comparable" in notes
    assert "modified reference implementation, not an exact reproduction" in notes
    by_track = {row.track: row for row in table.rows}
    assert by_track["A"].baseline == 0.708
    assert by_track["B"].baseline == 0.641
    assert by_track["C"].baseline == 0.375
    assert by_track["C"].source_development is True
    assert by_track["A"].difference == pytest.approx(0.5 - 0.708)
    assert by_track["B"].difference == pytest.approx(0.3 - 0.641)
    assert by_track["C"].difference == pytest.approx(0.25 - 0.375)


def test_english_baselines_are_not_applied_to_other_languages(tmp_path):
    table = comparison_table(
        load_reference_suite(build_suite(tmp_path / "suite", language="deu"))
    )
    assert table.baselines_applied is False
    assert all(row.baseline is None and row.difference is None for row in table.rows)
    notes = " ".join(table.notes)
    assert "were not applied" in notes
    assert "deu" in notes
    assert "0.708" not in notes
    assert "0.641" not in notes
    assert "0.375" not in notes


def test_figures_are_high_resolution_and_do_not_modify_the_run(tmp_path):
    run = build_suite(tmp_path / "suite")
    before = snapshot(run)
    output = tmp_path / "figures"
    written = write_reference_figures(run, output)
    assert snapshot(run) == before
    assert [path.name for path in written] == list(FIGURE_NAMES)
    for path in written:
        width, height = png_size(path)
        assert width >= 2000
        assert height >= 700


def test_existing_figures_require_overwrite(tmp_path):
    run = build_suite(tmp_path / "suite")
    output = tmp_path / "figures"
    write_reference_figures(run, output)
    sentinel = output / "notes.txt"
    sentinel.write_text("keep", encoding="utf-8")
    before = snapshot(output)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_reference_figures(run, output)
    assert snapshot(output) == before
    write_reference_figures(run, output, overwrite=True)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert {path.name for path in output.iterdir() if path.suffix == ".png"} == set(
        FIGURE_NAMES
    )


def test_missing_or_unsupported_results_are_explained(tmp_path):
    run = build_suite(tmp_path / "suite")
    test_report = run / "a" / "test_report.json"
    test_report.unlink()
    with pytest.raises(ValueError, match="test_report.json"):
        load_reference_suite(run)

    run = build_suite(tmp_path / "test-curve")
    history = run / "b" / "epoch_history.jsonl"
    record = json.loads(history.read_text(encoding="utf-8").splitlines()[0])
    record["split"] = "test"
    history.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="test performance"):
        load_reference_suite(run)

    run = build_suite(tmp_path / "missing-epoch")
    metadata_path = run / "c" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["training"]["best_epoch"] = 9
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="selected epoch"):
        load_reference_suite(run)

    single = tmp_path / "single"
    single.mkdir()
    (single / "epoch_history.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="single-track"):
        load_reference_suite(single)

    run = build_suite(tmp_path / "failed")
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["tracks"]["b"] = {"status": "failed", "error": "disk full"}
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError, match="disk full"):
        load_reference_suite(run)


def test_figures_are_not_written_into_the_run(tmp_path):
    run = build_suite(tmp_path / "suite")
    before = snapshot(run)
    with pytest.raises(ValueError, match="inside the run directory"):
        write_reference_figures(run, run / "reports")
    assert snapshot(run) == before


def test_cli_writes_figures_without_loading_a_model(tmp_path, capsys):
    run = build_suite(tmp_path / "reference-suite-synthetic")
    output = tmp_path / "reports" / "figures" / "custom"
    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "report",
                "reference",
                "--run-dir",
                str(run),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    message = capsys.readouterr().out
    assert "no training or evaluation" in message
    for name in FIGURE_NAMES:
        assert (output / name).is_file()
    assert (
        main(
            [
                "report",
                "reference",
                "--run-dir",
                str(run),
                "--output-dir",
                str(output),
            ]
        )
        == 2
    )
    assert "overwrite" in capsys.readouterr().err


def test_report_process_does_not_import_torch(tmp_path):
    run = build_suite(tmp_path / "suite")
    output = tmp_path / "out"
    script = (
        "import sys\n"
        "from emotion_detection.cli import main\n"
        "assert main(sys.argv[1:]) == 0\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'transformers' not in sys.modules\n"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "report",
            "reference",
            "--run-dir",
            str(run),
            "--output-dir",
            str(output),
            "--project-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "development_progress.png").is_file()
