import json
from types import SimpleNamespace

import pytest
from transformers import TrainerControl, TrainerState

from emotion_detection.evaluation.metrics import multilabel_diagnostics
from emotion_detection.evaluation.reporting import format_development_report
from emotion_detection.reference_baselines.training import (
    DevelopmentReportingCallback,
    validate_training_control,
)


def test_binary_confusion_axes_and_micro_f1():
    gold = [[0], [0], [0], [1], [1], [1], [1], [1]]
    predicted = [[0], [0], [1], [0], [0], [0], [1], [1]]
    report = multilabel_diagnostics(gold, predicted, ["joy"])
    assert report["confusion_matrices"]["joy"] == [[2, 1], [3, 2]]
    assert report["support"]["joy"] == 5
    assert report["metrics"]["macro_f1"] == 0.5
    assert report["metrics"]["micro_f1"] == 0.5
    record = {
        **report,
        "metrics": {**report["metrics"], "loss": 0.4},
        "epoch": 1,
        "global_step": 2,
        "monitor": "loss",
        "direction": "minimize",
    }
    rendered = format_development_report(record)
    assert "Macro-F1: 0.5000" in rendered and "Micro-F1: 0.5000" in rendered
    rows = rendered.splitlines()
    assert next(row for row in rows if row.startswith("actual absent")).split()[
        -2:
    ] == ["2", "1"]
    assert next(row for row in rows if row.startswith("actual present")).split()[
        -2:
    ] == ["3", "2"]


def test_absent_emotions_still_have_two_by_two_confusion_matrices():
    report = multilabel_diagnostics(
        [[0, 0], [0, 0]], [[0, 0], [0, 0]], ["disgust", "fear"]
    )
    assert report["confusion_matrices"] == {
        "disgust": [[2, 0], [0, 0]],
        "fear": [[2, 0], [0, 0]],
    }
    assert report["support"] == {"disgust": 0, "fear": 0}
    assert report["metrics"]["micro_f1"] == 0
    assert "surprise" not in report["label_names"]
    json.dumps(report, allow_nan=False)


def test_epoch_callback_prints_and_persists_without_inference(tmp_path, capsys):
    callback = DevelopmentReportingCallback(
        tmp_path, ["joy"], monitor="macro_f1", threshold=0.5
    )
    computed = callback.capture([[1], [0]], [[1], [1]])
    metrics = {f"eval_{name}": value for name, value in computed.items()}
    metrics["eval_loss"] = 0.6
    state = TrainerState(epoch=1.0, global_step=4, is_world_process_zero=True)
    callback.on_evaluate(SimpleNamespace(), state, TrainerControl(), metrics)
    history = [
        json.loads(line)
        for line in (tmp_path / "epoch_history.jsonl").read_text().splitlines()
    ]
    assert len(history) == 1
    assert history[0]["confusion_matrices"]["joy"] == [[0, 1], [0, 1]]
    assert history[0]["split"] == "dev"
    assert "Development metrics - epoch 1" in capsys.readouterr().out
    callback.epoch_reporting = False
    callback.on_evaluate(SimpleNamespace(), state, TrainerControl(), metrics)
    assert len((tmp_path / "epoch_history.jsonl").read_text().splitlines()) == 1
    assert capsys.readouterr().out == ""
    callback.selected_report(epoch=1, global_step=4, metrics=metrics)
    assert "Selected checkpoint" in capsys.readouterr().out
    assert (
        json.loads((tmp_path / "validation_report.json").read_text())["global_step"]
        == 4
    )
    assert len((tmp_path / "epoch_history.jsonl").read_text().splitlines()) == 1


def test_epoch_callback_writes_only_on_main_process(tmp_path):
    callback = DevelopmentReportingCallback(
        tmp_path, ["joy"], monitor="macro_f1", threshold=0.5
    )
    state = TrainerState(epoch=1.0, global_step=1, is_world_process_zero=False)
    callback.on_evaluate(SimpleNamespace(), state, TrainerControl(), {})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "monitor,patience,delta",
    [
        ("accuracy", None, 0),
        ("loss", 0, 0),
        ("loss", -1, 0),
        ("loss", True, 0),
        ("macro_f1", 1.5, 0),
        ("loss", 3, -0.1),
        ("loss", 3, float("nan")),
        ("loss", 3, float("inf")),
        ("macro_f1", None, 0.01),
    ],
)
def test_invalid_training_control_options_fail(monitor, patience, delta):
    with pytest.raises(ValueError):
        validate_training_control(monitor, patience, delta)


def test_training_control_allows_default_and_loss_monitoring():
    validate_training_control("macro_f1", None, 0)
    validate_training_control("macro_f1", 3, 0.001)
    validate_training_control("loss", 3, 0.001)
