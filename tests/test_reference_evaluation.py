"""Held-out reference evaluation uses synthetic data and never downloads weights."""

import hashlib
import json
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from emotion_detection.evaluation.reference import evaluate_reference_test


@pytest.fixture
def selected_model(tmp_path):
    frame = pd.DataFrame(
        {
            "id": ["test-b", "test-a"],
            "text": ["One", "Two"],
            "joy": [1, 0],
            "fear": [0, 1],
        },
        index=[20, 10],
    )
    fingerprint = hashlib.sha256(
        frame.to_json(orient="split", index=False, force_ascii=False).encode()
    ).hexdigest()
    metadata = {
        "label_names": ["joy", "fear"],
        "threshold": 0.5,
        "data": {"provenance": {"prepared_fingerprints": {"test": fingerprint}}},
        "training": {
            "monitor": "macro_f1",
            "best_epoch": 2,
            "best_checkpoint": "training/checkpoint-8",
            "mixed_precision": "no",
            "positive_class_weights": [2, 1],
        },
    }
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(json.dumps(metadata))
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "model.safetensors").write_bytes(b"synthetic checkpoint marker")
    (tmp_path / "metrics.json").write_text('{"eval_macro_f1": 0.42}')

    class Classifier:
        run_dir = tmp_path
        label_names = ("joy", "fear")
        threshold = 0.5
        model = torch.nn.Linear(2, 2)
        calls = 0

        def predict_logits(self, data, *, batch_size):
            self.calls += 1
            assert data["id"].tolist() == frame["id"].tolist()
            assert batch_size == 2
            return pd.DataFrame(
                {
                    "id": data["id"],
                    "joy": [2.0, -2.0],
                    "fear": [-1.0, 1.0],
                },
                index=data.index,
            )

    return Classifier(), frame


def test_test_evaluation_saves_separate_metrics_and_preserves_checkpoint(
    selected_model, capsys
):
    model, frame = selected_model
    before = {
        path: path.read_bytes() for path in model.run_dir.rglob("*") if path.is_file()
    }
    weights = {name: value.clone() for name, value in model.model.state_dict().items()}
    report = evaluate_reference_test(model, frame)
    assert model.calls == 1
    assert report["metrics"]["macro_f1"] == report["metrics"]["micro_f1"] == 1
    expected_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([[2.0, -1.0], [-2.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        pos_weight=torch.tensor([2.0, 1.0]),
    ).item()
    assert report["metrics"]["loss"] == expected_loss
    assert report["confusion_matrices"] == {
        "joy": [[1, 0], [0, 1]],
        "fear": [[1, 0], [0, 1]],
    }
    assert report["epoch"] == 2 and report["global_step"] == 8
    assert "Held-out test metrics" in capsys.readouterr().out
    assert all(path.read_bytes() == content for path, content in before.items())
    assert all(
        torch.equal(value, model.model.state_dict()[name])
        for name, value in weights.items()
    )
    saved = pd.read_csv(model.run_dir / "test_predictions.csv")
    assert saved["id"].tolist() == frame["id"].tolist()
    assert saved[["joy", "fear"]].to_numpy().tolist() == [[1, 0], [0, 1]]
    assert json.loads((model.run_dir / "test_report.json").read_text()) == report
    with pytest.raises(FileExistsError, match="refusing"):
        evaluate_reference_test(model, frame)
    assert model.calls == 1


@pytest.mark.parametrize("change", ["data", "threshold", "labels", "checkpoint"])
def test_mismatched_test_inputs_fail_before_inference(selected_model, change):
    model, frame = selected_model
    if change == "data":
        frame = frame.assign(text="changed")
    elif change == "threshold":
        model.threshold = 0.7
    elif change == "labels":
        model.label_names = tuple(reversed(model.label_names))
    else:
        (model.run_dir / "checkpoint/model.safetensors").unlink()
    with pytest.raises(ValueError):
        evaluate_reference_test(model, frame)
    assert model.calls == 0
    assert not list(model.run_dir.glob("test_*"))


def test_gpu_test_uses_recorded_bf16_precision(selected_model, monkeypatch):
    model, frame = selected_model
    metadata = model.run_dir / "checkpoint/metadata.json"
    saved = json.loads(metadata.read_text())
    saved["training"]["mixed_precision"] = "bf16"
    metadata.write_text(json.dumps(saved))
    model.model = SimpleNamespace(
        parameters=lambda: iter([SimpleNamespace(device=torch.device("cuda"))])
    )
    calls = []

    def autocast(device, *, dtype):
        calls.append((device, dtype))
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch, "autocast", autocast)
    report = evaluate_reference_test(model, frame)
    assert calls == [("cuda", torch.bfloat16)]
    assert report["inference_precision"] == report["training_precision"] == "bf16"


def test_nonfinite_logits_do_not_create_metrics(selected_model, monkeypatch):
    model, frame = selected_model
    monkeypatch.setattr(
        model,
        "predict_logits",
        lambda data, **kwargs: pd.DataFrame(
            {
                "id": data["id"],
                "joy": [np.nan, 0],
                "fear": [0, 0],
            }
        ),
    )
    with pytest.raises(ValueError, match="non-finite"):
        evaluate_reference_test(model, frame)
    assert not list(model.run_dir.glob("test_*"))
