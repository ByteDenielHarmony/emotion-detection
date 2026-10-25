import numpy as np
import pytest
import torch

from emotion_detection.evaluation.metrics import (
    intensity_diagnostics,
    masked_multilabel_diagnostics,
)
from emotion_detection.reference_baselines.tasks import BinaryEmotionTask, IntensityTask


def test_masked_unknown_cells_have_no_gradient_or_effect_on_metrics():
    task = BinaryEmotionTask(masked=True)
    logits = torch.tensor([[1.0, 2.0], [-1.0, -2.0]], requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    task.loss(logits, targets, torch.ones(2)).backward()
    assert torch.equal(logits.grad[:, 1], torch.zeros(2))
    assert logits.grad[:, 0].abs().sum() > 0
    report = task.report(
        targets.numpy(), np.array([[1, 1], [0, 1]]), ["joy", "disgust"]
    )
    assert report["metrics"]["macro_f1"] == 1
    assert report["metrics"]["f1_disgust"] is None
    assert report["known_counts"]["disgust"] == 0
    assert report["undefined_emotions"] == ["disgust"]


def test_intensity_decoding_loss_and_confusion():
    task = IntensityTask()
    logits = torch.tensor([[0.0, 1.0, 4.0, 2.0, 3.0, 2.0, 1.0, 0.0]])
    assert task.decode(logits.numpy(), 2, None).tolist() == [[2, 0]]
    np.testing.assert_allclose(
        task.activate(logits, 2).reshape(1, 2, 4).sum(-1).numpy(),
        [[1.0, 1.0]],
        atol=1e-6,
    )
    assert torch.isfinite(task.loss(logits, torch.tensor([[2.0, 0.0]]), None))
    report = intensity_diagnostics([[0], [1], [2], [3]], [[0], [1], [2], [3]], ["joy"])
    assert report["metrics"]["pearson"] == pytest.approx(1)
    assert report["metrics"]["mae"] == 0
    assert report["confusion_matrices"]["joy"] == np.eye(4, dtype=int).tolist()


def test_undefined_pearson_is_flagged_not_dropped():
    report = intensity_diagnostics([[0, 0], [1, 1]], [[0, 0], [0, 1]], ["joy", "fear"])
    assert report["undefined_correlations"] == ["joy"]
    assert report["metrics"]["pearson_joy"] == 0
    assert report["metrics"]["pearson"] == pytest.approx(0.5)


def test_unknown_observations_are_not_counted_as_negatives():
    report = masked_multilabel_diagnostics(
        [[1], [0]], [[1], [1]], ["joy"], [[True], [False]]
    )
    assert report["confusion_matrices"]["joy"] == [[0, 0], [0, 1]]
    assert report["metrics"]["macro_f1"] == 1
