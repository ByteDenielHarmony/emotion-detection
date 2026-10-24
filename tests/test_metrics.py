import pytest

from emotion_detection.evaluation.metrics import multilabel_metrics


def test_macro_f1_averages_emotions_and_handles_missing_positives():
    metrics = multilabel_metrics([[1, 0], [0, 0]], [[1, 0], [1, 0]], ["joy", "anger"])
    assert metrics["precision_joy"] == 0.5
    assert metrics["recall_joy"] == 1
    assert metrics["f1_joy"] == pytest.approx(2 / 3)
    assert metrics["f1_anger"] == 0
    assert metrics["macro_f1"] == pytest.approx(1 / 3)


def test_one_emotion_uses_positive_class_f1_not_negative_class_f1():
    assert multilabel_metrics([[0], [1]], [[0], [0]], ["joy"])["macro_f1"] == 0


@pytest.mark.parametrize(
    "gold, predictions, names",
    [
        ([], [], ["joy"]),
        ([[1]], [[1], [0]], ["joy"]),
        ([[1]], [[1]], ["joy", "anger"]),
        ([[2]], [[1]], ["joy"]),
        ([[1]], [[0.5]], ["joy"]),
        ([[1, 0]], [[1, 0]], ["joy", "joy"]),
    ],
)
def test_invalid_metric_inputs_fail(gold, predictions, names):
    with pytest.raises(ValueError):
        multilabel_metrics(gold, predictions, names)
