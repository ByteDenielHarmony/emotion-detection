"""Track B: four intensity classes per emotion, sharing the transformer engine."""

from emotion_detection.reference_baselines.base import ReferenceTransformer


class TrackBClassifier(ReferenceTransformer):
    track = "b"

    def __init__(self, *args, threshold=None, **kwargs):
        if threshold is not None:
            raise ValueError("Track B uses argmax intensity classes, not a threshold")
        super().__init__(*args, threshold=None, **kwargs)
