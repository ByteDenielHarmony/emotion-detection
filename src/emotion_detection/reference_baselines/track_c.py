"""Track C: source-only masked binary classification with native target outputs."""

from emotion_detection.reference_baselines.base import ReferenceTransformer
from emotion_detection.reference_baselines.data import EMOTIONS


class TrackCClassifier(ReferenceTransformer):
    track = "c"

    def __init__(self, *args, model_label_names=EMOTIONS, **kwargs):
        super().__init__(*args, model_label_names=model_label_names, **kwargs)
