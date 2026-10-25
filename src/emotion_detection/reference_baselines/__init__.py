"""Paper-referenced transformer baselines, separate from the main experiments."""

import json
from pathlib import Path

from emotion_detection.reference_baselines.track_a import TrackAClassifier
from emotion_detection.reference_baselines.track_b import TrackBClassifier
from emotion_detection.reference_baselines.track_c import TrackCClassifier

__all__ = [
    "TrackAClassifier",
    "TrackBClassifier",
    "TrackCClassifier",
    "load_reference_model",
]


def load_reference_model(checkpoint_dir, *, device="auto"):
    """Dispatch saved track metadata, including legacy Track A checkpoints."""
    metadata = json.loads((Path(checkpoint_dir) / "metadata.json").read_text())
    classes = {"a": TrackAClassifier, "b": TrackBClassifier, "c": TrackCClassifier}
    if metadata.get("track") not in classes:
        raise ValueError("Unsupported checkpoint track")
    return classes[metadata["track"]].load(checkpoint_dir, device=device)
