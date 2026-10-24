"""Independent Track A preparation with the original binary-label policy."""

from emotion_detection.datasets.reference_labeled import (
    PREPARATION_VARIANTS,
    prepare_labeled_track,
)

PREPARATION_VARIANT = PREPARATION_VARIANTS["a"]


def prepare_track_a(frames, *, label_names, language, source_metadata):
    return prepare_labeled_track(
        frames,
        track="a",
        label_names=label_names,
        language=language,
        source_metadata=source_metadata,
    )
