"""Independent Track B preparation; preserve ordinal intensities, not A labels."""

from emotion_detection.datasets.reference_labeled import (
    PREPARATION_VARIANTS,
    prepare_labeled_track,
)

PREPARATION_VARIANT = PREPARATION_VARIANTS["b"]


def prepare_track_b(frames, *, label_names, language, source_metadata):
    return prepare_labeled_track(
        frames,
        track="b",
        label_names=label_names,
        language=language,
        source_metadata=source_metadata,
    )
