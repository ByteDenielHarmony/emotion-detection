"""Backward-compatible Track A facade over the shared transformer engine."""

from emotion_detection.reference_baselines.base import (
    ReferenceTransformer,
    _fingerprint,
)
from emotion_detection.reference_baselines.transformer import load_sequence_classifier

__all__ = ["TrackAClassifier", "_fingerprint"]


class TrackAClassifier(ReferenceTransformer):
    track = "a"

    def _load_checkpoint(self, names):
        # Preserve the existing initialization seam for integrations/tests.
        return load_sequence_classifier(
            self.model_name,
            names,
            revision=self.revision,
            seed=self.seed,
            local_files_only=self.local_files_only,
        )
