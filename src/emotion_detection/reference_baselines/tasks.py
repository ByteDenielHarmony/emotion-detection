"""Task strategies: in-memory targets, head layout, loss, decoding, and reports."""

import numpy as np
import torch
from pandas.api.types import is_bool_dtype

from emotion_detection.evaluation.metrics import (
    intensity_diagnostics,
    masked_multilabel_diagnostics,
    multilabel_diagnostics,
)
from emotion_detection.reference_baselines.data import validate_frame


class BinaryEmotionTask:
    primary_metric = "macro_f1"

    def __init__(self, *, masked=False):
        self.masked = masked

    def output_names(self, names):
        return tuple(names)

    def validate(self, frame, names, *, require_labels=True, allow_empty=False):
        view = frame
        if self.masked and "source_language" in frame:
            if (
                not frame["id"]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
                or not frame["source_language"]
                .map(lambda value: isinstance(value, str) and bool(value.strip()))
                .all()
            ):
                raise ValueError("Track C requires source_language and original IDs")
            view = frame.assign(
                id=frame["source_language"].astype(str) + ":" + frame["id"].astype(str)
            )
        validate_frame(
            view,
            names,
            require_labels=require_labels and not self.masked,
            allow_empty=allow_empty,
        )
        if self.masked and require_labels:
            for name in names:
                if name not in frame or f"known_{name}" not in frame:
                    raise ValueError(f"Track C requires {name} and known_{name}")
                mask = frame[f"known_{name}"]
                if (
                    mask.isna().any()
                    or not is_bool_dtype(mask.dtype)
                    or not mask.eq(frame[name].notna()).all()
                ):
                    raise ValueError(
                        "Known-label masks must agree with nullable targets"
                    )
                if not frame.loc[mask, name].isin([0, 1]).all():
                    raise ValueError("Observed Track C annotations must be binary")
            if not frame[[f"known_{name}" for name in names]].any().any():
                raise ValueError("No observed annotations in the selected label space")

    def targets(self, frame, names):
        values = frame[list(names)].to_numpy(
            dtype=np.float32, na_value=np.nan, copy=True
        )
        if not self.masked:
            return values
        mask = frame[[f"known_{name}" for name in names]].to_numpy(dtype=bool)
        # Only tensor placeholders are zero; raw nullable annotations stay intact.
        return np.concatenate(
            [np.where(mask, values, 0), mask.astype(np.float32)], axis=1
        )

    def weights(self, targets, count):
        values = targets[:, :count]
        mask = targets[:, count:] if self.masked else np.ones_like(values)
        positives = (values * mask).sum(axis=0)
        negatives = ((1 - values) * mask).sum(axis=0)
        return torch.tensor(
            np.divide(
                negatives,
                positives,
                out=np.ones(count, dtype=np.float32),
                where=positives != 0,
            )
        )

    def loss(self, logits, targets, weights):
        count = logits.shape[-1]
        losses = torch.nn.functional.binary_cross_entropy_with_logits(
            logits.float(),
            targets[..., :count].float(),
            pos_weight=weights.to(logits.device),
            reduction="none",
        )
        if not self.masked:
            return losses.mean()
        mask = targets[..., count:].float()
        if mask.sum() == 0:
            # Keep an all-unknown evaluation batch neutral; training rows are
            # validated to have supervision before constructing a dataset.
            return losses.sum() * 0
        return (losses * mask).sum() / mask.sum()

    def decode(self, logits, count, threshold):
        return (torch.as_tensor(logits).sigmoid().numpy() >= threshold).astype(int)

    def activate(self, logits, count):
        return logits.sigmoid()

    def report(self, targets, predictions, names):
        if self.masked:
            count = len(names)
            return masked_multilabel_diagnostics(
                targets[:, :count], predictions, names, targets[:, count:].astype(bool)
            )
        return multilabel_diagnostics(targets, predictions, names)


class IntensityTask:
    primary_metric = "pearson"
    masked = False

    def output_names(self, names):
        return tuple(f"{name}_{level}" for name in names for level in range(4))

    def validate(self, frame, names, *, require_labels=True, allow_empty=False):
        validate_frame(frame, names, require_labels=False, allow_empty=allow_empty)
        if require_labels and (
            not set(names).issubset(frame.columns)
            or not frame[list(names)].isin([0, 1, 2, 3]).all().all()
        ):
            raise ValueError("Track B requires integer intensity labels 0, 1, 2, 3")

    def targets(self, frame, names):
        return frame[list(names)].to_numpy(dtype=np.float32, copy=True)

    def weights(self, targets, count):
        return None

    def loss(self, logits, targets, weights):
        return torch.nn.functional.cross_entropy(
            logits.float().reshape(-1, 4), targets.long().reshape(-1)
        )

    def decode(self, logits, count, threshold):
        return np.asarray(logits).reshape(-1, count, 4).argmax(axis=-1)

    def activate(self, logits, count):
        return logits.reshape(-1, count, 4).softmax(dim=-1).reshape(-1, count * 4)

    def report(self, targets, predictions, names):
        return intensity_diagnostics(targets, predictions, names)


def task_for_track(track):
    if track == "b":
        return IntensityTask()
    if track in {"a", "c"}:
        return BinaryEmotionTask(masked=track == "c")
    raise ValueError(f"Unsupported reference track: {track}")
