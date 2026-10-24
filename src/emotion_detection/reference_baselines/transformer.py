"""Shared transformer initialization and the Track A/C classification objective."""

from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    set_seed,
)

DEFAULT_MODEL_NAME = "FacebookAI/xlm-roberta-large"
# Pin known starting points; other checkpoints resolve and record their Hub SHA.
MODEL_REVISIONS = {
    DEFAULT_MODEL_NAME: "c23d21b0620b635a76227c604d44e43a9f0ee389",
    "FacebookAI/xlm-roberta-base": "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089",
}


def resolve_device(device: str) -> str:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_precision(precision: str, device: str) -> str:
    if precision not in {"auto", "no", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be 'auto', 'no', 'fp16', or 'bf16'")
    if precision == "auto":
        if device == "cpu":
            return "no"
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if precision != "no" and device != "cuda":
        raise ValueError("This baseline enables mixed precision on CUDA only")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("The CUDA device does not support BF16")
    return precision


def load_sequence_classifier(
    model_name,
    label_names,
    *,
    revision=None,
    seed=42,
    local_files_only=False,
    problem_type="multi_label_classification",
):
    """Initialize a model with explicit output metadata, independent of size.

    Future intensity heads can supply expanded output names and problem_type=None
    while providing their own objective, without changing checkpoint loading.
    """
    set_seed(seed)
    is_local = Path(model_name).is_dir()
    revision = None if is_local else revision or MODEL_REVISIONS.get(model_name)
    config = AutoConfig.from_pretrained(
        model_name, revision=revision, local_files_only=local_files_only
    )
    resolved_revision = getattr(config, "_commit_hash", None) or revision
    config.num_labels = len(label_names)
    config.id2label = dict(enumerate(label_names))
    config.label2id = {label: index for index, label in enumerate(label_names)}
    config.problem_type = problem_type
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=resolved_revision,
        use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        revision=resolved_revision,
        local_files_only=local_files_only,
        ignore_mismatched_sizes=True,
    )
    return model, tokenizer, resolved_revision


def positive_class_weights(labels):
    """Match upstream neg/pos weighting; use 1 when a class has no positives."""
    labels = np.asarray(labels)
    positives = labels.sum(axis=0)
    weights = np.divide(
        len(labels) - positives,
        positives,
        out=np.ones_like(positives, dtype=np.float32),
        where=positives != 0,
    )
    return torch.tensor(weights, dtype=torch.float32)


class WeightedMultilabelTrainer(Trainer):
    """Weighted BCE, shared by monolingual A and future cross-lingual C."""

    def __init__(self, *args, pos_weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weights = pos_weights
        # The loss below is a mean, not normalized by num_items_in_batch. Let
        # Trainer apply its usual gradient-accumulation scaling.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = dict(inputs)
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            outputs.logits.float(),
            labels.float(),
            pos_weight=self.pos_weights.to(outputs.logits.device),
        )
        return (loss, outputs) if return_outputs else loss
