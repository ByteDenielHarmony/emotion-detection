"""Offline Track A checks: synthetic frames and tiny local transformer weights."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import (
    PreTrainedTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    XLMRobertaConfig,
    XLMRobertaForSequenceClassification,
)

from emotion_detection.reference_baselines import TrackAClassifier
from emotion_detection.reference_baselines.data import (
    EmotionDataCollator,
    TokenizedEmotionDataset,
    infer_label_names,
    validate_frame,
)
from emotion_detection.reference_baselines.transformer import (
    DEFAULT_MODEL_NAME,
    MODEL_REVISIONS,
    load_sequence_classifier,
    positive_class_weights,
    resolve_device,
    resolve_precision,
)

ENGLISH = ("anger", "fear", "joy", "sadness", "surprise")


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")

    def no_network(*args, **kwargs):
        pytest.fail("Reference-baseline unit tests must not access the network")

    monkeypatch.setattr("requests.sessions.Session.request", no_network)


@pytest.fixture
def frames():
    train = pd.DataFrame(
        {
            "id": ["a-1", "a-2", "a-3", "a-4"],
            "text": ["happy news", "scared story", "sad day", "angry surprise"],
            "anger": [0, 0, 0, 1],
            "fear": [0, 1, 0, 0],
            "joy": [1, 0, 0, 0],
            "sadness": [0, 0, 1, 0],
            "surprise": [1, 0, 0, 1],
            "length": [2, 2, 2, 2],  # Not a target.
        },
        index=[10, 20, 30, 40],
    )
    validation = pd.DataFrame(
        {
            "text": ["a bright day", "another frightening tale"],
            "surprise": [0, 1],
            "joy": [1, 0],
            "id": ["a-dev-2", "a-dev-1"],
            "fear": [0, 1],
            "anger": [0, 0],
            "sadness": [0, 0],
        },
        index=[70, 50],
    )
    return train, validation


@pytest.fixture
def checkpoint_factory(tmp_path):
    def create(family="roberta", num_labels=2):
        path = tmp_path / f"pretrained-{family}-{num_labels}"
        vocabulary = {
            token: index
            for index, token in enumerate(
                ["<s>", "<pad>", "</s>", "<unk>", "<mask>", "happy", "sad", "day"]
            )
        }
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)]
        )
        fast = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<s>",
            eos_token="</s>",
            pad_token="<pad>",
            unk_token="<unk>",
            mask_token="<mask>",
            model_max_length=32,
            model_input_names=["input_ids", "attention_mask"],
        )
        fast.save_pretrained(path)
        config_class, model_class = {
            "roberta": (RobertaConfig, RobertaForSequenceClassification),
            "xlm-roberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification),
        }[family]
        config = config_class(
            vocab_size=len(vocabulary),
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=16,
            max_position_embeddings=64,
            num_labels=num_labels,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            classifier_dropout=0.0,
            pad_token_id=1,
            bos_token_id=0,
            eos_token_id=2,
        )
        model_class(config).save_pretrained(path)
        return path

    return create


def test_label_inference_uses_names_and_supports_six_emotions(frames):
    train, _ = frames
    assert infer_label_names(train) == ENGLISH
    six = train.assign(disgust=0)
    assert infer_label_names(six) == (
        "anger",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
    )
    validate_frame(six, infer_label_names(six))


def test_missing_annotations_are_not_zero_filled(frames):
    train, _ = frames
    null_disgust = train.assign(disgust=np.nan)
    with pytest.raises(ValueError, match="binary"):
        validate_frame(null_disgust, infer_label_names(null_disgust))
    # Explicitly selecting a five-emotion schema is allowed; no sixth label exists.
    validate_frame(null_disgust, ENGLISH)


@pytest.mark.parametrize("family", ["roberta", "xlm-roberta"])
def test_checkpoint_is_replaceable_and_label_head_is_dynamic(
    checkpoint_factory, family
):
    path = checkpoint_factory(family)
    model, tokenizer, revision = load_sequence_classifier(
        str(path), ENGLISH, seed=17, local_files_only=True
    )
    assert model.config.model_type == family
    assert model.config.num_labels == 5
    assert model.config.id2label == dict(enumerate(ENGLISH))
    assert model.config.problem_type == "multi_label_classification"
    assert tokenizer.pad_token_id == 1
    assert revision is None  # This is a local fixture, not a Hub revision.


def test_in_memory_adapter_preserves_text_and_target_order(checkpoint_factory):
    path = checkpoint_factory()
    _, tokenizer, _ = load_sequence_classifier(
        str(path), ENGLISH, local_files_only=True
    )
    labels = np.array([[0, 0, 1, 0, 1]], dtype=float)
    dataset = TokenizedEmotionDataset(
        ["HAPPY day"], tokenizer, labels=labels, max_length=4
    )
    labels[:] = 0  # Caller mutation must not alter the dataset targets.
    assert dataset.texts == ("HAPPY day",)
    assert len(dataset[0]["input_ids"]) <= 4
    assert dataset[0]["labels"].tolist() == [0, 0, 1, 0, 1]
    assert dataset[0]["labels"].dtype == torch.float32
    assert "labels" not in TokenizedEmotionDataset(["day"], tokenizer)[0]
    batch = EmotionDataCollator(tokenizer)([dataset[0], dataset[0]])
    assert batch["labels"].shape == (2, 5)
    assert batch["labels"].dtype == torch.float32


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda df: df.drop(columns="text"), "Missing required"),
        (lambda df: df.assign(anger=2), "binary"),
        (lambda df: df.assign(fear=np.nan), "binary"),
        (lambda df: df.assign(text=" "), "nonempty strings"),
        (lambda df: df.assign(id="same"), "unique"),
        (lambda df: df.assign(id=None), "non-null"),
        (lambda df: df.iloc[:0], "must not be empty"),
    ],
)
def test_invalid_training_data_fails_before_model_download(
    frames, monkeypatch, change, message
):
    train, validation = frames

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid data must fail before initializing a model")

    monkeypatch.setattr(
        "emotion_detection.reference_baselines.track_a.load_sequence_classifier",
        forbidden,
    )
    with pytest.raises(ValueError, match=message):
        TrackAClassifier().fit(change(train), validation)


@pytest.mark.parametrize("overlap", ["id", "text"])
def test_split_leakage_is_rejected(frames, overlap):
    train, validation = frames
    validation.loc[validation.index[0], overlap] = train.iloc[0][overlap]
    if overlap == "text":
        validation["text"] = validation["text"].str.upper()
    with pytest.raises(ValueError, match="overlap"):
        TrackAClassifier().fit(train, validation)


@pytest.mark.parametrize("family", ["roberta", "xlm-roberta"])
def test_train_save_load_and_predict_roundtrip(
    tmp_path, checkpoint_factory, frames, family
):
    train, validation = frames
    if family == "xlm-roberta":
        train = train.assign(disgust=[0, 1, 0, 0])
        validation = validation.assign(disgust=[1, 0])
    names = infer_label_names(train)
    original_train, original_validation = (
        train.copy(deep=True),
        validation.copy(deep=True),
    )
    path = checkpoint_factory(family)
    model_class = {
        "roberta": RobertaForSequenceClassification,
        "xlm-roberta": XLMRobertaForSequenceClassification,
    }[family]
    initial_encoder = (
        model_class.from_pretrained(path)
        .roberta.embeddings.word_embeddings.weight.detach()
        .clone()
    )
    classifier = TrackAClassifier(
        str(path), max_length=16, threshold=0.4, device="cpu", local_files_only=True
    )
    result = classifier.fit(
        train,
        validation,
        output_dir=tmp_path / "runs",
        run_id="reference-track-a-tiny",
        epochs=1,
        learning_rate=1e-3,
        batch_size=2,
        gradient_accumulation_steps=2,
        data_metadata={"revision": "synthetic-v1"},
    )
    assert result is classifier
    assert not torch.equal(
        initial_encoder,
        classifier.model.roberta.embeddings.word_embeddings.weight.detach(),
    )
    checkpoint = classifier.run_dir / "checkpoint"
    assert (checkpoint / "model.safetensors").is_file()
    assert (checkpoint / "tokenizer.json").is_file()
    assert (checkpoint / "config.json").is_file()
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["training"]["global_step"] == 1
    assert metadata["training"]["mixed_precision"] == "no"
    assert metadata["training"]["gradient_accumulation_steps"] == 2
    assert metadata["training_arguments"]["load_best_model_at_end"] is True
    assert metadata["training_arguments"]["metric_for_best_model"] == "macro_f1"
    assert metadata["data"]["provenance"] == {"revision": "synthetic-v1"}
    assert len(metadata["data"]["train_sha256"]) == 64
    assert metadata["training"]["best_macro_f1"] is not None
    assert (
        json.loads((classifier.run_dir / "metrics.json").read_text())["eval_macro_f1"]
        >= 0
    )

    unlabeled = validation[["id", "text"]]
    probabilities = classifier.predict_proba(unlabeled, batch_size=1)
    predictions = classifier.predict(unlabeled)
    assert list(predictions.columns) == ["id", *names]
    assert predictions.index.tolist() == validation.index.tolist()
    assert predictions["id"].tolist() == validation["id"].tolist()
    assert probabilities[list(names)].ge(0).all().all()
    assert probabilities[list(names)].le(1).all().all()
    assert predictions[list(names)].isin([0, 1]).all().all()
    assert set(classifier.evaluate(validation)) == {
        "macro_f1",
        *[
            f"{metric}_{label}"
            for metric in ("precision", "recall", "f1")
            for label in names
        ],
    }
    restored = TrackAClassifier.load(checkpoint, device="cpu")
    pd.testing.assert_frame_equal(
        probabilities, restored.predict_proba(unlabeled, batch_size=1)
    )
    pd.testing.assert_frame_equal(predictions, restored.predict(unlabeled))
    assert restored.threshold == 0.4
    assert restored.label_names == names
    pd.testing.assert_frame_equal(train, original_train)
    pd.testing.assert_frame_equal(validation, original_validation)
    assert list(restored.predict(unlabeled.iloc[:0]).columns) == ["id", *names]
    with pytest.raises(FileExistsError):
        restored.save(checkpoint)
    restored.threshold = 0.7
    restored.save(tmp_path / "adjusted-threshold")
    assert (
        TrackAClassifier.load(tmp_path / "adjusted-threshold", device="cpu").threshold
        == 0.7
    )


def test_existing_run_is_not_overwritten(tmp_path, frames):
    train, validation = frames
    run = tmp_path / "existing"
    run.mkdir()
    sentinel = run / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        TrackAClassifier().fit(
            train, validation, output_dir=tmp_path, run_id="existing"
        )
    assert sentinel.read_text() == "keep"


@pytest.mark.parametrize("run_id", ["../escape", "/absolute", "a/b", "a\\b"])
def test_run_id_is_not_a_path(frames, run_id):
    with pytest.raises(ValueError, match="run_id"):
        TrackAClassifier().fit(*frames, run_id=run_id)


def test_predict_requires_trained_weights(frames):
    with pytest.raises(RuntimeError, match="Fit"):
        TrackAClassifier().predict(frames[0])


def test_class_weights_match_upstream():
    weights = positive_class_weights([[1, 0, 1], [0, 0, 1], [0, 0, 1]])
    assert weights.tolist() == [2.0, 1.0, 0.0]


def test_default_checkpoint_has_its_own_revision():
    assert DEFAULT_MODEL_NAME == "FacebookAI/xlm-roberta-large"
    assert (
        MODEL_REVISIONS[DEFAULT_MODEL_NAME]
        != MODEL_REVISIONS["FacebookAI/xlm-roberta-base"]
    )


@pytest.mark.parametrize(
    "model_name, revision, expected",
    [
        (DEFAULT_MODEL_NAME, None, MODEL_REVISIONS[DEFAULT_MODEL_NAME]),
        (
            "FacebookAI/xlm-roberta-base",
            None,
            MODEL_REVISIONS["FacebookAI/xlm-roberta-base"],
        ),
        (DEFAULT_MODEL_NAME, "explicit-revision", "explicit-revision"),
        ("another/roberta-checkpoint", None, None),
    ],
)
def test_checkpoint_revision_selection_is_not_tied_to_the_default_model(
    monkeypatch, model_name, revision, expected
):
    calls = {}

    def load_config(name, **kwargs):
        calls["config"] = (name, kwargs["revision"])
        return SimpleNamespace(_commit_hash=kwargs["revision"] or "resolved-hub-sha")

    def load_tokenizer(name, **kwargs):
        calls["tokenizer"] = (name, kwargs["revision"])
        return object()

    def load_model(name, **kwargs):
        calls["model"] = (name, kwargs["revision"])
        return object()

    module = "emotion_detection.reference_baselines.transformer"
    monkeypatch.setattr(f"{module}.AutoConfig.from_pretrained", load_config)
    monkeypatch.setattr(f"{module}.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr(
        f"{module}.AutoModelForSequenceClassification.from_pretrained", load_model
    )
    _, _, resolved = load_sequence_classifier(model_name, ENGLISH, revision=revision)
    assert calls["config"] == (model_name, expected)
    assert resolved == (expected or "resolved-hub-sha")
    assert calls["tokenizer"] == (model_name, resolved)
    assert calls["model"] == (model_name, resolved)


def test_precision_and_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_precision("auto", "cpu") == "no"
    with pytest.raises(ValueError, match="CUDA"):
        resolve_device("cuda")
    with pytest.raises(ValueError, match="CUDA"):
        resolve_precision("fp16", "cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_precision("auto", "cuda") == "bf16"
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert resolve_precision("auto", "cuda") == "fp16"
    with pytest.raises(ValueError, match="BF16"):
        resolve_precision("bf16", "cuda")
