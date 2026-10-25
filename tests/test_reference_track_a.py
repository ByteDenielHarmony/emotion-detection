"""Offline Track A checks: synthetic frames and tiny local transformer weights."""

import hashlib
import json
from dataclasses import replace
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

from emotion_detection.reference_baselines import (
    TrackAClassifier,
    TrackBClassifier,
    TrackCClassifier,
    load_reference_model,
)
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
    legacy_path = tmp_path / "legacy-a-checkpoint"
    classifier.save(legacy_path)
    legacy_metadata = json.loads((legacy_path / "metadata.json").read_text())
    legacy_metadata.pop("model_label_names")
    legacy_metadata.pop("output_label_names")
    (legacy_path / "metadata.json").write_text(json.dumps(legacy_metadata))
    legacy = TrackAClassifier.load(legacy_path, device="cpu")
    pd.testing.assert_frame_equal(predictions, legacy.predict(unlabeled))
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


@pytest.mark.parametrize("skip_test", [False, True])
def test_cli_training_roundtrip_uses_prepared_data_only(
    tmp_path, checkpoint_factory, frames, monkeypatch, capsys, skip_test
):
    from emotion_detection.cli import main
    from emotion_detection.datasets import reference
    from emotion_detection.datasets.sources import get_reference_sources
    from emotion_detection.experiments import reference as experiment

    train, dev = frames
    test = dev.copy(deep=True)
    test["id"] = ["test-1", "test-2"]
    test["text"] = ["untouched held-out one", "untouched held-out two"]
    splits = {"train": train, "dev": dev, "test": test}
    sources = []
    for source in get_reference_sources():
        path = (
            tmp_path
            / "data/raw/reference_baselines"
            / source.release
            / source.relative_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        splits[source.split].to_csv(path, index=False)
        sources.append(
            replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    monkeypatch.setattr(
        reference, "get_reference_sources", lambda *a, **kw: tuple(sources)
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Training must neither access the network nor regenerate data")

    monkeypatch.setattr("emotion_detection.datasets.acquisition.urlopen", forbidden)
    assert (
        main(
            [
                "data",
                "prepare",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 0
    )
    prepared = tmp_path / "data/processed/reference_baselines/track_a/eng"
    before = {path: path.read_bytes() for path in prepared.iterdir()}
    monkeypatch.setattr(experiment, "prepare_reference_dataset", forbidden)
    pretrained = checkpoint_factory()
    assert (
        main(
            [
                "train",
                "reference",
                "--track",
                "a",
                "--project-root",
                str(tmp_path),
                "--model-name",
                str(pretrained),
                "--local-files-only",
                "--device",
                "cpu",
                "--max-length",
                "16",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--run-id",
                "cli-tiny",
                *(["--skip-test"] if skip_test else []),
            ]
        )
        == 0
    )
    run = tmp_path / "output/runs/cli-tiny"
    assert (run / "checkpoint/model.safetensors").is_file()
    restored = TrackAClassifier.load(run / "checkpoint", device="cpu")
    assert restored.metadata["data"]["train_rows"] == len(train)
    assert restored.metadata["data"]["validation_rows"] == len(dev)
    assert restored.metadata["data"]["provenance"]["language"] == "eng"
    assert all(path.read_bytes() == content for path, content in before.items())
    metrics = json.loads((run / "metrics.json").read_text())
    assert "eval_macro_f1" in metrics
    assert not any(key.startswith("test_") for key in metrics)
    output = capsys.readouterr().out
    if skip_test:
        assert "Final test evaluation skipped (--skip-test)" in output
        assert not list(run.glob("test_*"))
    else:
        assert "Held-out test metrics - selected checkpoint" in output
        assert "Final test metrics:" in output
        report = json.loads((run / "test_report.json").read_text())
        assert report["split"] == "test" and report["selection_split"] == "dev"
        assert report["epoch"] == restored.metadata["training"]["best_epoch"]
        assert report["inference_precision"] == "no"
        assert report["rows"] == len(test)
        assert (
            report["data_fingerprint"]
            == restored.metadata["data"]["provenance"]["prepared_fingerprints"]["test"]
        )
        predictions = pd.read_csv(run / "test_predictions.csv")
        assert predictions["id"].tolist() == test["id"].tolist()
        assert list(predictions.columns) == ["id", *restored.label_names]
        assert json.loads((run / "test_metrics.json").read_text()) == report["metrics"]


@pytest.mark.parametrize(
    "losses,min_delta,expected_epoch,best_epoch,best_loss",
    [
        ([0.8, 0.5, 0.6, 0.7, 0.5], 0.001, 4, 2, 0.5),
        ([0.8, 0.79, 0.78, 0.78], 0.1, 3, 3, 0.78),
    ],
)
def test_early_stopping_retains_best_and_latest_and_all_epoch_diagnostics(
    tmp_path,
    checkpoint_factory,
    frames,
    monkeypatch,
    capsys,
    losses,
    min_delta,
    expected_epoch,
    best_epoch,
    best_loss,
):
    from emotion_detection.reference_baselines.transformer import (
        WeightedMultilabelTrainer,
    )

    # Script only validation loss, not gradients, to exercise deterministic
    # checkpoint selection/early stopping while still doing real tiny-model training.
    original_loop = WeightedMultilabelTrainer.evaluation_loop
    scripted_losses = iter(losses)

    def evaluation_loop(self, *args, **kwargs):
        result = original_loop(self, *args, **kwargs)
        result.metrics["eval_loss"] = next(scripted_losses)
        return result

    monkeypatch.setattr(WeightedMultilabelTrainer, "evaluation_loop", evaluation_loop)
    other = tmp_path / "runs/another-run/training/checkpoint-999/sentinel.txt"
    other.parent.mkdir(parents=True)
    other.write_text("preserve unrelated run")
    model = TrackAClassifier(
        str(checkpoint_factory()), max_length=16, device="cpu", local_files_only=True
    )
    model.fit(
        *frames,
        output_dir=tmp_path / "runs",
        run_id="early-stop",
        epochs=6,
        batch_size=2,
        gradient_accumulation_steps=2,
        monitor="loss",
        early_stopping_patience=2,
        early_stopping_min_delta=min_delta,
    )
    metadata = model.metadata["training"]
    assert metadata["monitor"] == "loss"
    assert metadata["best_metric"] == best_loss
    assert metadata["monitor_direction"] == "minimize"
    assert metadata["stopped_early"] is True
    assert metadata["stopping_reason"] == "early_stopping"
    assert metadata["epochs_completed"] == expected_epoch
    assert metadata["global_step"] == expected_epoch
    assert metadata["best_epoch"] == best_epoch
    assert model.metadata["training_arguments"]["greater_is_better"] is False
    assert model.metadata["training_arguments"]["save_total_limit"] == 2

    history = [
        json.loads(line)
        for line in (model.run_dir / "epoch_history.jsonl").read_text().splitlines()
    ]
    assert [record["epoch"] for record in history] == list(range(1, expected_epoch + 1))
    assert (
        len(history) == expected_epoch
    )  # Final best-model evaluation is not an extra epoch.
    assert history[-1]["stop_requested"] is True
    assert all(set(record["confusion_matrices"]) == set(ENGLISH) for record in history)
    assert all(sum(record["support"].values()) > 0 for record in history)
    assert metadata["best_macro_f1"] == max(
        record["metrics"]["macro_f1"] for record in history
    )
    selected = json.loads((model.run_dir / "validation_report.json").read_text())
    assert selected["epoch"] == best_epoch
    assert selected["metrics"]["loss"] == best_loss
    assert (
        selected["confusion_matrices"] == history[best_epoch - 1]["confusion_matrices"]
    )
    metrics = json.loads((model.run_dir / "metrics.json").read_text())
    assert metadata["selected_checkpoint_macro_f1"] == metrics["eval_macro_f1"]
    assert metadata["selected_checkpoint_loss"] == best_loss
    checkpoints = list((model.run_dir / "training").glob("checkpoint-*"))
    assert len(checkpoints) == 2
    assert (model.run_dir / f"training/checkpoint-{best_epoch}") in checkpoints
    assert (model.run_dir / f"training/checkpoint-{expected_epoch}") in checkpoints
    assert other.read_text() == "preserve unrelated run"
    restored = TrackAClassifier.load(model.run_dir / "checkpoint", device="cpu")
    for name, value in model.model.state_dict().items():
        assert torch.equal(value.cpu(), restored.model.state_dict()[name].cpu())
    output = capsys.readouterr().out
    for epoch in range(1, expected_epoch + 1):
        assert f"Development metrics - epoch {epoch}" in output
    assert "Selected checkpoint development metrics" in output


def test_invalid_early_stopping_fails_before_initialization(frames, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid training control must fail before initializing weights")

    monkeypatch.setattr(
        "emotion_detection.reference_baselines.track_a.load_sequence_classifier",
        forbidden,
    )
    with pytest.raises(ValueError, match="requires"):
        TrackAClassifier().fit(*frames, early_stopping_min_delta=0.1)


@pytest.fixture
def track_datasets(frames):
    from emotion_detection.datasets.reference_types import (
        CrosslingualReferenceDataset,
        ReferenceDataset,
        prepared_fingerprint,
    )
    from emotion_detection.reference_baselines.data import EMOTIONS

    train, dev = (frame.copy(deep=True) for frame in frames)
    test = dev.assign(id=["test-1", "test-2"], text=["held-out one", "held-out two"])
    a = ReferenceDataset(train, dev, test, ENGLISH, {})
    b_frames = []
    for frame in (train, dev, test):
        frame = frame.copy(deep=True)
        for i, name in enumerate(ENGLISH):
            frame[name] = (np.arange(len(frame)) + i) % 4
        b_frames.append(frame)
    b = ReferenceDataset(*b_frames, ENGLISH, {})

    def masked(frame, language, split, source_track):
        frame = frame.copy(deep=True)
        for name in EMOTIONS:
            if name not in frame:
                frame[name] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
            frame[name] = frame[name].astype("Int64")
            frame[f"known_{name}"] = frame[name].notna()
        frame["source_language"], frame["source_split"], frame["source_track"] = (
            language,
            split,
            source_track,
        )
        return frame

    c_train = masked(train, "deu", "train", "a")
    c_train["source_language"] = ["deu", "afr", "deu", "afr"]
    c_train["id"] = ["shared", "shared", "third", "fourth"]  # Composite IDs.
    c_dev = masked(dev, "deu", "dev", "a")
    c_target = masked(
        test.assign(
            id=["target-dev-1", "target-dev-2"],
            text=["target dev one", "target dev two"],
        ),
        "eng",
        "dev",
        "c",
    )
    c_test = masked(test, "eng", "test", "c")
    c = CrosslingualReferenceDataset(
        c_train, c_dev, c_target, c_test, ENGLISH, EMOTIONS, {}
    )
    for track, data in (("a", a), ("b", b), ("c", c)):
        roles = (
            ("train", "source_dev", "target_dev", "test")
            if track == "c"
            else ("train", "dev", "test")
        )
        data.metadata.update(
            track=track,
            language="eng",
            label_names=list(ENGLISH),
            release="synthetic",
            prepared_fingerprints={
                role: prepared_fingerprint(getattr(data, role)) for role in roles
            },
        )
        if track == "c":
            data.metadata.update(
                model_label_names=list(EMOTIONS),
                source_languages=["deu", "afr"],
                protocol="synthetic",
            )
    return {"a": a, "b": b, "c": c}


@pytest.mark.parametrize(
    "track,model_class,metric",
    [("b", TrackBClassifier, "pearson"), ("c", TrackCClassifier, "macro_f1")],
)
def test_b_c_train_predict_save_load_and_test(
    track,
    model_class,
    metric,
    track_datasets,
    checkpoint_factory,
    tmp_path,
):
    from emotion_detection.evaluation.reference import evaluate_reference_test

    data = track_datasets[track]
    original = data.train.copy(deep=True)
    model = model_class(
        str(checkpoint_factory()),
        label_names=data.label_names,
        max_length=16,
        device="cpu",
        local_files_only=True,
    )
    validation = data.source_dev if track == "c" else data.dev
    model.fit(
        data.train,
        validation,
        output_dir=tmp_path / "runs",
        run_id=track,
        epochs=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        data_metadata=data.metadata,
    )
    assert model.metadata["training"]["monitor"] == metric
    assert model.model.config.num_labels == (20 if track == "b" else 6)
    assert model.metadata["data"]["validation_role"] == (
        "source_dev" if track == "c" else "dev"
    )
    assert model.metadata["data"]["validation_rows"] == len(validation)
    restored = load_reference_model(model.run_dir / "checkpoint", device="cpu")
    assert isinstance(restored, model_class)
    pd.testing.assert_frame_equal(model.predict(data.test), restored.predict(data.test))
    predictions = restored.predict(data.test)
    assert predictions.columns.tolist() == ["id", *ENGLISH]
    assert (
        predictions[list(ENGLISH)]
        .isin(range(4) if track == "b" else range(2))
        .all()
        .all()
    )
    report = evaluate_reference_test(restored, data.test)
    assert metric in report["metrics"]
    assert report["selection_split"] == ("source_dev" if track == "c" else "dev")
    assert len(report["confusion_matrices"]["joy"]) == (4 if track == "b" else 2)
    if track == "c":
        selected = json.loads((model.run_dir / "validation_report.json").read_text())
        assert selected["split"] == "source_dev"
        assert selected["metrics"]["f1_disgust"] is None
        assert "disgust" not in report["label_names"]
        assert restored.model_label_names == data.model_label_names
    pd.testing.assert_frame_equal(data.train, original)


def test_c_rejects_target_dev_for_selection(track_datasets, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid C selection must fail before loading a checkpoint")

    monkeypatch.setattr(
        "emotion_detection.reference_baselines.base.load_sequence_classifier", forbidden
    )
    data = track_datasets["c"]
    with pytest.raises(ValueError, match="source_dev"):
        TrackCClassifier(label_names=ENGLISH).fit(
            data.train, data.target_dev, data_metadata=data.metadata
        )


def test_default_suite_trains_fresh_a_b_c_sequentially(
    track_datasets,
    checkpoint_factory,
    tmp_path,
    monkeypatch,
):
    from emotion_detection.cli import main
    from emotion_detection.experiments import reference as experiment
    from emotion_detection.reference_baselines import base, track_a

    pretrained = str(checkpoint_factory())
    events, loaded_models = [], []
    original = base.load_sequence_classifier

    def load_model(name, *args, **kwargs):
        loaded_models.append(name)
        return original(name, *args, **kwargs)

    def load_data(track, *args, **kwargs):
        events.append(track)
        return track_datasets[track]

    monkeypatch.setattr(experiment, "load_prepared_reference_dataset", load_data)
    monkeypatch.setattr(base, "load_sequence_classifier", load_model)
    monkeypatch.setattr(track_a, "load_sequence_classifier", load_model)
    assert (
        main(
            [
                "train",
                "reference",
                "--project-root",
                str(tmp_path),
                "--device",
                "cpu",
                "--model-name",
                pretrained,
                "--local-files-only",
                "--epochs",
                "1",
                "--batch-size",
                "2",
                "--gradient-accumulation-steps",
                "2",
                "--max-length",
                "16",
                "--run-id",
                "suite",
            ]
        )
        == 0
    )
    assert events == ["a", "b", "c"]
    assert loaded_models == [pretrained] * 3
    suite = tmp_path / "output/runs/suite"
    summary = json.loads((suite / "summary.json").read_text())
    assert summary["status"] == "completed"
    assert [summary["tracks"][track]["metric"] for track in "abc"] == [
        "macro_f1",
        "pearson",
        "macro_f1",
    ]
    for track in "abc":
        assert summary["tracks"][track]["status"] == "completed"
        assert (suite / track / "checkpoint/model.safetensors").is_file()
        assert (suite / track / "test_report.json").is_file()


def test_suite_preflights_all_data_before_any_training(
    track_datasets, tmp_path, monkeypatch
):
    from emotion_detection.cli import main
    from emotion_detection.experiments import reference as experiment

    calls = []

    def load(track, *args, **kwargs):
        calls.append(track)
        if track == "c":
            raise ValueError("C is not prepared")
        return track_datasets[track]

    def forbidden(*args, **kwargs):
        pytest.fail("No track may train before the complete suite passes preflight")

    monkeypatch.setattr(experiment, "load_prepared_reference_dataset", load)
    monkeypatch.setattr(experiment, "_train_one", forbidden)
    assert (
        main(["train", "reference", "--project-root", str(tmp_path), "--device", "cpu"])
        == 2
    )
    assert calls == ["a", "b", "c"]
    assert not (tmp_path / "output").exists()


def test_suite_failure_preserves_completed_tracks(
    track_datasets, tmp_path, monkeypatch
):
    from emotion_detection.cli import main
    from emotion_detection.experiments import reference as experiment

    calls = []
    monkeypatch.setattr(
        experiment,
        "load_prepared_reference_dataset",
        lambda track, *args, **kw: track_datasets[track],
    )

    def train(options, data, *, output_dir):
        calls.append(options.track)
        if options.track == "b":
            raise RuntimeError("synthetic B failure")
        path = output_dir / options.run_id
        (path / "checkpoint").mkdir(parents=True)
        (path / "checkpoint/keep.txt").write_text("A completed")
        (path / "metrics.json").write_text('{"eval_macro_f1": 0.7}')
        return path

    monkeypatch.setattr(experiment, "_train_one", train)
    assert (
        main(
            [
                "train",
                "reference",
                "--project-root",
                str(tmp_path),
                "--device",
                "cpu",
                "--run-id",
                "partial",
                "--skip-test",
            ]
        )
        == 2
    )
    summary = json.loads((tmp_path / "output/runs/partial/summary.json").read_text())
    assert calls == ["a", "b"]
    assert summary["status"] == "failed"
    assert summary["tracks"]["a"]["status"] == "completed"
    assert summary["tracks"]["b"]["status"] == "failed"
    assert summary["tracks"]["c"]["status"] == "pending"
    assert (
        tmp_path / "output/runs/partial/a/checkpoint/keep.txt"
    ).read_text() == "A completed"


def test_c_rejects_project_checkpoint_with_target_supervision(tmp_path):
    from emotion_detection.cli import main

    checkpoint = tmp_path / "supervised"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps({"track": "a", "data": {"provenance": {"language": "eng"}}})
    )
    assert (
        main(
            [
                "train",
                "reference",
                "--track",
                "c",
                "--model-name",
                str(checkpoint),
                "--project-root",
                str(tmp_path),
                "--device",
                "cpu",
            ]
        )
        == 2
    )
    assert not (tmp_path / "output").exists()


def test_suite_rejects_incompatible_monitor_before_preparation(tmp_path, monkeypatch):
    from emotion_detection.cli import main
    from emotion_detection.experiments import reference as experiment

    def forbidden(*args, **kwargs):
        pytest.fail("Invalid suite settings must fail before data acquisition")

    monkeypatch.setattr(experiment, "prepare_reference_dataset", forbidden)
    assert (
        main(
            [
                "train",
                "reference",
                "--monitor",
                "macro_f1",
                "--prepare-data",
                "--device",
                "cpu",
                "--project-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert not (tmp_path / "output").exists()
