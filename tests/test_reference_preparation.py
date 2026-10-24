"""Independent Track A preparation and classifier input contract; no training."""

import hashlib
import json
from dataclasses import replace

import pandas as pd
import pytest

from emotion_detection.datasets import ReferenceDataset, reference
from emotion_detection.datasets.reference_cli import main
from emotion_detection.datasets.reference_track_a import prepare_track_a
from emotion_detection.datasets.reference_types import ReferencePreparationError
from emotion_detection.datasets.sources import (
    ENGLISH_LABELS,
    SIX_LABELS,
    get_reference_sources,
)


def row(identifier, text, *, labels=ENGLISH_LABELS, **values):
    return {
        "id": identifier,
        "text": text,
        **{name: values.get(name, 0) for name in labels},
    }


def frames(labels=ENGLISH_LABELS):
    return {
        "train": pd.DataFrame(
            [
                row("train-1", "Train Only", labels=labels, joy=1),
                row("train-2", "Another train", labels=labels),
            ]
        ),
        "dev": pd.DataFrame(
            [
                row("dev-2", "Development", labels=labels),
                row("dev-1", "Another dev", labels=labels),
            ]
        ),
        "test": pd.DataFrame(
            [
                row("test-2", "Held out", labels=labels),
                row("test-1", "Another test", labels=labels),
            ]
        ),
    }


def prepare(inputs, labels=ENGLISH_LABELS):
    return prepare_track_a(
        inputs,
        label_names=labels,
        language="eng" if labels == ENGLISH_LABELS else "deu",
        source_metadata=[],
    )


def test_a_only_examples_are_retained_without_loading_track_b():
    inputs = frames()
    original = {key: value.copy(deep=True) for key, value in inputs.items()}
    data, _ = prepare(inputs)
    assert isinstance(data, ReferenceDataset)
    assert data.train["id"].tolist() == ["train-1", "train-2"]
    assert data.test["id"].tolist() == ["test-2", "test-1"]
    assert data.train.iloc[0]["text"] == "Train Only"
    assert data.label_names == ENGLISH_LABELS
    json.dumps(data.metadata, allow_nan=False)
    for split in inputs:
        pd.testing.assert_frame_equal(inputs[split], original[split])


def test_six_emotion_schema_keeps_disgust():
    inputs = frames(SIX_LABELS)
    inputs["train"].loc[0, "disgust"] = 1
    data, _ = prepare(inputs, SIX_LABELS)
    assert data.label_names == SIX_LABELS
    assert data.train.iloc[0]["disgust"] == 1


def test_column_reordering_does_not_change_targets():
    inputs = frames()
    first, _ = prepare(inputs)
    reordered = {
        split: frame[list(reversed(frame.columns))] for split, frame in inputs.items()
    }
    second, _ = prepare(reordered)
    for split in inputs:
        pd.testing.assert_frame_equal(getattr(first, split), getattr(second, split))


def test_heldout_duplicates_and_conflicts_keep_ids_order_and_labels():
    inputs = frames()
    inputs["test"] = pd.DataFrame(
        [
            row("t-3", "Same text", fear=1),
            row("t-1", "SAME TEXT", fear=0),
            row("t-2", "Same text", fear=1),
        ]
    )
    data, audits = prepare(inputs)
    assert data.test["id"].tolist() == ["t-3", "t-1", "t-2"]
    assert data.test["fear"].tolist() == [1, 0, 1]
    assert data.test["text"].tolist() == inputs["test"]["text"].tolist()
    assert len(audits["duplicates"]) == 3
    assert audits["duplicates"]["conflicting_annotations"].all()
    assert data.metadata["test_ids_and_order_preserved"]


def test_training_conflicts_are_reported_not_silently_adjudicated():
    inputs = frames()
    inputs["train"] = pd.DataFrame(
        [row("a", "Repeated", anger=1), row("b", "Repeated", anger=0)]
    )
    data, audits = prepare(inputs)
    assert len(data.train) == 2
    assert data.train.anger.tolist() == [1, 0]
    assert len(audits["duplicates"]) == 2


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("bad", [None, "", 2, "unknown"])
def test_invalid_evaluation_annotations_are_explicit_errors(split, bad):
    inputs = frames()
    inputs[split]["fear"] = inputs[split]["fear"].astype(object)
    inputs[split].loc[0, "fear"] = bad
    with pytest.raises(
        ReferencePreparationError, match="not silently removed"
    ) as error:
        prepare(inputs)
    issues = error.value.diagnostics["annotation_issues"]
    assert issues.iloc[0]["emotion"] == "fear"
    assert issues.iloc[0]["split"] == split


def test_invalid_training_annotation_is_excluded_not_zero_filled():
    inputs = frames()
    inputs["train"]["fear"] = pd.Series([None, 1], dtype=object)
    data, audits = prepare(inputs)
    assert data.train["id"].tolist() == ["train-2"]
    assert data.train.fear.tolist() == [1]
    assert len(audits["annotation_issues"]) == 1
    assert len(audits["exclusions"]) == 1


def test_missing_schema_column_cannot_be_fabricated():
    inputs = frames(SIX_LABELS)
    inputs["test"] = inputs["test"].drop(columns="disgust")
    with pytest.raises(ReferencePreparationError, match="missing fields"):
        prepare(inputs, SIX_LABELS)


def test_overlap_removal_is_deterministic_and_preserves_test():
    inputs = frames()
    inputs["train"] = pd.concat(
        [
            inputs["train"],
            pd.DataFrame(
                [
                    row("train-extra", "HELD OUT"),
                    row("train-dev", "Development"),
                    row("test-1", "Different text"),
                ]
            ),
        ],
        ignore_index=True,
    )
    inputs["dev"] = pd.concat(
        [inputs["dev"], pd.DataFrame([row("dev-extra", "held out")])], ignore_index=True
    )
    data, audits = prepare(inputs)
    assert data.train.id.tolist() == ["train-1", "train-2"]
    assert data.dev.id.tolist() == ["dev-2", "dev-1"]
    assert data.test.id.tolist() == ["test-2", "test-1"]
    assert len(audits["overlaps"]) == 4
    again, repeated_audits = prepare(inputs)
    assert data.metadata == again.metadata
    pd.testing.assert_frame_equal(audits["overlaps"], repeated_audits["overlaps"])


@pytest.mark.parametrize("split", ["dev", "test"])
def test_evaluation_duplicate_ids_fail_instead_of_changing_ids(split):
    inputs = frames()
    inputs[split]["id"] = "duplicate"
    with pytest.raises(ReferencePreparationError) as error:
        prepare(inputs)
    assert (
        "duplicate_id" in error.value.diagnostics["annotation_issues"].reason.tolist()
    )


@pytest.fixture
def local_inputs(tmp_path, monkeypatch):
    specs = []
    data = frames()
    for source in get_reference_sources():
        path = (
            tmp_path
            / "data/raw/reference_baselines"
            / source.release
            / source.relative_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        data[source.split].to_csv(path, index=False)
        specs.append(
            replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )
    monkeypatch.setattr(
        reference, "get_reference_sources", lambda *a, **kw: tuple(specs)
    )
    return tmp_path, tuple(specs), data


def test_loader_produces_canonical_outputs_and_csv_audits(local_inputs, monkeypatch):
    root, sources, _ = local_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Prepared reference data must never read joint data or need a model"
        )

    from emotion_detection.datasets import acquisition

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    before = {
        root / "data/raw/reference_baselines" / s.release / s.relative_path: s.sha256
        for s in sources
    }
    first = reference.load_reference_dataset(project_root=root, interactive=False)
    outputs = list((root / "data/processed/reference_baselines").rglob("*.parquet"))
    outputs += list((root / "output/analysis/reference_baselines").rglob("*.csv"))
    initial = {p: p.read_bytes() for p in outputs}
    second = reference.load_reference_dataset(project_root=root, interactive=False)
    assert first.metadata == second.metadata
    assert all(path.read_bytes() == payload for path, payload in initial.items())
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest
        for path, digest in before.items()
    )
    assert (
        len(
            list(
                (root / "data/processed/reference_baselines/track_a/eng").glob(
                    "*.parquet"
                )
            )
        )
        == 3
    )
    assert not list(root.rglob("*.json*"))
    assert not (root / "data/processed/train.parquet").exists()
    manifest = pd.read_csv(
        root / "output/analysis/reference_baselines/track_a/eng/source_manifest.csv"
    )
    assert manifest.label_names.tolist() == ["|".join(ENGLISH_LABELS)] * 3


def test_prepared_data_passes_existing_classifier_validation_without_training(
    local_inputs,
):
    from emotion_detection.reference_baselines.data import (
        validate_frame,
        validate_partition_separation,
    )

    root, _, _ = local_inputs
    data = reference.load_reference_dataset(project_root=root, interactive=False)
    for frame in (data.train, data.dev, data.test):
        validate_frame(frame, data.label_names)
    validate_partition_separation(data.train, data.dev)


def test_cli_for_reference_data_only(local_inputs, capsys):
    root, _, _ = local_inputs
    assert main(["--project-root", str(root), "--non-interactive"]) == 0
    assert "prepared rows" in capsys.readouterr().out


def test_invalid_evaluation_writes_issues_but_not_canonical_data(
    local_inputs, monkeypatch, capsys
):
    root, sources, data = local_inputs
    source = sources[2]
    path = root / "data/raw/reference_baselines" / source.release / source.relative_path
    data["test"].loc[0, "fear"] = 9
    data["test"].to_csv(path, index=False)
    specs = (
        *sources[:2],
        replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    monkeypatch.setattr(reference, "get_reference_sources", lambda *a, **kw: specs)
    assert main(["--project-root", str(root), "--non-interactive"]) == 2
    assert not (root / "data/processed/reference_baselines/track_a/eng").exists()
    issues = pd.read_csv(
        root / "output/analysis/reference_baselines/track_a/eng/annotation_issues.csv"
    )
    assert issues.iloc[0].emotion == "fear"
    assert "[ERROR]" in capsys.readouterr().err
