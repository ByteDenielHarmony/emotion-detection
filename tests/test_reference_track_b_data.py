"""Independent ordinal data preparation, acquisition and read-only CLI; no models."""

import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from emotion_detection.cli import main
from emotion_detection.datasets import acquisition, reference
from emotion_detection.datasets.reference_track_a import prepare_track_a
from emotion_detection.datasets.reference_track_b import prepare_track_b
from emotion_detection.datasets.reference_types import (
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import (
    ENGLISH_LABELS,
    OFFICIAL_REVISION,
    SIX_LABELS,
    SPLITS,
    TrackBLanguageSpec,
    available_reference_languages,
    get_reference_sources,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Track B data tests must not use the network")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)


def row(identifier, text, labels=ENGLISH_LABELS, **values):
    return {
        "id": identifier,
        "text": text,
        **{name: values.get(name, 0) for name in labels},
    }


def inputs(labels=ENGLISH_LABELS):
    return {
        split: pd.DataFrame(
            [
                row(f"{split}-02", f"{split} Привіт 你好", labels, anger=2, joy=3),
                row(f"{split}-01", f"{split} Another text", labels, fear=1),
            ]
        )
        for split in SPLITS
    }


def prepare(frames, labels=ENGLISH_LABELS):
    return prepare_track_b(
        frames, label_names=labels, language="eng", source_metadata=[]
    )


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def seed(root, monkeypatch, language="eng"):
    sources = get_reference_sources("b", language)
    frames = inputs(sources[0].label_names)
    selected = []
    for source in sources:
        path = (
            root
            / "data/raw/reference_baselines"
            / source.release
            / source.relative_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[source.split].to_csv(path, index=False)
        selected.append(
            replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )

    def select(track, chosen_language, release):
        assert (track, chosen_language, release) == ("b", language, sources[0].release)
        return tuple(selected)

    monkeypatch.setattr(reference, "get_reference_sources", select)
    return tuple(selected), frames


def test_b_availability_requires_complete_own_inputs():
    assert available_reference_languages("b") == (
        "amh",
        "arq",
        "chn",
        "deu",
        "eng",
        "esp",
        "hau",
        "ptbr",
        "rus",
        "ukr",
    )
    with pytest.raises(ValueError, match="no Track B train/ron.csv"):
        get_reference_sources("b", "ron")
    with pytest.raises(ValueError, match="Unsupported Track B language"):
        get_reference_sources("b", "afr")
    with pytest.raises(ValueError, match="Unsupported release"):
        available_reference_languages("b", "main")


@pytest.mark.parametrize("language", available_reference_languages("b"))
def test_b_sources_are_independent_and_pinned(language):
    sources = get_reference_sources("b", language)
    a_sources = get_reference_sources("a", language)
    assert tuple(s.split for s in sources) == SPLITS
    for source, a_source in zip(sources, a_sources, strict=True):
        assert source.track == "b"
        assert source.revision == OFFICIAL_REVISION
        assert source.label_names == (
            ENGLISH_LABELS if language == "eng" else SIX_LABELS
        )
        assert source.url.endswith(f"/track_b/{source.split}/{language}.csv")
        assert source.relative_path == f"track_b/{language}/{source.split}.csv"
        assert source.sha256 != a_source.sha256


def test_b_manifest_matches_the_inspected_official_release():
    # Locked after checking all 30 files against their official Git blob IDs,
    # including each schema and all ordinal values, not copied from Track A.
    manifest = {
        language: [s.metadata() for s in get_reference_sources("b", language)]
        for language in available_reference_languages("b")
    }
    encoded = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        "183d208310ecfcb9a23a1d510fe41db65773fd13f6609c642edf6db2d8ce9f2b"
    )


def test_b_registration_is_validated_and_immutable():
    spec = TrackBLanguageSpec(ENGLISH_LABELS, "0" * 64, "1" * 64, "2" * 64)
    with pytest.raises(FrozenInstanceError):
        spec.label_names = SIX_LABELS
    with pytest.raises(ValueError, match="Language schema"):
        replace(spec, label_names=("joy", "joy"))
    for split in SPLITS:
        with pytest.raises(ValueError, match=f"Invalid SHA-256 for Track B {split}"):
            replace(spec, **{f"{split}_sha256": "invalid"})
    with pytest.raises(TypeError):
        TrackBLanguageSpec(ENGLISH_LABELS, train_sha256="0" * 64, dev_sha256="1" * 64)


@pytest.mark.parametrize("labels", [ENGLISH_LABELS, SIX_LABELS])
def test_b_preserves_intensities_order_text_and_inputs(labels):
    frames = inputs(labels)
    original = {split: frame.copy(deep=True) for split, frame in frames.items()}
    data, _ = prepare(frames, labels)
    assert data.metadata["track"] == "b"
    assert data.metadata["target_values"] == [0, 1, 2, 3]
    assert data.metadata["preparation_variant"] == "official-b-leakage-controlled-v1"
    assert data.metadata["test_ids_and_order_preserved"]
    json.dumps(data.metadata, allow_nan=False)
    for split in SPLITS:
        frame = getattr(data, split)
        pd.testing.assert_frame_equal(frame[["id", "text", *labels]], original[split])
        pd.testing.assert_frame_equal(frames[split], original[split])
        assert frame.anger.tolist() == [2, 0]
        assert frame.joy.tolist() == [3, 0]
        assert frame.source_row.tolist() == [0, 1]
        assert frame.source_split.eq(split).all()
    with pytest.raises(ReferencePreparationError):
        prepare_track_a(frames, label_names=labels, language="eng", source_metadata=[])


def test_b_column_order_is_resolved_by_names():
    frames = inputs(SIX_LABELS)
    frames["train"]["disgust"] = [3, 2]
    first, _ = prepare(frames, SIX_LABELS)
    second, _ = prepare(
        {s: f[list(reversed(f.columns))] for s, f in frames.items()}, SIX_LABELS
    )
    for split in SPLITS:
        pd.testing.assert_frame_equal(getattr(first, split), getattr(second, split))
    assert second.train.disgust.tolist() == [3, 2]


@pytest.mark.parametrize("split", ["dev", "test"])
@pytest.mark.parametrize("bad", [None, "", "unknown", -1, 4, 1.5, float("inf")])
def test_b_invalid_evaluation_is_not_dropped_or_imputed(split, bad):
    frames = inputs()
    frames[split]["fear"] = frames[split]["fear"].astype(object)
    frames[split].loc[0, "fear"] = bad
    with pytest.raises(
        ReferencePreparationError, match="not silently removed"
    ) as error:
        prepare(frames)
    issue = error.value.diagnostics["annotation_issues"].iloc[0]
    assert (issue.split, issue.emotion, issue.reason) == (
        split,
        "fear",
        "missing_or_invalid_ordinal_annotation",
    )


def test_b_invalid_training_is_excluded_not_converted_to_absence():
    frames = inputs()
    frames["train"]["fear"] = pd.Series([None, 3], dtype=object)
    data, diagnostics = prepare(frames)
    assert data.train.id.tolist() == ["train-01"]
    assert data.train.fear.tolist() == [3]
    assert data.train.source_row.tolist() == [1]
    assert len(diagnostics["exclusions"]) == 1


@pytest.mark.parametrize("split", SPLITS)
def test_b_retains_valid_within_split_conflicts(split):
    frames = inputs()
    frames[split] = pd.DataFrame([row("02", "Same", fear=3), row("01", "SAME", fear=1)])
    data, diagnostics = prepare(frames)
    assert getattr(data, split).id.tolist() == ["02", "01"]
    assert getattr(data, split).fear.tolist() == [3, 1]
    assert diagnostics["duplicates"].conflicting_annotations.all()


def test_b_overlap_policy_keeps_test_and_source_order():
    frames = inputs()
    frames["train"] = pd.concat(
        [
            frames["train"],
            pd.DataFrame(
                [
                    row("shared", "test Another text", joy=3),
                    row("dev-02", "Different content", anger=2),
                ]
            ),
        ],
        ignore_index=True,
    )
    frames["dev"] = pd.concat(
        [frames["dev"], pd.DataFrame([row("dev-extra", "TEST ANOTHER TEXT", joy=1)])],
        ignore_index=True,
    )
    data, diagnostics = prepare(frames)
    assert data.train.id.tolist() == ["train-02", "train-01"]
    assert data.dev.id.tolist() == ["dev-02", "dev-01"]
    assert len(diagnostics["overlaps"]) == 3
    pd.testing.assert_frame_equal(
        data.test[["id", "text", *ENGLISH_LABELS]], frames["test"]
    )
    again, repeated = prepare(frames)
    assert again.metadata == data.metadata
    for name in diagnostics:
        pd.testing.assert_frame_equal(diagnostics[name], repeated[name])


@pytest.mark.parametrize(
    "problem",
    [
        "schema",
        "extra_emotion",
        "duplicate_id",
        "blank_text",
        "missing_split",
        "empty_split",
    ],
)
def test_b_rejects_malformed_inputs(problem):
    frames = inputs()
    if problem == "schema":
        frames["test"] = frames["test"].drop(columns="fear")
    elif problem == "extra_emotion":
        frames["test"]["disgust"] = 0
    elif problem == "duplicate_id":
        frames["test"]["id"] = "same"
    elif problem == "blank_text":
        frames["test"].loc[0, "text"] = "  "
    elif problem == "missing_split":
        del frames["train"]
    else:
        frames["dev"] = frames["dev"].iloc[:0]
    with pytest.raises(ReferencePreparationError):
        prepare(frames)


@pytest.mark.parametrize("language", available_reference_languages("b"))
def test_b_cli_prepare_repeat_and_read_only_check(
    language, tmp_path, monkeypatch, capsys
):
    sources, _ = seed(tmp_path, monkeypatch, language)
    args = ["--track", "b", "--language", language, "--project-root", str(tmp_path)]
    assert main(["data", "prepare", *args, "--non-interactive"]) == 0
    before = snapshot(tmp_path)
    assert main(["data", "prepare", *args, "--non-interactive"]) == 0
    assert snapshot(tmp_path) == before
    monkeypatch.setattr(
        reference,
        "prepare_reference_dataset",
        lambda *a, **kw: pytest.fail("Read-only check must not prepare"),
    )
    assert main(["data", "check", *args]) == 0
    assert "Ready:" in capsys.readouterr().out
    assert snapshot(tmp_path) == before
    data = reference.load_prepared_reference_dataset(
        "b", language, project_root=tmp_path
    )
    assert data.train.joy.tolist() == [3, 0]
    assert data.label_names == sources[0].label_names
    assert not (tmp_path / "data/raw/track_a").exists()
    assert not (
        tmp_path / "data/raw/reference_baselines" / sources[0].release / "track_a"
    ).exists()
    assert not list(tmp_path.rglob("*.json*"))
    # Prepared loading is independent of raw inputs and remains strictly read-only.
    for source in sources:
        (
            tmp_path
            / "data/raw/reference_baselines"
            / source.release
            / source.relative_path
        ).unlink()
    before = snapshot(tmp_path)
    reference.load_prepared_reference_dataset("b", language, project_root=tmp_path)
    assert snapshot(tmp_path) == before


def test_b_downloads_only_missing_own_source_files(tmp_path, monkeypatch):
    sources, frames = seed(tmp_path, monkeypatch)
    missing = sources[1]
    path = (
        tmp_path
        / "data/raw/reference_baselines"
        / missing.release
        / missing.relative_path
    )
    payload = path.read_bytes()
    path.unlink()
    calls = []

    def download(request, **kwargs):
        calls.append(request.full_url)
        assert request.full_url == missing.url
        return io.BytesIO(payload)

    monkeypatch.setattr(acquisition, "urlopen", download)
    data = reference.prepare_reference_dataset(
        "b", project_root=tmp_path, interactive=False
    )
    assert calls == [missing.url]
    assert data.dev.joy.tolist() == frames["dev"].joy.tolist()


def test_b_failure_writes_csv_issues_not_canonical_files(tmp_path, monkeypatch):
    sources, frames = seed(tmp_path, monkeypatch)
    source = sources[2]
    path = (
        tmp_path
        / "data/raw/reference_baselines"
        / source.release
        / source.relative_path
    )
    frames["test"].loc[0, "joy"] = 4
    frames["test"].to_csv(path, index=False)
    sources = (
        *sources[:2],
        replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest()),
    )
    monkeypatch.setattr(reference, "get_reference_sources", lambda *a, **kw: sources)
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "b",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 2
    )
    assert not (tmp_path / "data/processed/reference_baselines/track_b/eng").exists()
    issues = pd.read_csv(
        tmp_path
        / "output/analysis/reference_baselines/track_b/eng/annotation_issues.csv"
    )
    assert issues.iloc[0].reason == "missing_or_invalid_ordinal_annotation"


@pytest.mark.parametrize(
    "tamper", ["annotation", "variant", "domain", "manifest", "fingerprint"]
)
def test_b_read_only_loader_rejects_tampering(tmp_path, monkeypatch, tamper):
    seed(tmp_path, monkeypatch)
    reference.prepare_reference_dataset("b", project_root=tmp_path, interactive=False)
    directory = tmp_path / "output/analysis/reference_baselines/track_b/eng"
    path = directory / "metadata.csv"
    records = pd.read_csv(path, keep_default_na=False)
    if tamper in {"annotation", "fingerprint"}:
        canonical = (
            tmp_path / "data/processed/reference_baselines/track_b/eng/train.parquet"
        )
        frame = pd.read_parquet(canonical)
        frame.loc[0, "joy"] = 4 if tamper == "annotation" else 2
        frame.to_parquet(canonical, index=False)
        if tamper == "annotation":
            # Validation must reject invalid targets even with a matching fingerprint.
            fingerprints = json.loads(
                records.loc[records.key.eq("prepared_fingerprints"), "value"].iloc[0]
            )
            fingerprints["train"] = prepared_fingerprint(frame)
            records.loc[records.key.eq("prepared_fingerprints"), "value"] = json.dumps(
                fingerprints
            )
    elif tamper == "variant":
        records.loc[records.key.eq("preparation_variant"), "value"] = json.dumps(
            "official-a-leakage-controlled-v1"
        )
    elif tamper == "domain":
        records.loc[records.key.eq("target_values"), "value"] = json.dumps([0, 1])
    else:
        manifest = pd.read_csv(directory / "source_manifest.csv", keep_default_na=False)
        manifest.loc[0, "track"] = "a"
        manifest.to_csv(directory / "source_manifest.csv", index=False)
    records.to_csv(path, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(ReferencePreparationError, match="emotion data prepare"):
        reference.load_prepared_reference_dataset("b", project_root=tmp_path)
    assert snapshot(tmp_path) == before


def test_b_cli_language_discovery_and_missing_data_are_read_only(tmp_path, capsys):
    args = ["--track", "b", "--project-root", str(tmp_path)]
    assert main(["data", "languages", *args]) == 0
    assert tuple(capsys.readouterr().out.split()) == available_reference_languages("b")
    assert main(["data", "check", *args]) == 2
    assert "emotion data prepare" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_b_incomplete_romanian_is_rejected_before_creating_files(tmp_path, capsys):
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                "b",
                "--language",
                "ron",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 2
    )
    assert "no Track B train/ron.csv" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_b_compatibility_preparation_api_and_cli(tmp_path, monkeypatch, capsys):
    from emotion_detection.datasets import reference_cli

    seed(tmp_path, monkeypatch)
    data = reference.load_reference_dataset(
        "b", project_root=tmp_path, interactive=False
    )
    assert data.train.joy.tolist() == [3, 0]
    before = snapshot(tmp_path)
    assert (
        reference_cli.main(
            [
                "--track",
                "b",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 0
    )
    assert "prepared rows" in capsys.readouterr().out
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("track", ["invalid", "d"])
def test_unknown_training_track_fails_before_preparation_or_model_creation(
    track, tmp_path, monkeypatch, capsys
):
    from emotion_detection.experiments import reference as experiment

    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported training must not prepare data or construct a model")

    monkeypatch.setattr(experiment, "TrackAClassifier", forbidden)
    monkeypatch.setattr(experiment, "prepare_reference_dataset", forbidden)
    assert (
        main(
            [
                "train",
                "reference",
                "--track",
                track,
                "--prepare-data",
                "--project-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "Select track" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())
