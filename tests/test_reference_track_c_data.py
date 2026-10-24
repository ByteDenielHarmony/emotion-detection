"""Cross-lingual selection, masked labels and four-role persistence; synthetic only."""

import hashlib
import io
import json
from dataclasses import FrozenInstanceError, replace

import pandas as pd
import pytest

from emotion_detection.cli import main
from emotion_detection.datasets import acquisition, reference
from emotion_detection.datasets import reference_crosslingual as persistence
from emotion_detection.datasets.reference_track_c import (
    prepare_track_c,
    source_key,
)
from emotion_detection.datasets.reference_types import (
    CrosslingualReferenceDataset,
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import (
    DEFAULT_RELEASE,
    SIX_LABELS,
    available_reference_languages,
    get_reference_sources,
)
from emotion_detection.datasets.sources_track_c import (
    DEFAULT_C_PROTOCOL,
    TrackCTargetSpec,
    get_crosslingual_sources,
    track_c_source_languages,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("C data tests must not use the real network")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)


def make_inputs(language="tat"):
    sources = get_crosslingual_sources(language)
    frames = {}
    for source in sources:
        frames[source_key(source)] = pd.DataFrame(
            [
                {
                    "id": f"{source.split}-00{index}",
                    "text": f"{source.track} {source.language} {source.split} {index} Привіт 你好",
                    **{
                        name: int((index + n) % 2 == 0)
                        for n, name in enumerate(source.label_names)
                    },
                }
                for index in (2, 1)
            ]
        )
    return sources, frames


def prepare(sources, frames, language="tat"):
    return prepare_track_c(
        frames, sources=sources, language=language, release=DEFAULT_RELEASE
    )


def snapshot(root):
    return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def seed(root, monkeypatch, language="tat", frames=None):
    sources, defaults = make_inputs(language)
    frames = defaults if frames is None else frames
    selected = []
    for source in sources:
        path = (
            root
            / "data/raw/reference_baselines"
            / source.release
            / "crosslingual"
            / source.relative_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        frames[source_key(source)].to_csv(path, index=False)
        selected.append(
            replace(source, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        )

    def select(chosen, release, protocol):
        assert (chosen, release, protocol) == (
            language,
            DEFAULT_RELEASE,
            DEFAULT_C_PROTOCOL,
        )
        return tuple(selected)

    monkeypatch.setattr(persistence, "get_crosslingual_sources", select)
    return tuple(selected), frames


def test_c_has_32_targets_and_no_target_training_files():
    assert len(available_reference_languages("c")) == 32
    assert set(available_reference_languages("c")) - set(
        available_reference_languages("a")
    ) == {"ind", "jav", "xho", "zul"}
    for language in available_reference_languages("c"):
        targets = get_reference_sources("c", language)
        assert [s.split for s in targets] == ["dev", "test"]
        assert all(s.track == "c" and "/track_c/" in s.url for s in targets)
        assert all(s.relative_path.startswith(f"track_c/{language}/") for s in targets)


@pytest.mark.parametrize(
    "language,sources",
    [
        ("tat", ("rus", "ukr")),
        ("pcm", ("swa", "yor")),
        ("chn", ("rus",)),
        ("ind", ("sun",)),
        ("jav", ("sun",)),
    ],
)
def test_c_explicit_exceptions(language, sources):
    assert track_c_source_languages(language) == sources


def test_c_uses_broad_task_paper_families_not_inferred_subgroups():
    assert track_c_source_languages("eng") == (
        "afr",
        "deu",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    )
    assert track_c_source_languages("amh") == ("arq", "ary", "hau", "orm", "som", "tir")
    assert track_c_source_languages("xho") == ("ibo", "kin", "swa", "vmw", "yor")


def test_c_spec_is_validated_and_frozen():
    spec = TrackCTargetSpec(SIX_LABELS, "0" * 64, "1" * 64)
    with pytest.raises(FrozenInstanceError):
        spec.dev_sha256 = "2" * 64
    with pytest.raises(ValueError):
        replace(spec, dev_sha256="bad")
    with pytest.raises(ValueError):
        replace(spec, label_names=("joy", "joy"))


def test_c_pinned_manifests_and_protocol_selections_match_reviewed_baseline():
    manifest = {
        language: [s.metadata() for s in get_reference_sources("c", language)]
        for language in available_reference_languages("c")
    }
    encoded = json.dumps(
        manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "a67abfe069524f537be2e8aa09d3f0b090d3c01a5d7fd6fc68fe06d261db6b5b"
    )
    selection = {
        language: list(track_c_source_languages(language))
        for language in available_reference_languages("c")
        if language != "sun"
    }
    encoded = json.dumps(selection, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(encoded).hexdigest() == (
        "2df6c0a5272ecf1bfc4260164d6dbdd0d1e4f13ceb5057e85083af70c4b25ce7"
    )


@pytest.mark.parametrize(
    "language", [l for l in available_reference_languages("c") if l != "sun"]
)
def test_every_c_selection_prepares_only_permitted_roles(language):
    sources, frames = make_inputs(language)
    original = {key: f.copy(deep=True) for key, f in frames.items()}
    data, _ = prepare(sources, frames, language)
    assert isinstance(data, CrosslingualReferenceDataset)
    assert not hasattr(data, "dev")
    assert data.metadata["source_languages"] == list(track_c_source_languages(language))
    assert data.model_label_names == SIX_LABELS
    assert data.label_names == sources[-1].label_names
    for source in sources:
        assert source.track == "c" or (
            source.track == "a"
            and source.split != "test"
            and source.language != language
        )
        pd.testing.assert_frame_equal(
            frames[source_key(source)], original[source_key(source)]
        )
    for role in ("target_dev", "test"):
        split = "dev" if role == "target_dev" else "test"
        target = getattr(data, role)
        pd.testing.assert_frame_equal(
            target[["id", "text"]], frames[("c", language, split)][["id", "text"]]
        )
        for name in data.label_names:
            assert (
                target[name].tolist() == frames[("c", language, split)][name].tolist()
            )
    json.dumps(data.metadata, allow_nan=False)


def test_native_schema_gaps_remain_unknown_with_masks():
    sources, frames = make_inputs("deu")
    data, _ = prepare(sources, frames, "deu")
    english = data.train[data.train.source_language.eq("eng")]
    afrikaans = data.train[data.train.source_language.eq("afr")]
    assert english.disgust.isna().all() and not english.known_disgust.any()
    assert afrikaans.surprise.isna().all() and not afrikaans.known_surprise.any()
    assert english.known_surprise.all() and afrikaans.known_disgust.all()


def test_cell_level_missing_labels_are_masked_not_zero_filled():
    sources, frames = make_inputs()
    frames[("a", "rus", "train")]["fear"] = pd.Series([None, ""], dtype=object)
    frames[("c", "tat", "test")]["joy"] = pd.Series([None, 1], dtype=object)
    data, audits = prepare(sources, frames)
    russian = data.train[data.train.source_language.eq("rus")]
    assert russian.fear.isna().all() and not russian.known_fear.any()
    assert pd.isna(data.test.iloc[0].joy) and not data.test.iloc[0].known_joy
    assert len(audits["annotation_issues"]) == 3
    assert set(audits["annotation_issues"].reason) == {"missing_annotation_masked"}


def test_c_same_local_ids_in_different_languages_are_not_collapsed():
    sources, frames = make_inputs()
    data, _ = prepare(sources, frames)
    assert len(data.train) == 4 and len(data.source_dev) == 4
    assert data.train.id.tolist() == [
        "train-002",
        "train-001",
        "train-002",
        "train-001",
    ]


def test_c_leakage_controls_preserve_both_target_partitions():
    sources, frames = make_inputs()
    held_out = frames[("c", "tat", "test")].loc[0, "text"]
    frames[("c", "tat", "dev")].loc[0, "text"] = held_out
    frames[("a", "rus", "train")].loc[0, "text"] = held_out.upper()
    frames[("a", "rus", "dev")].loc[0, "text"] = held_out
    frames[("a", "ukr", "train")].loc[0, "text"] = frames[("a", "ukr", "dev")].loc[
        1, "text"
    ]
    data, audits = prepare(sources, frames)
    assert len(data.train) == 2 and len(data.source_dev) == 3
    assert len(data.target_dev) == 2 and len(data.test) == 2
    assert data.metadata["target_dev_test_overlap_texts"] == 1
    assert len(audits["exclusions"]) == 3
    assert len(audits["overlaps"]) == 6
    repeated, _ = prepare(sources, frames)
    assert data.metadata == repeated.metadata


def test_c_duplicates_and_conflicting_labels_are_retained():
    sources, frames = make_inputs()
    frame = frames[("c", "tat", "test")]
    frame.loc[1, "text"] = frame.loc[0, "text"]
    data, audits = prepare(sources, frames)
    assert len(data.test) == 2
    assert len(audits["duplicates"]) == 2
    assert audits["duplicates"].conflicting_annotations.all()


def test_c_cross_language_training_duplicates_are_reported_not_adjudicated():
    sources, frames = make_inputs()
    frames[("a", "ukr", "train")].loc[0, "text"] = frames[("a", "rus", "train")].loc[
        1, "text"
    ]
    data, audits = prepare(sources, frames)
    assert len(data.train) == 4
    assert len(audits["duplicates"]) == 2
    assert audits["duplicates"].conflicting_annotations.all()


@pytest.mark.parametrize(
    "key", [("a", "rus", "dev"), ("c", "tat", "dev"), ("c", "tat", "test")]
)
@pytest.mark.parametrize("problem", ["annotation", "id", "text"])
def test_c_invalid_validation_and_evaluation_fail_explicitly(key, problem):
    sources, frames = make_inputs()
    if problem == "annotation":
        frames[key].loc[0, "anger"] = 2
    elif problem == "id":
        frames[key]["id"] = "duplicate"
    else:
        frames[key].loc[0, "text"] = " "
    with pytest.raises(
        ReferencePreparationError, match="not silently removed"
    ) as error:
        prepare(sources, frames)
    assert not error.value.diagnostics["annotation_issues"].empty


def test_c_invalid_and_unsupervised_train_rows_are_excluded():
    sources, frames = make_inputs()
    frames[("a", "rus", "train")].loc[0, "anger"] = 2
    for name in SIX_LABELS:
        frames[("a", "rus", "train")][name] = frames[("a", "rus", "train")][
            name
        ].astype(object)
        frames[("a", "rus", "train")].loc[1, name] = None
    data, audits = prepare(sources, frames)
    assert set(data.train.source_language) == {"ukr"}
    assert len(audits["exclusions"]) == 2


@pytest.mark.parametrize(
    "problem",
    ["source_test", "target_train", "missing_input", "schema", "extra_schema"],
)
def test_c_forbids_unselected_data_and_invalid_schemas(problem):
    sources, frames = make_inputs()
    if problem in {"source_test", "target_train"}:
        extra = (
            get_reference_sources("a", "rus")[2]
            if problem == "source_test"
            else get_reference_sources("a", "tat")[0]
        )
        sources = (*sources, extra)
        frames[source_key(extra)] = frames[("a", "rus", "train")].copy()
    elif problem == "missing_input":
        del frames[("a", "rus", "dev")]
    elif problem == "schema":
        frames[("c", "tat", "test")] = frames[("c", "tat", "test")].drop(
            columns="anger"
        )
    else:
        sources, frames = make_inputs("eng")
        frames[("c", "eng", "test")]["disgust"] = 0
    with pytest.raises(ReferencePreparationError):
        prepare(sources, frames, "eng" if problem == "extra_schema" else "tat")


@pytest.mark.parametrize(
    "language", ["eng", "afr", "arq", "tat", "pcm", "chn", "ind", "xho"]
)
def test_c_prepare_repeat_read_only_cli_and_no_forbidden_inputs(
    language, tmp_path, monkeypatch, capsys
):
    sources, _ = seed(tmp_path, monkeypatch, language)
    args = ["--track", "c", "--language", language, "--project-root", str(tmp_path)]
    assert main(["data", "prepare", *args, "--non-interactive"]) == 0
    assert "source_dev:" in capsys.readouterr().out
    before = snapshot(tmp_path)
    assert main(["data", "prepare", *args, "--non-interactive"]) == 0
    assert snapshot(tmp_path) == before
    monkeypatch.setattr(
        persistence,
        "prepare_crosslingual_dataset",
        lambda *a, **kw: pytest.fail("read-only check prepared data"),
    )
    assert main(["data", "check", *args]) == 0
    assert snapshot(tmp_path) == before
    data = reference.load_prepared_reference_dataset(
        "c", language, project_root=tmp_path
    )
    assert data.metadata["source_languages"] == list(track_c_source_languages(language))
    raw = tmp_path / "data/raw/reference_baselines" / DEFAULT_RELEASE / "crosslingual"
    assert sorted(str(p.relative_to(raw)) for p in raw.rglob("*.csv")) == sorted(
        s.relative_path for s in sources
    )
    for source in sources:
        (raw / source.relative_path).unlink()
    before = snapshot(tmp_path)
    reference.load_prepared_reference_dataset("c", language, project_root=tmp_path)
    assert snapshot(tmp_path) == before
    assert not list(tmp_path.rglob("*.json*"))


def test_c_download_resume_fetches_only_missing_selected_file(tmp_path, monkeypatch):
    sources, _ = seed(tmp_path, monkeypatch)
    missing = sources[0]
    path = (
        tmp_path
        / "data/raw/reference_baselines"
        / DEFAULT_RELEASE
        / "crosslingual"
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
    reference.prepare_reference_dataset(
        "c", "tat", project_root=tmp_path, interactive=False
    )
    assert calls == [missing.url]


@pytest.mark.parametrize(
    "tamper",
    [
        "mask",
        "annotation",
        "schema_gap",
        "provenance",
        "row_order",
        "leakage",
        "manifest",
        "protocol",
        "policy",
        "counts",
        "target_population",
    ],
)
def test_c_read_only_loader_rejects_tampering_even_with_new_fingerprint(
    tamper, tmp_path, monkeypatch
):
    language = "eng"
    seed(tmp_path, monkeypatch, language)
    reference.prepare_reference_dataset(
        "c", language, project_root=tmp_path, interactive=False
    )
    directory = (
        tmp_path
        / "output/analysis/reference_baselines/track_c"
        / language
        / DEFAULT_C_PROTOCOL
    )
    metadata_path = directory / "metadata.csv"
    records = pd.read_csv(metadata_path, keep_default_na=False)
    metadata = {r.key: json.loads(r.value) for r in records.itertuples()}
    canonical = (
        tmp_path
        / "data/processed/reference_baselines/track_c"
        / language
        / DEFAULT_C_PROTOCOL
        / "train.parquet"
    )
    frame = pd.read_parquet(canonical)
    if tamper == "mask":
        frame.loc[0, "known_anger"] = False
    elif tamper == "annotation":
        frame.loc[0, "anger"] = 2
    elif tamper == "schema_gap":
        canonical = canonical.with_name("test.parquet")
        frame = pd.read_parquet(canonical)
        frame.loc[0, "disgust"] = 0
        frame.loc[0, "known_disgust"] = True
    elif tamper == "provenance":
        frame.loc[0, "source_language"] = language
    elif tamper == "row_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif tamper == "leakage":
        target = pd.read_parquet(canonical.with_name("test.parquet"))
        frame.loc[0, "text"] = target.loc[0, "text"]
    elif tamper == "manifest":
        manifest = pd.read_csv(directory / "source_manifest.csv", keep_default_na=False)
        manifest.loc[0, "split"] = "test"
        manifest.to_csv(directory / "source_manifest.csv", index=False)
    elif tamper == "policy":
        metadata["policies"]["validation"] = "target_dev"
    elif tamper == "counts":
        metadata["source_row_counts"]["a/afr/train"] += 1
    elif tamper == "target_population":
        canonical = canonical.with_name("target_dev.parquet")
        frame = pd.read_parquet(canonical).iloc[1:].reset_index(drop=True)
        metadata["split_counts"]["target_dev"]["output_rows"] = len(frame)
    else:
        metadata["protocol"] = "other-protocol"
    frame.to_parquet(canonical, index=False)
    metadata["prepared_fingerprints"][canonical.stem] = prepared_fingerprint(frame)
    records["value"] = [
        json.dumps(metadata[key], allow_nan=False) for key in records.key
    ]
    records.to_csv(metadata_path, index=False)
    before = snapshot(tmp_path)
    with pytest.raises(ReferencePreparationError, match="emotion data prepare"):
        reference.load_prepared_reference_dataset("c", language, project_root=tmp_path)
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("command", ["prepare", "check"])
def test_c_sun_and_unknown_protocol_fail_before_writes(command, tmp_path, capsys):
    args = ["data", command, "--track", "c", "--project-root", str(tmp_path)]
    assert main([*args, "--language", "sun"]) == 2
    assert "no available other Austronesian" in capsys.readouterr().err
    assert main([*args, "--protocol", "../bad"]) == 2
    assert "Unsupported Track C protocol" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_c_language_discovery_and_missing_check_are_read_only(tmp_path, capsys):
    assert (
        main(["data", "languages", "--track", "c", "--project-root", str(tmp_path)])
        == 0
    )
    assert tuple(capsys.readouterr().out.split()) == available_reference_languages("c")
    assert main(["data", "check", "--track", "c", "--project-root", str(tmp_path)]) == 2
    assert "emotion data prepare" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("track", ["a", "b"])
def test_c_protocol_cannot_silently_apply_to_other_tracks(track, tmp_path, capsys):
    assert (
        main(
            [
                "data",
                "prepare",
                "--track",
                track,
                "--protocol",
                DEFAULT_C_PROTOCOL,
                "--project-root",
                str(tmp_path),
            ]
        )
        == 2
    )
    assert "Track C only" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_c_failure_writes_csv_audits_without_canonical_outputs(tmp_path, monkeypatch):
    _, frames = make_inputs()
    frames[("c", "tat", "test")].loc[0, "anger"] = 3
    seed(tmp_path, monkeypatch, frames=frames)
    with pytest.raises(ReferencePreparationError):
        reference.prepare_reference_dataset(
            "c", "tat", project_root=tmp_path, interactive=False
        )
    assert not list((tmp_path / "data/processed").rglob("*.parquet"))
    issues = pd.read_csv(
        tmp_path
        / "output/analysis/reference_baselines/track_c/tat"
        / DEFAULT_C_PROTOCOL
        / "annotation_issues.csv"
    )
    assert "invalid_binary_annotation" in issues.reason.tolist()


def test_c_masks_round_trip_without_filling_missing_cells(tmp_path, monkeypatch):
    _, frames = make_inputs()
    frames[("a", "rus", "train")]["fear"] = pd.Series([None, 1], dtype=object)
    frames[("c", "tat", "test")]["joy"] = pd.Series([None, 0], dtype=object)
    seed(tmp_path, monkeypatch, frames=frames)
    prepared = reference.prepare_reference_dataset(
        "c", "tat", project_root=tmp_path, interactive=False
    )
    loaded = reference.load_prepared_reference_dataset(
        "c", "tat", project_root=tmp_path
    )
    pd.testing.assert_frame_equal(prepared.test, loaded.test)
    assert pd.isna(loaded.test.iloc[0].joy)
    assert not loaded.test.iloc[0].known_joy
    assert loaded.test.iloc[1].joy == 0 and loaded.test.iloc[1].known_joy


def test_c_legacy_cli_and_preparation_wrapper(tmp_path, monkeypatch, capsys):
    from emotion_detection.datasets import reference_cli

    seed(tmp_path, monkeypatch)
    data = reference.load_reference_dataset(
        "c", "tat", project_root=tmp_path, interactive=False
    )
    assert isinstance(data, CrosslingualReferenceDataset)
    before = snapshot(tmp_path)
    assert (
        reference_cli.main(
            [
                "--track",
                "c",
                "--language",
                "tat",
                "--project-root",
                str(tmp_path),
                "--non-interactive",
            ]
        )
        == 0
    )
    assert "source_dev:" in capsys.readouterr().out
    assert snapshot(tmp_path) == before
