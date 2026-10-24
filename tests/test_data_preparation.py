"""Synthetic tests do not require downloaded BRIGHTER data."""

import hashlib

import pandas as pd
import pytest

from emotion_detection.datasets.prepare import (
    EMOTIONS,
    SPLITS,
    audit,
    clean_track,
    exclude_cross_split_overlap,
    merge_tracks,
    prepare,
    write_diagnostic_csv,
)


def row(identifier, text, **labels):
    return {
        "id": identifier,
        "text": text,
        "disgust": None,
        **{emotion: labels.get(emotion, 0) for emotion in EMOTIONS},
    }


def clean(rows, track="a", split="train"):
    return clean_track(pd.DataFrame(rows), track, split)


def test_lowercase_text_is_key_not_source_id():
    a, *_ = clean([row("unrelated-a-42", "I am HAPPY!", joy=1)])
    b, *_ = clean([row("unrelated-b-99", "I am happy!", joy=3)], "b")
    paired = merge_tracks(a, b, "train")
    assert len(paired) == 1
    assert paired.iloc[0]["text_key"] == "i am happy!"
    assert paired.iloc[0]["text"] == "I am happy!"
    assert paired.iloc[0]["text_a"] == "I am HAPPY!"
    assert paired.iloc[0]["joy_present"] == 1
    assert paired.iloc[0]["joy_intensity"] == 3
    assert paired.iloc[0]["presence_intensity_consistent"]
    assert "disgust" not in paired


@pytest.mark.parametrize("other", [" hello", "hello ", "hello!", "hel  lo", "##URL##"])
def test_key_does_not_strip_rewrite_or_fuzzy_match(other):
    a, *_ = clean([row("a", "hello")])
    b, *_ = clean([row("b", other)], "b")
    assert merge_tracks(a, b, "train").empty


def test_collapse_identical_annotations_and_preserve_all_ids():
    a, stats, diagnostics, conflicts = clean(
        [
            row("eng_dev_track_c_00001", "HELLO", joy=1),
            row("eng_dev_track_a_00001", "hello", joy=1),
        ]
    )
    assert len(a) == 1
    assert stats["identical_annotation_copies_collapsed"] == 1
    assert a.iloc[0]["id"] == "eng_dev_track_a_00001"
    assert a.iloc[0]["source_ids"] == [
        "eng_dev_track_a_00001",
        "eng_dev_track_c_00001",
    ]
    assert not diagnostics and not conflicts


def test_conflicting_intensities_are_quarantined_not_averaged():
    b, stats, diagnostics, conflicts = clean(
        [
            row("b1", "I am sad.", sadness=1),
            row("b2", "I am sad.", sadness=2),
        ],
        "b",
    )
    assert b.empty
    assert conflicts == {"i am sad."}
    assert stats["conflicting_rows"] == 2
    assert len(diagnostics) == 2


def test_presence_intensity_disagreement_is_flagged_not_repaired():
    a, *_ = clean([row("a", "text", joy=1)])
    b, *_ = clean([row("b", "TEXT", joy=0)], "b")
    paired = merge_tracks(a, b, "train")
    assert len(paired) == 1
    assert not paired.iloc[0]["presence_intensity_consistent"]
    assert paired.iloc[0]["joy_present"] == 1
    assert paired.iloc[0]["joy_intensity"] == 0


def test_invalid_records_are_reported_and_input_is_unchanged():
    frame = pd.DataFrame(
        [
            row("a1", "   "),
            row("a2", "valid", anger=2),
            row("a3", None),
            row("a4", "missing label", joy=None),
            row("a5", "fine", fear=1),
        ]
    )
    original = frame.copy(deep=True)
    a, stats, diagnostics, _ = clean_track(frame, "a", "train")
    pd.testing.assert_frame_equal(frame, original)
    assert a["id"].tolist() == ["a5"]
    assert stats["invalid_rows"] == 4
    assert len(diagnostics) == 4


def test_duplicate_source_id_fails():
    with pytest.raises(ValueError, match="source IDs are not unique"):
        clean([row("same", "one"), row("same", "two")])


def test_mixed_type_invalid_annotation_can_be_audited_and_quarantined():
    frame = pd.DataFrame([row("a1", "one", joy="invalid"), row("a2", "two")])
    assert audit(frame)["label_values"]["joy"] == [0, "invalid"]
    cleaned, stats, diagnostics, _ = clean_track(frame, "a", "train")
    assert cleaned["id"].tolist() == ["a2"]
    assert stats["invalid_rows"] == 1
    assert diagnostics[0]["reason"] == "invalid_joy"


def test_all_conflicting_rows_can_produce_empty_paired_table():
    a, *_ = clean([row("a1", "text", joy=1), row("a2", "text", joy=0)])
    b, *_ = clean([row("b1", "text", joy=1)], "b")
    paired = merge_tracks(a, b, "train")
    assert paired.empty
    assert "joy_intensity" in paired
    assert "presence_intensity_consistent" in paired


def test_non_null_disgust_fails_instead_of_silently_discarding_annotations():
    with pytest.raises(ValueError, match="non-null disgust"):
        clean([{**row("a", "text"), "disgust": 1}])


def test_missing_required_column_fails():
    frame = pd.DataFrame([row("a", "text")]).drop(columns="fear")
    with pytest.raises(ValueError, match="missing columns"):
        clean_track(frame, "a", "train")


def test_merge_rejects_many_to_many():
    a, *_ = clean([row("a", "text")])
    b, *_ = clean([row("b", "text")], "b")
    with pytest.raises(pd.errors.MergeError):
        merge_tracks(pd.concat([a, a]), pd.concat([b, b]), "train")


def test_overlap_policy_includes_unmatched_evaluation_source_keys():
    frames = {
        "train": pd.DataFrame({"text_key": ["shared", "dev-only", "train"]}),
        "dev": pd.DataFrame({"text_key": ["shared", "dev-only"]}),
        "test": pd.DataFrame({"text_key": ["test"]}),
    }
    source_keys = {
        "train": {"shared", "dev-only", "train"},
        "dev": {"shared", "dev-only"},
        "test": {"shared", "test"},
    }
    counts, diagnostics = exclude_cross_split_overlap(frames, source_keys)
    assert counts == {"train": 2, "dev": 1, "test": 0}
    assert frames["train"]["text_key"].tolist() == ["train"]
    assert frames["dev"]["text_key"].tolist() == ["dev-only"]
    assert len(diagnostics) == 3


def test_pipeline_is_reproducible_and_preserves_raw_files(tmp_path):
    raw, output, analysis = (
        tmp_path / name for name in ("raw", "processed", "analysis")
    )
    for track in ("a", "b"):
        directory = raw / f"track_{track}" / "eng"
        directory.mkdir(parents=True)
        for split in SPLITS:
            text = f"{split} text" if track == "a" else f"{split} TEXT"
            rows = [row(f"{track}-{split}", text, joy=1 if track == "a" else 2)]
            if track == "a" and split == "train":
                rows.append(row("a-unmatched", "only a"))
            pd.DataFrame(rows).to_parquet(directory / f"{split}-00000-of-00001.parquet")
    before = {
        p: hashlib.sha256(p.read_bytes()).hexdigest() for p in raw.rglob("*.parquet")
    }
    first = prepare(raw, output, analysis)
    files = list(output.glob("*.parquet")) + list(analysis.iterdir())
    first_bytes = {p: p.read_bytes() for p in files}
    second = prepare(raw, output, analysis)
    assert first == second
    assert all(p.read_bytes() == content for p, content in first_bytes.items())
    assert all(
        hashlib.sha256(p.read_bytes()).hexdigest() == sha for p, sha in before.items()
    )
    assert all(first["splits"][split]["final_rows"] == 1 for split in SPLITS)
    assert first["unmatched_records_count"] == 1
    summary = pd.read_csv(analysis / "summary.csv").set_index("metric")["value"]
    assert int(summary["schema_version"]) == 1
    unmatched = pd.read_csv(analysis / "unmatched_records.csv")
    assert unmatched.iloc[0]["id"] == "a-unmatched"
    assert unmatched.iloc[0]["source_ids"] == "a-unmatched"
    assert not list(analysis.glob("*.json*"))
    empty = pd.read_csv(analysis / "presence_intensity_mismatches.csv")
    assert empty.empty
    assert "joy_intensity" in empty


def test_diagnostic_csv_preserves_special_characters_and_source_ids(tmp_path):
    path = tmp_path / "diagnostic.csv"
    text = 'Hello, "world"!\nПривет 👋'
    write_diagnostic_csv(
        [{"text": text, "source_ids": ["a-1", "a-2"]}],
        ["text", "source_ids"],
        path,
    )
    frame = pd.read_csv(path)
    assert frame.iloc[0]["text"] == text
    assert frame.iloc[0]["source_ids"] == "a-1|a-2"


def test_output_cannot_overwrite_raw_directory(tmp_path):
    raw = tmp_path / "raw"
    with pytest.raises(ValueError, match="must not overlap"):
        prepare(raw, raw, tmp_path / "analysis")
