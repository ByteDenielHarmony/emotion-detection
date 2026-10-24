"""Shared mechanics for independently prepared labeled reference tracks A and B.

Remove invalid training rows and lower-priority cross-split occurrences. Retain
valid within-split duplicates, without majority voting or cross-track matching.
Track-specific target domains are explicit; missing annotations are never filled.
"""

import pandas as pd

from emotion_detection.datasets.reference_types import (
    ReferenceDataset,
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import SIX_LABELS, SPLITS

ISSUE_COLUMNS = ("split", "source_row", "id", "text", "emotion", "value", "reason")
EXCLUSION_COLUMNS = ("split", "source_row", "id", "text", "text_key", "reason")
OVERLAP_COLUMNS = (*EXCLUSION_COLUMNS, "higher_priority_splits")
PREPARATION_VARIANTS = {
    "a": "official-a-leakage-controlled-v1",
    "b": "official-b-leakage-controlled-v1",
}
TARGET_VALUES = {"a": (0, 1), "b": (0, 1, 2, 3)}


def prepare_labeled_track(frames, *, track, label_names, language, source_metadata):
    allowed_values = TARGET_VALUES[track]
    annotation_reason = (
        "missing_or_invalid_binary_annotation"
        if track == "a"
        else "missing_or_invalid_ordinal_annotation"
    )
    names = tuple(label_names)
    if (
        not names
        or len(set(names)) != len(names)
        or not set(names).issubset(SIX_LABELS)
    ):
        raise ReferencePreparationError("Declare a nonempty, unique emotion schema")
    if set(frames) != set(SPLITS):
        raise ReferencePreparationError(
            f"Track {track.upper()} requires explicit train, dev and test inputs"
        )
    issues, exclusions, duplicates, overlaps = [], [], [], []
    validated, counts = {}, {}
    invalid_evaluation = False
    for split in SPLITS:
        frame = frames[split].copy(deep=True).reset_index(drop=True)
        missing = {"id", "text", *names} - set(frame.columns)
        if not frame.columns.is_unique or missing:
            raise ReferencePreparationError(
                f"{split}: nonunique columns or missing fields {sorted(missing)}"
            )
        if frame.empty:
            raise ReferencePreparationError(
                f"{split}: reference input must not be empty"
            )
        for emotion in set(SIX_LABELS) - set(names):
            if (
                emotion in frame
                and frame[emotion].map(lambda v: not pd.isna(v) and str(v) != "").any()
            ):
                raise ReferencePreparationError(
                    f"{split}: annotated {emotion} is outside the declared schema"
                )
        valid = pd.Series(True, index=frame.index)
        duplicate_ids = frame["id"].duplicated(keep=False)
        numeric = {name: pd.to_numeric(frame[name], errors="coerce") for name in names}
        for index, row in frame.iterrows():
            problems = []
            if not isinstance(row["id"], str) or not row["id"].strip():
                problems.append(("", row["id"], "invalid_id"))
            elif duplicate_ids.loc[index]:
                problems.append(("", row["id"], "duplicate_id"))
            if not isinstance(row["text"], str) or not row["text"].strip():
                problems.append(("", row["text"], "missing_or_blank_text"))
            for name in names:
                value = numeric[name].loc[index]
                if pd.isna(value) or value not in allowed_values:
                    problems.append((name, row[name], annotation_reason))
            for emotion, value, reason in problems:
                issues.append(
                    {
                        "split": split,
                        "source_row": index,
                        "id": row["id"],
                        "text": row["text"],
                        "emotion": emotion,
                        "value": value,
                        "reason": reason,
                    }
                )
            if problems:
                valid.loc[index] = False
                if split == "train":
                    exclusions.append(
                        {
                            "split": split,
                            "source_row": index,
                            "id": row["id"],
                            "text": row["text"],
                            "text_key": row["text"].lower()
                            if isinstance(row["text"], str)
                            else "",
                            "reason": "|".join(sorted({p[2] for p in problems})),
                        }
                    )
                else:
                    invalid_evaluation = True
        # Detect annotation disagreements without choosing which annotator is right.
        with_text = frame[
            frame["text"].map(lambda t: isinstance(t, str) and bool(t.strip()))
        ].copy()
        with_text["text_key"] = with_text["text"].map(str.lower)
        duplicate_groups = conflict_groups = 0
        for key, group in with_text.groupby("text_key", sort=False):
            if len(group) < 2:
                continue
            duplicate_groups += 1
            conflict = len(group[list(names)].drop_duplicates()) > 1
            conflict_groups += int(conflict)
            for index, row in group.iterrows():
                duplicates.append(
                    {
                        "split": split,
                        "source_row": index,
                        "id": row["id"],
                        "text": row["text"],
                        "text_key": key,
                        "group_size": len(group),
                        "conflicting_annotations": conflict,
                        **{name: row[name] for name in names},
                    }
                )
        usable = frame.loc[valid, ["id", "text", *names]].copy()
        for name in names:
            usable[name] = numeric[name].loc[valid].astype("int64")
        usable["source_row"] = usable.index
        usable["source_language"] = language
        usable["source_split"] = split
        validated[split] = usable
        counts[split] = {
            "input_rows": len(frame),
            "invalid_rows": int((~valid).sum()),
            "duplicate_text_groups": duplicate_groups,
            "conflicting_text_groups": conflict_groups,
        }

    def diagnostic_frames():
        return {
            "annotation_issues": pd.DataFrame(issues, columns=ISSUE_COLUMNS),
            "exclusions": pd.DataFrame(exclusions, columns=EXCLUSION_COLUMNS),
            "duplicates": pd.DataFrame(
                duplicates,
                columns=[
                    "split",
                    "source_row",
                    "id",
                    "text",
                    "text_key",
                    "group_size",
                    "conflicting_annotations",
                    *names,
                ],
            ),
            "overlaps": pd.DataFrame(overlaps, columns=OVERLAP_COLUMNS),
        }

    if invalid_evaluation:
        raise ReferencePreparationError(
            "Invalid dev/test records or annotations. Evaluation examples were not silently removed or relabeled; inspect annotation_issues.csv.",
            diagnostic_frames(),
        )
    higher, prepared = {}, {}
    for split in reversed(SPLITS):
        frame = validated[split]
        remove = []
        for index, row in frame.iterrows():
            key = row["text"].lower()
            matching = [
                other
                for other, (ids, keys) in higher.items()
                if row["id"] in ids or key in keys
            ]
            if not matching:
                continue
            reasons = []
            if any(row["id"] in higher[other][0] for other in matching):
                reasons.append("cross_split_id")
            if any(key in higher[other][1] for other in matching):
                reasons.append("cross_split_text")
            record = {
                "split": split,
                "source_row": row["source_row"],
                "id": row["id"],
                "text": row["text"],
                "text_key": key,
                "reason": "|".join(reasons),
            }
            exclusions.append(record)
            overlaps.append({**record, "higher_priority_splits": "|".join(matching)})
            remove.append(index)
        higher[split] = (set(frame["id"]), set(frame["text"].map(str.lower)))
        prepared[split] = frame.drop(index=remove).reset_index(drop=True)
        counts[split]["overlap_rows_removed"] = len(remove)
        counts[split]["output_rows"] = len(prepared[split])
        if prepared[split].empty:
            raise ReferencePreparationError(
                f"{split}: no usable records after preparation", diagnostic_frames()
            )
    metadata = {
        "schema_version": 1,
        "track": track,
        "language": language,
        "label_names": list(names),
        "preparation_variant": PREPARATION_VARIANTS[track],
        "policies": {
            "text_identity": "Python str.lower() for diagnostics only; model text unchanged",
            "split_priority": ["test", "dev", "train"],
            "within_split_duplicates": "retain valid observations, including contradictory targets",
            "invalid_train": "exclude with diagnostics",
            "invalid_evaluation": "fail explicitly",
            "annotations": (
                "own Track A labels only; no A/B intersection or imputation"
                if track == "a"
                else "own Track B ordinal intensities 0-3 only; no A/B intersection, binarization or imputation"
            ),
        },
        "source_manifest": source_metadata,
        "split_counts": counts,
        "test_ids_and_order_preserved": prepared["test"]["id"].tolist()
        == frames["test"]["id"].tolist(),
        "original_split_membership_changed": any(
            value["invalid_rows"] or value["overlap_rows_removed"]
            for value in counts.values()
        ),
        "prepared_fingerprints": {
            split: prepared_fingerprint(prepared[split]) for split in SPLITS
        },
        "references": [
            "https://aclanthology.org/2025.semeval-1.327/",
            "https://arxiv.org/abs/2502.11926",
        ],
    }
    if track == "b":
        metadata["target_values"] = list(allowed_values)
    return ReferenceDataset(
        train=prepared["train"],
        dev=prepared["dev"],
        test=prepared["test"],
        label_names=names,
        metadata=metadata,
    ), diagnostic_frames()
