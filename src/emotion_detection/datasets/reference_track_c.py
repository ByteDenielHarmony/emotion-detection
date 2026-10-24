"""In-memory cross-lingual preparation; no acquisition, models or source tests."""

import pandas as pd

from emotion_detection.datasets.reference_types import (
    CrosslingualReferenceDataset,
    ReferencePreparationError,
    prepared_fingerprint,
)
from emotion_detection.datasets.sources import SIX_LABELS
from emotion_detection.datasets.sources_track_c import (
    DEFAULT_C_PROTOCOL,
    get_crosslingual_sources,
    track_c_source_languages,
)

C_ROLES = ("train", "source_dev", "target_dev", "test")
PREPARATION_VARIANT = "official-c-masked-leakage-controlled-v1"
C_POLICIES = {
    "selection": "explicit task-paper broad-family mapping, not BRIGHTER subgroups",
    "validation": "source_dev only; target_dev is separate and not used for selection",
    "source_test": "never acquired or read",
    "target_train": "never acquired or read",
    "unknown_annotations": "nullable values with known_<emotion> masks; never negatives",
    "overlaps": "target dev/test retained; remove overlapping source_dev then train",
    "identity": "(source_language, id); text.lower() across languages",
    "within_split_duplicates": "retain valid observations including conflicting targets",
}
C_DEVIATIONS = (
    "Explicit source-dev selection, missing-label masks and leakage-controlled exclusions; not an exact score reproduction",
)
MASK_COLUMNS = tuple(f"known_{name}" for name in SIX_LABELS)
C_COLUMNS = (
    "id",
    "text",
    *SIX_LABELS,
    *MASK_COLUMNS,
    "source_track",
    "source_language",
    "source_split",
    "source_row",
)
AUDIT_COLUMNS = (
    "role",
    "source_track",
    "source_language",
    "source_split",
    "source_row",
    "id",
    "text",
    "emotion",
    "value",
    "reason",
)


def source_key(source):
    return source.track, source.language, source.split


def source_role(source):
    if source.track == "c":
        return "target_dev" if source.split == "dev" else "test"
    return "train" if source.split == "train" else "source_dev"


def validate_c_sources(sources, language, release, protocol):
    expected = get_crosslingual_sources(language, release, protocol)
    identity = lambda s: (
        s.track,
        s.language,
        s.split,
        s.label_names,
        s.release,
        s.repository,
        s.revision,
        s.url,
        s.relative_path,
        s.file_format,
    )
    if [identity(s) for s in sources] != [identity(s) for s in expected]:
        raise ReferencePreparationError(
            "Track C requires exactly the declared other-language A train/dev and target C dev/test; "
            "target train and source test are forbidden"
        )


def _identities(frame):
    return set(zip(frame.source_language, frame.id, strict=True))


def _keys(frame):
    return set(frame.text.str.lower())


def prepare_track_c(frames, *, sources, language, release, protocol=DEFAULT_C_PROTOCOL):
    sources = tuple(sources)
    validate_c_sources(sources, language, release, protocol)
    if set(frames) != {source_key(s) for s in sources}:
        raise ReferencePreparationError(
            "Track C inputs do not match the selected source manifest"
        )
    issues, exclusions, duplicates, overlaps = [], [], [], []
    parts = {role: [] for role in C_ROLES}
    counts = {role: {"input_rows": 0, "invalid_rows": 0} for role in C_ROLES}
    invalid_evaluation = False

    def diagnostics():
        return {
            "annotation_issues": pd.DataFrame(issues, columns=AUDIT_COLUMNS),
            "exclusions": pd.DataFrame(exclusions, columns=AUDIT_COLUMNS),
            "duplicates": pd.DataFrame(
                duplicates,
                columns=[*AUDIT_COLUMNS, "conflicting_annotations", *SIX_LABELS],
            ),
            "overlaps": pd.DataFrame(
                overlaps, columns=[*AUDIT_COLUMNS, "higher_priority_roles"]
            ),
        }

    for source in sources:
        role = source_role(source)
        frame = frames[source_key(source)].copy(deep=True).reset_index(drop=True)
        if (
            frame.empty
            or not frame.columns.is_unique
            or not {"id", "text", *source.label_names}.issubset(frame.columns)
        ):
            raise ReferencePreparationError(
                f"{source.label}: empty input or invalid columns"
            )
        canonical = frame[["id", "text"]].copy()
        valid = pd.Series(True, index=frame.index)

        def record(
            index,
            emotion="",
            value="",
            reason="",
            *,
            role=role,
            source=source,
            frame=frame,
        ):
            return {
                "role": role,
                "source_track": source.track,
                "source_language": source.language,
                "source_split": source.split,
                "source_row": index,
                "id": frame.at[index, "id"],
                "text": frame.at[index, "text"],
                "emotion": emotion,
                "value": value,
                "reason": reason,
            }

        checks = {
            "invalid_id": ~frame.id.map(
                lambda v: isinstance(v, str) and bool(v.strip())
            ),
            "duplicate_id": frame.id.duplicated(keep=False),
            "missing_or_blank_text": ~frame.text.map(
                lambda v: isinstance(v, str) and bool(v.strip())
            ),
        }
        for reason, bad in checks.items():
            for index in frame.index[bad]:
                issues.append(record(index, reason=reason))
            valid &= ~bad
        for name in SIX_LABELS:
            if name not in source.label_names:
                if (
                    name in frame
                    and frame[name].map(lambda v: not pd.isna(v) and str(v) != "").any()
                ):
                    raise ReferencePreparationError(
                        f"{source.label}: annotated {name} outside native schema"
                    )
                canonical[name] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
                canonical[f"known_{name}"] = False
                continue
            missing = frame[name].map(
                lambda v: pd.isna(v) or (isinstance(v, str) and not v.strip())
            )
            numeric = pd.to_numeric(frame[name], errors="coerce")
            bad = ~missing & ~numeric.isin([0, 1])
            for index in frame.index[missing]:
                issues.append(
                    record(
                        index, name, frame.at[index, name], "missing_annotation_masked"
                    )
                )
            for index in frame.index[bad]:
                issues.append(
                    record(
                        index, name, frame.at[index, name], "invalid_binary_annotation"
                    )
                )
            valid &= ~bad
            canonical[name] = numeric.where(~missing & ~bad).astype("Int64")
            canonical[f"known_{name}"] = ~missing & ~bad
        # An unsupervised source row cannot contribute to fitting; never zero-fill.
        if role == "train":
            unobserved = ~canonical[list(MASK_COLUMNS)].any(axis=1)
            for index in frame.index[unobserved]:
                issues.append(record(index, reason="no_observed_annotations"))
            valid &= ~unobserved
        if not valid.all():
            if role == "train":
                for index in frame.index[~valid]:
                    exclusions.append(record(index, reason="invalid_training_record"))
            else:
                invalid_evaluation = True
        canonical["source_track"] = source.track
        canonical["source_language"] = source.language
        canonical["source_split"] = source.split
        canonical["source_row"] = frame.index
        usable = canonical.loc[valid, list(C_COLUMNS)].copy()
        parts[role].append(usable)
        counts[role]["input_rows"] += len(frame)
        counts[role]["invalid_rows"] += int((~valid).sum())

    if invalid_evaluation:
        raise ReferencePreparationError(
            "Invalid source-dev or target evaluation records; examples were not silently removed or relabeled",
            diagnostics(),
        )
    prepared = {role: pd.concat(parts[role], ignore_index=True) for role in C_ROLES}
    for role, frame in prepared.items():
        keys = frame.text.str.lower()
        repeated = frame.loc[keys.duplicated(keep=False)]
        for _, group in repeated.groupby(repeated.text.str.lower(), sort=False):
            conflict = len(group[list(SIX_LABELS)].drop_duplicates()) > 1
            for _, row in group.iterrows():
                duplicates.append(
                    {
                        **{
                            name: row[name]
                            for name in (
                                "source_track",
                                "source_language",
                                "source_split",
                                "source_row",
                                "id",
                                "text",
                                *SIX_LABELS,
                            )
                        },
                        "role": role,
                        "emotion": "",
                        "value": "",
                        "reason": "within_role_duplicate_text",
                        "conflicting_annotations": conflict,
                    }
                )
    # Evaluation partitions stay intact, including their mutual overlaps. Only
    # source roles lose lower-priority observations; IDs are language-namespaced.
    higher = {
        role: (_identities(prepared[role]), _keys(prepared[role]))
        for role in ("target_dev", "test")
    }
    for role in ("source_dev", "train"):
        frame = prepared[role]
        remove = pd.Series(False, index=frame.index)
        for other, (ids, keys) in higher.items():
            matching = pd.Series(
                list(zip(frame.source_language, frame.id, strict=True)),
                index=frame.index,
            ).isin(ids) | frame.text.str.lower().isin(keys)
            for index in frame.index[matching]:
                row = frame.loc[index]
                audit = {
                    name: row[name]
                    for name in (
                        "source_track",
                        "source_language",
                        "source_split",
                        "source_row",
                        "id",
                        "text",
                    )
                }
                audit.update(
                    role=role, emotion="", value="", reason="cross_role_id_or_text"
                )
                overlaps.append({**audit, "higher_priority_roles": other})
                if not remove.loc[index]:
                    exclusions.append(audit)
            remove |= matching
        higher[role] = (_identities(frame), _keys(frame))
        prepared[role] = frame.loc[~remove].reset_index(drop=True)
        counts[role]["overlap_rows_removed"] = int(remove.sum())
    for role, frame in prepared.items():
        if frame.empty or not frame[list(MASK_COLUMNS)].any().any():
            raise ReferencePreparationError(
                f"{role}: no usable labeled records after preparation", diagnostics()
            )
        counts[role].setdefault("overlap_rows_removed", 0)
        counts[role]["output_rows"] = len(frame)
        counts[role]["unknown_annotation_cells"] = int(
            frame[list(SIX_LABELS)].isna().sum().sum()
        )
    metadata = {
        "schema_version": 1,
        "track": "c",
        "language": language,
        "release": release,
        "protocol": protocol,
        "source_languages": list(track_c_source_languages(language, protocol)),
        "label_names": list(sources[-1].label_names),
        "model_label_names": list(SIX_LABELS),
        "preparation_variant": PREPARATION_VARIANT,
        "policies": dict(C_POLICIES),
        "protocol_deviations": list(C_DEVIATIONS),
        "source_manifest": [s.metadata() for s in sources],
        "split_counts": counts,
        "target_ids_and_order_preserved": all(
            prepared[role].id.tolist() == frames[("c", language, split)].id.tolist()
            for role, split in (("target_dev", "dev"), ("test", "test"))
        ),
        "target_dev_test_overlap_texts": len(
            _keys(prepared["target_dev"]) & _keys(prepared["test"])
        ),
        "target_dev_test_overlap_ids": len(
            _identities(prepared["target_dev"]) & _identities(prepared["test"])
        ),
        "prepared_fingerprints": {
            role: prepared_fingerprint(f) for role, f in prepared.items()
        },
        "references": [
            "https://aclanthology.org/2025.semeval-1.327/",
            "https://arxiv.org/abs/2502.11926v1",
        ],
    }
    return CrosslingualReferenceDataset(
        **prepared,
        label_names=tuple(metadata["label_names"]),
        model_label_names=SIX_LABELS,
        metadata=metadata,
    ), diagnostics()
