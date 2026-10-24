"""Pinned, explicit reference inputs, independent of the joint A+B snapshots.

The official post-evaluation repository revision below contains labeled English
test data. Later revisions change English membership and include an unlabeled
Track A test file; they must not be substituted silently. Source description:
https://github.com/emotion-analysis-project/SemEval2025-Task11
"""

import re
from dataclasses import asdict, dataclass
from pathlib import PurePosixPath

SPLITS = ("train", "dev", "test")
ENGLISH_LABELS = ("anger", "fear", "joy", "sadness", "surprise")
SIX_LABELS = ("anger", "disgust", "fear", "joy", "sadness", "surprise")
OFFICIAL_REPOSITORY = "emotion-analysis-project/SemEval2025-Task11"
OFFICIAL_REVISION = "44ddc04d0fcf2ee90c5e4caf531d4d0fa36b755a"
DEFAULT_RELEASE = "semeval2025-posteval-44ddc04d"


@dataclass(frozen=True)
class SourceSpec:
    """An acquisition record, not a prepared dataset or a training example."""

    track: str
    language: str
    split: str
    release: str
    repository: str
    revision: str
    url: str
    relative_path: str
    file_format: str
    sha256: str
    label_names: tuple[str, ...]

    def __post_init__(self):
        path = PurePosixPath(self.relative_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in self.relative_path
            or not path.parts
            or ":" in self.relative_path
        ):
            raise ValueError("Source destination must be a safe relative path")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.release):
            raise ValueError("Release must be a simple directory name")
        if self.track not in {"a", "b", "c"} or self.split not in SPLITS:
            raise ValueError("Unsupported source track or split")
        if not re.fullmatch(r"[a-z]{3,4}", self.language):
            raise ValueError("Source language must be an explicit language code")
        if not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(
                "Source revision must be a pinned commit SHA, not a moving branch"
            )
        if self.file_format not in {"csv", "parquet"}:
            raise ValueError("Unsupported source format")
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError(
                "Source SHA-256 must contain 64 lowercase hexadecimal digits"
            )
        if (
            not self.label_names
            or len(set(self.label_names)) != len(self.label_names)
            or not set(self.label_names).issubset(SIX_LABELS)
        ):
            raise ValueError("Source must declare an ordered, unique emotion schema")
        if (
            not self.revision
            or not self.repository
            or not self.url.startswith("https://")
        ):
            raise ValueError("Source requires a revision, repository and HTTPS URL")

    @property
    def label(self):
        return f"Track {self.track.upper()} / {self.language} / {self.split}"

    def metadata(self):
        record = asdict(self)
        record["label_names"] = list(self.label_names)
        return record


# Availability is explicit. Other languages/tracks need independently verified
# files, hashes and schemas before registration; no inferred URLs are attempted.
_HASHES = {
    "eng": (
        "234dfe6ccb34d91d36c9cae1d7f4b867e26a1abbe38e5ac6f40ae3dd5f83d608",
        "9e123425f99228d647d9051e78e6b469c389170b975e4519d9645462be7b265f",
        "422323aaa202a2138aa5532373209204cbd2c8c73228a0c5cd84e4ca0cae74e7",
    ),
    "deu": (
        "e0006e837a74afcc46f12c877442e4823a0de6a396afa37b7949487e2940b295",
        "cd0371398dbc7a08075d8cbabbf253b802975100d536106ee5ed6da4c50ffb9c",
        "40d5314ab90d20927a20d713db6231af137be21e9c29562005607d0939e95ecf",
    ),
}
_SCHEMAS = {"eng": ENGLISH_LABELS, "deu": SIX_LABELS}


def get_reference_sources(track="a", language="eng", release=DEFAULT_RELEASE):
    if track != "a":
        raise ValueError(
            "Reference data preparation currently implements Track A only; B and C are not implemented"
        )
    if release != DEFAULT_RELEASE:
        raise ValueError(f"Unsupported release: {release}")
    if language not in _HASHES:
        raise ValueError(
            f"Unsupported Track A language: {language}; available: {', '.join(_HASHES)}"
        )
    labels = _SCHEMAS[language]
    return tuple(
        SourceSpec(
            track=track,
            language=language,
            split=split,
            release=release,
            repository=OFFICIAL_REPOSITORY,
            revision=OFFICIAL_REVISION,
            url=(
                f"https://raw.githubusercontent.com/{OFFICIAL_REPOSITORY}/{OFFICIAL_REVISION}/"
                f"task-dataset/semeval-2025-task11-dataset/track_a/{split}/{language}.csv"
            ),
            relative_path=f"track_a/{language}/{split}.csv",
            file_format="csv",
            sha256=digest,
            label_names=labels,
        )
        for split, digest in zip(SPLITS, _HASHES[language], strict=True)
    )
