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
AFRIKAANS_LABELS = ("anger", "disgust", "fear", "joy", "sadness")
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


@dataclass(frozen=True)
class TrackALanguageSpec:
    """One language's schema and its explicitly required Track A source hashes."""

    label_names: tuple[str, ...]
    train_sha256: str
    dev_sha256: str
    test_sha256: str

    def __post_init__(self):
        if (
            not isinstance(self.label_names, tuple)
            or not self.label_names
            or len(set(self.label_names)) != len(self.label_names)
            or not set(self.label_names).issubset(SIX_LABELS)
        ):
            raise ValueError(
                "Language schema must be an ordered tuple of unique emotions"
            )
        for split in SPLITS:
            digest = getattr(self, f"{split}_sha256")
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"Invalid SHA-256 for Track A {split}")


# One registry owns both availability and label schemas: 28 languages, 84 files.
# Each entry declares labels, then train/dev/test hashes using the field order
# of TrackALanguageSpec. All three headers and binary annotations
# were checked for every registered language. Track C-only languages (ind, jav,
# xho, zul) are not Track A inputs. Extra upstream files such as
# train/ron_track1_test.csv are not part of the selected split manifest.
# Importing this manifest does not download any file.
_TRACK_A_LANGUAGES = {
    "afr": TrackALanguageSpec(
        AFRIKAANS_LABELS,
        "3bb4cbffa42e481eb809027a52a90db247273f0a8b267591651fb7acea0a047a",
        "d8b2241e648d313c13cc32053c7c2bde855992d44466b8522ad9dfd9698e3948",
        "43fa1237196567f9c845570ef57816bb46c9c15102d3b923d3edd0fe4dd03219",
    ),
    "amh": TrackALanguageSpec(
        SIX_LABELS,
        "6cef364f1eaffd903f46df1962d4efc64855ec23e41d7af3cceecaaee66f21e8",
        "5c49f4b14bd8d75c1c2a6b218fc15bcc3080359e1ef935ec0cc7d3d99e921969",
        "0510394dec6c40f618f2beafcf89717805915f3cbf6f58b67d8e6583de5f1ac3",
    ),
    "arq": TrackALanguageSpec(
        SIX_LABELS,
        "3fea52dd0318abb7db758b69cb4715308e536d4f50cfea39a644cd8f9e0aaebe",
        "40e326fd627b1fc6ec512961532fbca74eb02b90cce3b3d6f5b806f7eb5fbbba",
        "63ae0b8cce5a0c56fbe1fd50c9cc5a9565bcbb6bb2ec4eb084e0014760e44eb8",
    ),
    "ary": TrackALanguageSpec(
        SIX_LABELS,
        "5ec17fde01473572678839258f9ff4c1d7c848187b25883327c11a312e822d8a",
        "e70286e78494ac194ab4842fa0e62818f3c51c972d6c18b207f1a3080fe0115b",
        "4367875f77ef00cdd72ed7720aaa231d41bc95f05b6a60e702b68563a76d7a73",
    ),
    "chn": TrackALanguageSpec(
        SIX_LABELS,
        "0c8be88f0c1d1e9c6b1e86e27a18ea1f868ed770547c7828c6ccd51cea872b72",
        "d896f2e9138dacfed13c944b55ee7db5a55f43c63323a27d183d90f00d0a85ab",
        "b3200411e207b6b5848462cffbd450b73ceed879f6c43761624377b533ae8cbd",
    ),
    "deu": TrackALanguageSpec(
        SIX_LABELS,
        "e0006e837a74afcc46f12c877442e4823a0de6a396afa37b7949487e2940b295",
        "cd0371398dbc7a08075d8cbabbf253b802975100d536106ee5ed6da4c50ffb9c",
        "40d5314ab90d20927a20d713db6231af137be21e9c29562005607d0939e95ecf",
    ),
    "eng": TrackALanguageSpec(
        ENGLISH_LABELS,
        "234dfe6ccb34d91d36c9cae1d7f4b867e26a1abbe38e5ac6f40ae3dd5f83d608",
        "9e123425f99228d647d9051e78e6b469c389170b975e4519d9645462be7b265f",
        "422323aaa202a2138aa5532373209204cbd2c8c73228a0c5cd84e4ca0cae74e7",
    ),
    "esp": TrackALanguageSpec(
        SIX_LABELS,
        "136e5a49e3fd0f270e40985b66c7c804cca649888a606ce47d0536082ea412d7",
        "3bd2d0cabe03fcd09cc1fc4b39b2e9b554f5170fc7ac5415449bc4f55b8d0b41",
        "e82f7295d0123fd2bfa420671005398a1305f7303c1719c65582a03d8750bd48",
    ),
    "hau": TrackALanguageSpec(
        SIX_LABELS,
        "a1cc66f557e890dcab22ee907b02e32f53ec7c9d9f4caf15bcf2c038a5a25b0c",
        "8ca435a964617e26cb46d9bf82b145120c0dd9ebbd66cf7a2a4f9f06bf241d0e",
        "976602640f104355f4649940bc650bffa67666a44f998321d93e8c74150adc35",
    ),
    "hin": TrackALanguageSpec(
        SIX_LABELS,
        "7f2f9c085ce12c2736ad13d02c30acf1a161c240f08a88285d819bbe52ba2cc6",
        "c44cd4828017e910795735662ae04843d3bf3315eedecb6c6ea7e73717d73fbc",
        "2a89ed3bcec51afce771005c9b8977f0168f32eec5c45bb696552938d952b575",
    ),
    "ibo": TrackALanguageSpec(
        SIX_LABELS,
        "b7c8d10d54e25aafa0061eb1321a887b51b3ad4c35151e9147514473d738eb92",
        "336d1cbd154f491217f99cb8f67e41a098a5e00c6c320add3e29e573dd269cc8",
        "32b302347a4adb4d3b965895c9d0160135ccbfd29a40d693f4310f578e33b015",
    ),
    "kin": TrackALanguageSpec(
        SIX_LABELS,
        "d6a4d80d0b8e65f10e161d8e12e2219a0afb2049370db46dd3e3a4830cf7e99b",
        "17f6bd0a52c1ba7c88d6db7ad962b8eaf973030da853830bae37d83b38104746",
        "4eb2027475fee4865f460082444ad9aefeef25569ae501b4fdf0ba88bde9b7f0",
    ),
    "mar": TrackALanguageSpec(
        SIX_LABELS,
        "9c3c7c1a6eadfe9f8a56e8b311d1ffefc6fe846a8be628bdf692c1389da0fe14",
        "18ed4e78509d7e4606ed22e1031c31015c4372351852dbe8f884342f978a5681",
        "cb6889deff20766708d20c717bd36b840760da7b44dfdb7e877ca577f0f2a842",
    ),
    "orm": TrackALanguageSpec(
        SIX_LABELS,
        "025291ec11e5f7e0f9ed7b0169162a65e4752413030c374e76b6172cceeda306",
        "975da911137c29f946b04a82a618d91146a913f5fc50c2edf5998c8fcd56fb3b",
        "98d346b69930d031c8d756b7b02e04ac28e4fed31fa2207bb90eb2bf1d16967d",
    ),
    "pcm": TrackALanguageSpec(
        SIX_LABELS,
        "ece02fb4215776ea592884904bb29f0560badc00d9e411180b80324e698d4059",
        "9d5df6446f3bff276ab9f0ed92c61d623feb4a86fb01edbdc473a631c5309550",
        "b3fce26de12a264d00284ee2e9731a092c24d059d63c4202f7b12b7692ef79e5",
    ),
    "ptbr": TrackALanguageSpec(
        SIX_LABELS,
        "ca24d3eab10afef0682e778a735e7c21611f3d027bd88341d3f7aa3d9aee6ae0",
        "a462854eb28e8ce4c81ed9c1f3aff6d2bab405c9202b29466c7db59468d22d47",
        "ce5e41b4ad762815ed204fc08aa505c5288aa37faf295671a07628124e63f2f2",
    ),
    "ptmz": TrackALanguageSpec(
        SIX_LABELS,
        "86592aaf5a32221dbe8907109ef41fbb94a43786e4eacd7826f22d2590329c9b",
        "e82586ce84099afa35edf4e8f74d925a4dcf54028468b889902c4146699b2f21",
        "9961007fbefb3beb3cc4437b42a4ace7a2839651f9b1fd409d616f8cb1f36da2",
    ),
    "ron": TrackALanguageSpec(
        SIX_LABELS,
        "b3384f9c163d5a30f38c70db6edd5d60fa7d510a823d674979138a6fa503d463",
        "18096929f561e85e818ca49b29683171a911bd9fe539fd17fc0c42a0734a722e",
        "76b6ff406d5cd69387f7f609c9be0cd2f22df51764c2589364b6a551cb6a9eeb",
    ),
    "rus": TrackALanguageSpec(
        SIX_LABELS,
        "d0591c3bd83189fe3b446146053981b3c30c84c6c60e550453f8fea87826d3a4",
        "228b36ce83a18b17c4e01eab241b2d3ac35a2fea02527046e06817e8b52e47bd",
        "bd160a11e0401271af0d13e53d2ece695d892d0c20763640412185817b4c37ed",
    ),
    "som": TrackALanguageSpec(
        SIX_LABELS,
        "82541be0fd2dbe2ef5dc0005afc28781dcec270d50959268baa2c38d3aaa3217",
        "dcb343bf13f5044b5adf06f71f236abb893ad32d0b0851d897238b46cf2a6315",
        "490b6079764d94fc28e60e4a70e8069cb57651271d8693320bf2e764c10742fe",
    ),
    "sun": TrackALanguageSpec(
        SIX_LABELS,
        "0a4cad9079807a3e586b5c1f956089219cd1aa0ee073c8c9143152c25c53c279",
        "eca7f6ec04d733a1e7389c28a01666c8e8592e4ecb112d55285ffcbb2a7918f0",
        "92171cb38ae918b9d67a0ec5dd1d04aa0cf456e82009795fa0d6f67139a6e1c7",
    ),
    "swa": TrackALanguageSpec(
        SIX_LABELS,
        "888f0bed367826b7a15b4f485604c553a6940a9699b3917119f978d04137f1ab",
        "2fa9c8bddda6c0eb72d06b378ffb38734d17ba056404406792cc07d4bce4c76c",
        "a828d338e9eeb7222f6f9f593914c3013a9c57b78583a47c28a133f12dbe8de3",
    ),
    "swe": TrackALanguageSpec(
        SIX_LABELS,
        "32cf50cdc089fa542fc73fd50998348c9d8a0feb76c710d33fa783942b483234",
        "2e295db37076874866cdc4fea11e6bfe7d9893855ae4a8be58ee0c8cd7531ceb",
        "3878d0fb8213823b2b40c7593710e790bcfff30a1b562e9e2662fbbda6b66a55",
    ),
    "tat": TrackALanguageSpec(
        SIX_LABELS,
        "3114064d6b477d22cb0377f4b5597cc04f34fd2660cc62666af1225556d5002a",
        "e6ed958b84874faba6629afa79ad3a16bd190d5b759b9a81217547cc20ec4d4e",
        "db329d6e620d643d1640ace9352a81854694440e8829f0820a7fe0db9024b997",
    ),
    "tir": TrackALanguageSpec(
        SIX_LABELS,
        "50813eb3ec14177f63fa9c2d7943df447d1047f957c89efdef9096cb1701bb74",
        "d0e2f3b36252645c4a05f8d4594033eed9cd40d920c74063ffd19b5787ada53b",
        "6da94646ab206ee07b55fc12f85e9f87c4ade3aa11eb8d358a4afbeed6eaa1de",
    ),
    "ukr": TrackALanguageSpec(
        SIX_LABELS,
        "433fc4430cf670c9816a4615df65911c552232843939968836915c762e0c119a",
        "4a6c5ddc559c5fcb19085832207dc82fd7665ff71e5b4e7a1529d591ed66935e",
        "092a021ff0dedebd28cfdc685c5c3ccbbc5cd4a84256cb9ade6a434e608dd4ef",
    ),
    "vmw": TrackALanguageSpec(
        SIX_LABELS,
        "bcdb57cfeccaa0daccd35fc43800673ad0bfc28eced804cee597970d0a046e03",
        "fc7f1bc37654e2c58bc53487527ebb6634053dd7e15b98a9e6c40951ad859422",
        "989fc9f5dc4681b08c323a3e535553b05787e266b0c7bd2492706b6973399b74",
    ),
    "yor": TrackALanguageSpec(
        SIX_LABELS,
        "cef3cbab720436bdabc3a0659ad6a2361defd3e3d71ad687e4853dfd92afa448",
        "7ba48a939dd97365e77b04f83e9de33c3177da48bfee65018eaafad5c1b7b80c",
        "c81b3568ad3c5174432d96b2de196da29db0aa61602192764d13ed6465274be2",
    ),
}


def available_reference_languages(track="a", release=DEFAULT_RELEASE):
    """List registered languages without file I/O, downloads or preparation."""
    if track != "a":
        raise ValueError(
            "Reference data preparation currently implements Track A only; B and C are not implemented"
        )
    if release != DEFAULT_RELEASE:
        raise ValueError(f"Unsupported release: {release}")
    return tuple(sorted(_TRACK_A_LANGUAGES))


def get_reference_sources(track="a", language="eng", release=DEFAULT_RELEASE):
    languages = available_reference_languages(track, release)
    if language not in languages:
        raise ValueError(
            f"Unsupported Track A language: {language}; available: {', '.join(languages)}"
        )
    specification = _TRACK_A_LANGUAGES[language]
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
            sha256=getattr(specification, f"{split}_sha256"),
            label_names=specification.label_names,
        )
        for split in SPLITS
    )
