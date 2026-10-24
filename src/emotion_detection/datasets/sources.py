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
        _validate_language_spec(self, "A")


@dataclass(frozen=True)
class TrackBLanguageSpec:
    """An independently verified Track B schema and all three source hashes."""

    label_names: tuple[str, ...]
    train_sha256: str
    dev_sha256: str
    test_sha256: str

    def __post_init__(self):
        _validate_language_spec(self, "B")


def _validate_language_spec(specification, track):
    names = specification.label_names
    if (
        not isinstance(names, tuple)
        or not names
        or len(set(names)) != len(names)
        or not set(names).issubset(SIX_LABELS)
    ):
        raise ValueError("Language schema must be an ordered tuple of unique emotions")
    for split in SPLITS:
        digest = getattr(specification, f"{split}_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Invalid SHA-256 for Track {track} {split}")


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


# Track B has ten complete language triplets at the pinned revision. All headers
# and targets were inspected independently of Track A. Romanian has dev/test but
# no train/ron.csv here: do not advertise it or substitute its Track A training.
_TRACK_B_LANGUAGES = {
    "amh": TrackBLanguageSpec(
        SIX_LABELS,
        "8708c7a40d2d5ac767c709a48a7ed55c43579c3a3016d773fc80fa58b4a162fd",
        "af1841e8f246127be23a39a51fdaf6e3b39f6c8ce74124b0de190efc3669383d",
        "a8fbced28ba185e21b2103833a0a9d0717b437101ae28ca0b5dc597d6e561e86",
    ),
    "arq": TrackBLanguageSpec(
        SIX_LABELS,
        "c5ebb9466efcb78ef02041ffe0fedfdebc444c1b777215cdd9c2d49774228d06",
        "acd137d56c29706f74b6eeb5d7e1f78689a14c60d8b6e9bc8e8bb6770d62e068",
        "0c111238647c118ff92390f6d9a399e4edf97623b0bcfd26a35e8a81724e6c4e",
    ),
    "chn": TrackBLanguageSpec(
        SIX_LABELS,
        "f844de6c1da6f33ab2a0c5ee7e280683973a31818067807c2ecb87a9879f84cd",
        "10ee994ec515263f5a9498467c5871e83b68ba4b226b413afc326d46cf18c19d",
        "4e74e9a6c92f54d77c9a13a9d7f88799f9e5a6066682ad4f990cfa8c7cb0637a",
    ),
    "deu": TrackBLanguageSpec(
        SIX_LABELS,
        "c495c65a5bc21cbcbabf18108dd0c1211f725508fd759c3383a00c38ba8ec39c",
        "a04b12c27f6f70e944d33096401a50a04e0252a435b9d8f29bd626e6159d908a",
        "cc876202ecec5a6877eaf017f5fde4946efac0dcd6cc471c4c5103a865e36333",
    ),
    "eng": TrackBLanguageSpec(
        ENGLISH_LABELS,
        "55167973e06833c9cf511140b3b44b25b1b04c91a2a953aa2a677126e838a0d7",
        "3c2d7a27d5f9d9dbfbde29c098d173c995081f166796d88695cb0f2d9419f559",
        "5092cd688d2d9dee129c34cb63d4abafb73759d5115e5c0fb144347c95991d42",
    ),
    "esp": TrackBLanguageSpec(
        SIX_LABELS,
        "3d0d55c5afb5dff90cc581a4280d024d6c8b1bcea314cabf00923250e19f617d",
        "88991a80a86ee84f2689a0b61918a1e7b21964df60845bb3e24cf4429d226e38",
        "e9a47593fffd0dac97736993d181f23364b81b8773d4eadb8f1750cc2f8886f1",
    ),
    "hau": TrackBLanguageSpec(
        SIX_LABELS,
        "13138c1b94e9b4559e0f60f89cfff46b0f4f779498ab24b39abe5cb2d1b617bd",
        "5e693e698713c52ab9262f0c43da388fe08a8a79e3b30fdbc7b9a0c7cd0f4c88",
        "99df4093c8bdb66bc93a4d6075a1ed38e3eab82932c56afbc3bcf4599779369d",
    ),
    "ptbr": TrackBLanguageSpec(
        SIX_LABELS,
        "6ade4b4bb0351f1a5efcf474aa0f71b5a5f68be09e53de57b5cfc8fe38902ab2",
        "6ed2ff8add43381dab7a17881c45f08c9a4cd19b66d0bd955ef0e2b196663100",
        "be8af0b237b89d76fcd8b8cd43b9c33c4b6389599e9a8e97526c70ca1d486906",
    ),
    "rus": TrackBLanguageSpec(
        SIX_LABELS,
        "6687afacd225ada937108635780be2dd21371c35af42d9f841a6aaae35a20ec7",
        "d2b04f3dfc16283a41f53e2698061ae64e417355d5ba978045ee748bf8d9d014",
        "7058100197efaef188cfb0ee429e9d8f434900be2a1572353dd0df07a4c16c8f",
    ),
    "ukr": TrackBLanguageSpec(
        SIX_LABELS,
        "92781503f1052c5cf00c29025f983397c174f3bde4f0fbb4ceb11d2559e7e780",
        "60849f09be000289b4c34389d314acfef9274d114c3f65f93c591b3244e2699a",
        "5b80a65488378feb051ded73bf76a53a0f3b9919c8fc6684b3d30867ca5cbbdf",
    ),
}


def available_reference_languages(track="a", release=DEFAULT_RELEASE):
    """List registered languages without I/O; C lists targets, not protocol readiness."""
    if track == "c":
        from emotion_detection.datasets.sources_track_c import available_track_c_targets

        return available_track_c_targets(release)
    if track not in {"a", "b"}:
        raise ValueError(
            f"Unsupported reference track: {track}"
        )
    if release != DEFAULT_RELEASE:
        raise ValueError(f"Unsupported release: {release}")
    registry = _TRACK_A_LANGUAGES if track == "a" else _TRACK_B_LANGUAGES
    return tuple(sorted(registry))


def get_reference_sources(track="a", language="eng", release=DEFAULT_RELEASE):
    """Pinned inputs: A/B triplets or C target dev/test (not source selections)."""
    if track == "c":
        from emotion_detection.datasets.sources_track_c import get_track_c_targets

        return get_track_c_targets(language, release)
    languages = available_reference_languages(track, release)
    if track == "b" and language == "ron":
        raise ValueError(
            "Unsupported Track B language: ron; the pinned release has no Track B train/ron.csv. "
            "Track A training data will not be substituted."
        )
    if language not in languages:
        raise ValueError(
            f"Unsupported Track {track.upper()} language: {language}; available: {', '.join(languages)}"
        )
    registry = _TRACK_A_LANGUAGES if track == "a" else _TRACK_B_LANGUAGES
    specification = registry[language]
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
                f"task-dataset/semeval-2025-task11-dataset/track_{track}/{split}/{language}.csv"
            ),
            relative_path=f"track_{track}/{language}/{split}.csv",
            file_format="csv",
            sha256=getattr(specification, f"{split}_sha256"),
            label_names=specification.label_names,
        )
        for split in SPLITS
    )
