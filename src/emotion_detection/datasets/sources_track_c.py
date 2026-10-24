"""Pinned C target inputs and explicit source selections, not inferred families.

The task paper section 4.3 uses broad families (e.g. Indo-European), plus three
explicit exceptions. This is not interchangeable with BRIGHTER subgroup setups.
Source-dev selection, masking and leakage controls are project preparation choices,
not evidence of exact reproduction of the published scores.
"""

from dataclasses import dataclass

from emotion_detection.datasets.sources import (
    AFRIKAANS_LABELS,
    DEFAULT_RELEASE,
    ENGLISH_LABELS,
    OFFICIAL_REPOSITORY,
    OFFICIAL_REVISION,
    SIX_LABELS,
    SourceSpec,
    get_reference_sources,
)

DEFAULT_C_PROTOCOL = "task-paper-family-v1"


@dataclass(frozen=True)
class TrackCTargetSpec:
    label_names: tuple[str, ...]
    dev_sha256: str
    test_sha256: str

    def __post_init__(self):
        if not isinstance(self.label_names, tuple) or not self.label_names:
            raise ValueError("Target labels must be an ordered tuple")
        # Reuse the source record's schema and hash validation without file I/O.
        for split in ("dev", "test"):
            _target_source("eng", split, self, DEFAULT_RELEASE)


def _target_source(language, split, spec, release):
    return SourceSpec(
        track="c",
        language=language,
        split=split,
        release=release,
        repository=OFFICIAL_REPOSITORY,
        revision=OFFICIAL_REVISION,
        url=(
            f"https://raw.githubusercontent.com/{OFFICIAL_REPOSITORY}/{OFFICIAL_REVISION}/"
            f"task-dataset/semeval-2025-task11-dataset/track_c/{split}/{language}.csv"
        ),
        relative_path=f"track_c/{language}/{split}.csv",
        file_format="csv",
        sha256=getattr(spec, f"{split}_sha256"),
        label_names=spec.label_names,
    )


# All 64 files were checked against official Git blob identities and binary labels.
_TRACK_C_TARGETS = {
    "afr": TrackCTargetSpec(
        AFRIKAANS_LABELS,
        "1bdbe003a179558a07259579d4ff2b120f8d51adf6f0e386ab82ebd1fe5b558c",
        "50741add0463905d89dfc92963712b71e0865a3fa855a73cb8d08037953759ef",
    ),
    "amh": TrackCTargetSpec(
        SIX_LABELS,
        "904a875c33709806a9490fbec7c3155e2ed92592164516dfe9b8cfa7a4863f25",
        "53f2e85dd73d46d3f22204dda565c22dde1524a951b4bc57a44c923d2dbdf85d",
    ),
    "arq": TrackCTargetSpec(
        SIX_LABELS,
        "3b16ccd1c933cdf91fa7d8c93b9da6233206067dcefedac73f4398ed4a7c335b",
        "b2fd2a77ebb1e965f64c70a6d55e9431dc5200314b3b4d53744d9ba570bfdd6c",
    ),
    "ary": TrackCTargetSpec(
        SIX_LABELS,
        "7ce605da2ddf18073a26d96c6fd9e7f72efad6ff3ff2078f0bb87057e8c3eeb1",
        "3ff59c14a59bf0396246fe7cdb3548f956ddb74e723c4840667d42671975a9d6",
    ),
    "chn": TrackCTargetSpec(
        SIX_LABELS,
        "0077289a927fe82224a1ba7901d19563750adb926327e995ad054391b378966f",
        "280b3545282ca4abf2699f9334d584970fa24f9e700b0d6c31f3687b2eb4a94a",
    ),
    "deu": TrackCTargetSpec(
        SIX_LABELS,
        "25044b00e7b0870dd27adcf6dc2f03f63d48edb148895bdfc7a08b5ed807434f",
        "9a7eefb0ac73efff4f7898fab41cf9695fff885f1dd04bbcbc82000e2ea9ace7",
    ),
    "eng": TrackCTargetSpec(
        ENGLISH_LABELS,
        "d9a0c6f75c24af6b8f2ddb504e7935d6b40726d848b1e12a0af82d4d25bfda88",
        "53d4437bcbf6a60eef0b72b56040540b8233331fbaad4a963431fab739dca2a0",
    ),
    "esp": TrackCTargetSpec(
        SIX_LABELS,
        "cbc02c5c971eed44e80ee568be7b5d1f171ca8ed1fe3e08379fe7d6f52874a96",
        "8716c53bb67174d8a0453526062686465e155f2afa62ab3191c55acf72bef057",
    ),
    "hau": TrackCTargetSpec(
        SIX_LABELS,
        "12ad910118b795d42e5bc777c08f97eaaba42943df1e9adace9a03f75e31fe96",
        "d8b9bf68525caebb3e06aff0563636bc11c523121c742ebe0c5706b53b5d1871",
    ),
    "hin": TrackCTargetSpec(
        SIX_LABELS,
        "5d5d070062bc23aa8a92a156f53c31cd09dc99cfd77fd5304105981ce378ca68",
        "1746854c7956f3bb9e039781d62259040b377a79854dec5524746cc6b13fd075",
    ),
    "ibo": TrackCTargetSpec(
        SIX_LABELS,
        "a19db5405ae7eea8663f8242f3e189b9eb0fcde1222a28e97724c8456af73b1b",
        "ca4b9d8be177958dc91702d8144d15b1640f2a001aad055356c1d78d047a2cbc",
    ),
    "ind": TrackCTargetSpec(
        SIX_LABELS,
        "638f0c5d5e41412ade7e84a4d0ec4e0c296559f0280709adbb79cbb6dd1dd4bc",
        "b2c49bec32e09e8a923a9b4c394ee859b1602c8fd0d4437a6ded7755de207b85",
    ),
    "jav": TrackCTargetSpec(
        SIX_LABELS,
        "33f44c1ca79c76e0aaa2c2f6413cdf1329574c3dbdcc019e3bab9f748d072ea5",
        "de6bbd5430c68d0d37e7a8b104eb57cf7c96604e71794d5ea9501e793c76a7c4",
    ),
    "kin": TrackCTargetSpec(
        SIX_LABELS,
        "42b24054890af85bc6085686d45ebf95d78d2aa45da23fb796a614142d3265da",
        "1464dc1ea54af64696f24b95c102dfe30ba82d4100ac4fbe1a435cd9deab11ef",
    ),
    "mar": TrackCTargetSpec(
        SIX_LABELS,
        "33a302279fd589690870b05026276e10b85e80cfe8019234e8dca25efa4857bb",
        "e8ed703b1193a9d6fa935c10150730cb09413e961f642fd1ee226e0e80394582",
    ),
    "orm": TrackCTargetSpec(
        SIX_LABELS,
        "a94071b1297085a9b849ba1f41c2c70617a57417519bc0680a91924dca545058",
        "6221a6ae0047a61bd6a1245ea51a8f63f8621e677090ba618fb52ad05684c0e5",
    ),
    "pcm": TrackCTargetSpec(
        SIX_LABELS,
        "0a5d5bd5429d45f7f961e2f54f7059235c8e1ba3aa2ad44b7be09566cd63fde1",
        "f66e35c44f0d3ddba2d47973d9908ecda220a3383433d8457424d69719713c56",
    ),
    "ptbr": TrackCTargetSpec(
        SIX_LABELS,
        "c9c257b87b3ada06e9164d9e58f66add97012a9f63d8979bc1b7129205d80938",
        "59564d1ae5c862f4a0dbe18836308b449dadf5658530d63fc10f492fabbeeb72",
    ),
    "ptmz": TrackCTargetSpec(
        SIX_LABELS,
        "bb870bb2f4af887ebe8a5f87a869310dbf12e1cb0ca2a1b80a65449024cd8b36",
        "a6c8933fa481649aa143379d614abc90600779d8a0a29bb858856a312ecf5bb8",
    ),
    "ron": TrackCTargetSpec(
        SIX_LABELS,
        "28a6d7fa571cbb00132d748a53b1dfc78853869a50c7fb6226401fc2deb10bf3",
        "0942c62c9f44fb101cec16d0627349b621dedd173ccc01712ad5dc5d0dbd4f46",
    ),
    "rus": TrackCTargetSpec(
        SIX_LABELS,
        "6826f22faff3a120c6f46623049f8c1f8479e4b9bb43dfc69850017c4c886ed3",
        "e2ea34052d04fbfac6ea94010b540ee7400f289e5e6e73ba32439d978276b447",
    ),
    "som": TrackCTargetSpec(
        SIX_LABELS,
        "7ad50fd29a596342991295911339e2d042fe4e03d9e0fabb8a2189fcc09091da",
        "fdd05f92de75ede0a3edd3d2d741826d31ad043c3b016729d7ef383d403f7aeb",
    ),
    "sun": TrackCTargetSpec(
        SIX_LABELS,
        "6da257924ca84973ed84e0362e03661b652c0c903527e825ac1b3714914c12fe",
        "284fc161d7b2e2b6e5aa2e93b20dae57b858f95e4007811931b6a6934985ed05",
    ),
    "swa": TrackCTargetSpec(
        SIX_LABELS,
        "6d09829eb5d860751ea2cca77b04148ffdaccd3215c11acc73721253867cdc5a",
        "cdad0be3e324c0874997417b143d12918adba2447ebb01dfd69ab0007e06e5ca",
    ),
    "swe": TrackCTargetSpec(
        SIX_LABELS,
        "8730d68bc20151ec73ecd03e32d5509597249e40d5cce57059f63bbc67e18883",
        "773e4a64dc0d98f82060cfc59d4f7846d616604381faaf37036699741364cc5d",
    ),
    "tat": TrackCTargetSpec(
        SIX_LABELS,
        "752a1cd2d055d58a09cf9a56131e7e0dac4eb568f7a09741dfea53a0276b910a",
        "27d35834efc9227c03767d9c086dbeb2d2740653ff81263ba4f97bcc4a0f4c7e",
    ),
    "tir": TrackCTargetSpec(
        SIX_LABELS,
        "4b9556357ef588019b95035a7390163c0c200210225571161bdda2754821a475",
        "3d5d699165bdcb03987158ccc218d68f29ba8c99e922efd5a4a28fed8236a54a",
    ),
    "ukr": TrackCTargetSpec(
        SIX_LABELS,
        "56739ec52aef944686241557ef7185daa1b50a552f43f6eca18e6f416a3ed1e1",
        "232ae427fcc92d60c30aa7b82629d48f9edc924419caee322410dc6127bd3b72",
    ),
    "vmw": TrackCTargetSpec(
        SIX_LABELS,
        "315b9100004256956a5f59acbabc890afef7e9d9f0b191f738a7861ef1a8aa51",
        "02edd57d8ae43cc185bf4f0b12238f1e3699dc5744c61d85f7631052de01123b",
    ),
    "xho": TrackCTargetSpec(
        SIX_LABELS,
        "9b41a76df23d6ed95e326ce1f84dc256bbdf3ac4e25927c2f0dc6252d672e384",
        "d350ec7f8d77e3f6bddfdd1b79bbd91b3dc0aff9826e1bbc21fe34c61f33c430",
    ),
    "yor": TrackCTargetSpec(
        SIX_LABELS,
        "841317d92f5a689143b7c6b233ae829b8289698f08ae4b7a026f16199db511aa",
        "0dc466d662361fb9fa44ace49fc04ac7b840ab6f1da72b54c8db3c9d84479606",
    ),
    "zul": TrackCTargetSpec(
        SIX_LABELS,
        "c8d6e75fe43fe40910b5e95329926342f914c973ecbdd1907e7da869f957c005",
        "40f60d88a0503f07577327c9a4f1de730999b772e730ca0308a96585579fba7d",
    ),
}

# Explicit, versioned task-paper expansion; never infer a family from a code.
_SOURCE_LANGUAGES = {
    "afr": (
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "deu": (
        "afr",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "eng": (
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
    ),
    "esp": (
        "afr",
        "deu",
        "eng",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "hin": (
        "afr",
        "deu",
        "eng",
        "esp",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "mar": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "ptbr": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptmz",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "ptmz": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ron",
        "rus",
        "swe",
        "ukr",
    ),
    "ron": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "rus",
        "swe",
        "ukr",
    ),
    "rus": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "swe",
        "ukr",
    ),
    "swe": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "ukr",
    ),
    "ukr": (
        "afr",
        "deu",
        "eng",
        "esp",
        "hin",
        "mar",
        "ptbr",
        "ptmz",
        "ron",
        "rus",
        "swe",
    ),
    "amh": ("arq", "ary", "hau", "orm", "som", "tir"),
    "arq": ("amh", "ary", "hau", "orm", "som", "tir"),
    "ary": ("amh", "arq", "hau", "orm", "som", "tir"),
    "hau": ("amh", "arq", "ary", "orm", "som", "tir"),
    "orm": ("amh", "arq", "ary", "hau", "som", "tir"),
    "som": ("amh", "arq", "ary", "hau", "orm", "tir"),
    "tir": ("amh", "arq", "ary", "hau", "orm", "som"),
    "ibo": ("kin", "swa", "vmw", "yor"),
    "kin": ("ibo", "swa", "vmw", "yor"),
    "swa": ("ibo", "kin", "vmw", "yor"),
    "vmw": ("ibo", "kin", "swa", "yor"),
    "yor": ("ibo", "kin", "swa", "vmw"),
    "xho": ("ibo", "kin", "swa", "vmw", "yor"),
    "zul": ("ibo", "kin", "swa", "vmw", "yor"),
    "ind": ("sun",),
    "jav": ("sun",),
    "tat": ("rus", "ukr"),
    "pcm": ("swa", "yor"),
    "chn": ("rus",),
}


def available_track_c_targets(release=DEFAULT_RELEASE):
    if release != DEFAULT_RELEASE:
        raise ValueError(f"Unsupported release: {release}")
    return tuple(sorted(_TRACK_C_TARGETS))


def get_track_c_targets(language, release=DEFAULT_RELEASE):
    if language not in available_track_c_targets(release):
        raise ValueError(f"Unsupported Track C target: {language}")
    return tuple(
        _target_source(language, split, _TRACK_C_TARGETS[language], release)
        for split in ("dev", "test")
    )


def track_c_source_languages(language, protocol=DEFAULT_C_PROTOCOL):
    if protocol != DEFAULT_C_PROTOCOL:
        raise ValueError(f"Unsupported Track C protocol: {protocol}")
    if language not in available_track_c_targets():
        raise ValueError(f"Unsupported Track C target: {language}")
    if language == "sun":
        raise ValueError(
            "Track C sun has no available other Austronesian training language; "
            "an alternative source protocol must be explicitly agreed, not silently substituted"
        )
    return _SOURCE_LANGUAGES[language]


def get_crosslingual_sources(
    language, release=DEFAULT_RELEASE, protocol=DEFAULT_C_PROTOCOL
):
    targets = get_track_c_targets(language, release)
    languages = track_c_source_languages(language, protocol)
    return (
        tuple(
            source
            for name in languages
            for source in get_reference_sources("a", name, release)
            if source.split in ("train", "dev")
        )
        + targets
    )
