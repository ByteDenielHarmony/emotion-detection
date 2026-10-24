"""Reference manifests and generic acquisition; synthetic CSV/Parquet only."""

import hashlib
import io
from dataclasses import replace
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

from emotion_detection.datasets import acquisition
from emotion_detection.datasets.sources import (
    DEFAULT_RELEASE,
    ENGLISH_LABELS,
    OFFICIAL_REVISION,
    SIX_LABELS,
    get_reference_sources,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No real network in reference data tests")

    monkeypatch.setattr(acquisition, "urlopen", forbidden)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)


@pytest.fixture
def csv_source():
    payload = b"id,text,anger,fear,joy,sadness,surprise\n001,NA,0,0,1,0,0\n"
    source = replace(
        get_reference_sources()[0], sha256=hashlib.sha256(payload).hexdigest()
    )
    return source, payload


@pytest.mark.parametrize(
    "language,labels", [("eng", ENGLISH_LABELS), ("deu", SIX_LABELS)]
)
def test_explicit_manifest_pins_complete_track_a(language, labels):
    sources = get_reference_sources("a", language)
    assert [s.split for s in sources] == ["train", "dev", "test"]
    assert len({s.relative_path for s in sources}) == 3
    for source in sources:
        assert source.release == DEFAULT_RELEASE
        assert source.revision == OFFICIAL_REVISION
        assert source.label_names == labels
        assert OFFICIAL_REVISION in source.url
        assert source.language == language
        assert len(source.sha256) == 64
        assert "huggingface" not in source.url


@pytest.mark.parametrize(
    "kwargs",
    [
        {"track": "b"},
        {"track": "c"},
        {"language": "unknown"},
        {"release": "main"},
    ],
)
def test_unregistered_tracks_languages_and_releases_fail_explicitly(kwargs):
    with pytest.raises(ValueError):
        get_reference_sources(**kwargs)


@pytest.mark.parametrize(
    "path", ["../file.csv", "/file.csv", "C:/file.csv", "a\\file.csv"]
)
def test_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValueError):
        replace(get_reference_sources()[0], relative_path=path)


def test_manifest_rejects_a_moving_revision():
    with pytest.raises(ValueError, match="pinned commit"):
        replace(get_reference_sources()[0], revision="main")


def test_csv_preserves_literal_text_and_leading_zero_ids(csv_source, tmp_path):
    source, payload = csv_source
    path = tmp_path / "train.csv"
    path.write_bytes(payload)
    acquisition.verify_source(path, source)
    frame = acquisition.read_source(path, source)
    assert frame.iloc[0]["id"] == "001"
    assert frame.iloc[0]["text"] == "NA"


def test_parquet_format_is_verified_too(csv_source, tmp_path):
    source, _ = csv_source
    frame = pd.DataFrame(
        {"id": ["a"], "text": ["TEXT"], **{name: [0] for name in ENGLISH_LABELS}}
    )
    path = tmp_path / "train.parquet"
    frame.to_parquet(path, index=False)
    source = replace(
        source,
        file_format="parquet",
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    acquisition.verify_source(path, source)
    pd.testing.assert_frame_equal(acquisition.read_source(path, source), frame)


def test_existing_file_uses_no_network_and_is_never_overwritten(csv_source, tmp_path):
    source, payload = csv_source
    path = tmp_path / source.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    acquisition.acquire_sources([source], tmp_path, interactive=False)
    path.write_bytes(b"different user data")
    with pytest.raises(acquisition.AcquisitionError, match="rerun"):
        acquisition.acquire_sources([source], tmp_path, interactive=False)
    assert path.read_bytes() == b"different user data"


def test_csv_download_checks_hash_before_publication(csv_source, tmp_path, monkeypatch):
    source, payload = csv_source
    monkeypatch.setattr(acquisition, "urlopen", lambda *a, **kw: io.BytesIO(payload))
    acquisition.acquire_sources([source], tmp_path, interactive=False)
    assert (tmp_path / source.relative_path).read_bytes() == payload
    assert not list(tmp_path.rglob("*.part"))


def test_incorrect_download_is_rejected_and_partial_is_removed(
    csv_source, tmp_path, monkeypatch
):
    source, _ = csv_source
    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong version")
    )
    with pytest.raises(acquisition.AcquisitionError, match="Checksum mismatch"):
        acquisition.download_source(source, tmp_path / "train.csv")
    assert not (tmp_path / "train.csv").exists()
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("status,attempts", [(404, 1), (403, 1), (503, 3), (429, 3)])
def test_http_failures_have_bounded_retries(
    csv_source, tmp_path, monkeypatch, status, attempts
):
    source, _ = csv_source
    calls = []

    def unavailable(request, **kwargs):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "unavailable", {}, None)

    monkeypatch.setattr(acquisition, "urlopen", unavailable)
    with pytest.raises(acquisition.AcquisitionError, match=f"HTTP {status}"):
        acquisition.download_source(source, tmp_path / "train.csv")
    assert len(calls) == attempts


def test_manual_recovery_verifies_replacement_and_resumes(
    csv_source, tmp_path, monkeypatch
):
    source, payload = csv_source
    calls, prompts = [], []

    def unavailable(*args, **kwargs):
        calls.append(1)
        raise URLError("offline")

    def supply(prompt):
        prompts.append(prompt)
        (tmp_path / source.relative_path).write_bytes(
            b"invalid" if len(prompts) == 1 else payload
        )
        return ""

    monkeypatch.setattr(acquisition, "urlopen", unavailable)
    monkeypatch.setattr("builtins.input", supply)
    acquisition.acquire_sources([source], tmp_path, interactive=True)
    assert len(calls) == 3 and len(prompts) == 2
    acquisition.verify_source(tmp_path / source.relative_path, source)


def test_noninteractive_failure_reports_manual_path(
    csv_source, tmp_path, monkeypatch, capsys
):
    source, _ = csv_source
    monkeypatch.setattr(
        acquisition,
        "urlopen",
        lambda *a, **kw: (_ for _ in ()).throw(URLError("offline")),
    )
    with pytest.raises(acquisition.AcquisitionError, match="rerun"):
        acquisition.acquire_sources([source], tmp_path, interactive=False)
    error = capsys.readouterr().err
    assert str(tmp_path / source.relative_path) in error
    assert source.revision in error and source.sha256 in error


def test_unexpected_input_is_rejected(csv_source, tmp_path):
    source, payload = csv_source
    path = tmp_path / source.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    (path.parent / "other-release.csv").write_bytes(payload)
    with pytest.raises(
        acquisition.AcquisitionError, match="Unexpected reference input"
    ):
        acquisition.acquire_sources([source], tmp_path, interactive=False)


def test_duplicate_csv_headers_are_not_silently_renamed(csv_source, tmp_path):
    source, _ = csv_source
    payload = b"id,text,anger,anger\na,T,0,1\n"
    path = tmp_path / "bad.csv"
    path.write_bytes(payload)
    source = replace(source, sha256=hashlib.sha256(payload).hexdigest())
    with pytest.raises(acquisition.AcquisitionError, match="Duplicate CSV"):
        acquisition.verify_source(path, source)


def test_interrupted_reference_download_removes_temporary_file(
    csv_source, tmp_path, monkeypatch
):
    source, payload = csv_source

    class BrokenResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("connection lost")
            return super().read(5)

    monkeypatch.setattr(
        acquisition, "urlopen", lambda *a, **kw: BrokenResponse(payload)
    )
    with pytest.raises(acquisition.AcquisitionError, match="connection lost"):
        acquisition.download_source(source, tmp_path / "train.csv")
    assert not (tmp_path / "train.csv").exists()
    assert not list(tmp_path.glob("*.part"))


def test_concurrently_supplied_reference_file_is_not_overwritten(
    csv_source, tmp_path, monkeypatch
):
    source, payload = csv_source
    destination = tmp_path / "train.csv"

    def response(*args, **kwargs):
        destination.write_bytes(b"user data")
        return io.BytesIO(payload)

    monkeypatch.setattr(acquisition, "urlopen", response)
    with pytest.raises(acquisition.AcquisitionError, match="Checksum mismatch"):
        acquisition.download_source(source, destination)
    assert destination.read_bytes() == b"user data"
    assert not list(tmp_path.glob("*.part"))


def test_reference_acquisition_resumes_and_preserves_completed_files(
    csv_source, tmp_path, monkeypatch
):
    source, payload = csv_source
    second = replace(source, split="dev", relative_path="track_a/eng/dev.csv")
    completed = tmp_path / source.relative_path
    completed.parent.mkdir(parents=True)
    completed.write_bytes(payload)

    def unavailable(*args, **kwargs):
        raise URLError("offline")

    monkeypatch.setattr(acquisition, "urlopen", unavailable)
    monkeypatch.setattr("builtins.input", lambda prompt: "q")
    with pytest.raises(acquisition.AcquisitionError, match="Existing files were kept"):
        acquisition.acquire_sources([source, second], tmp_path, interactive=True)
    assert completed.read_bytes() == payload
    calls = []

    def recovered(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(payload)

    monkeypatch.setattr(acquisition, "urlopen", recovered)
    acquisition.acquire_sources([source, second], tmp_path, interactive=False)
    assert len(calls) == 1
    assert completed.read_bytes() == payload
