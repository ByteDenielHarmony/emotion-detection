"""Acquisition and first-run integration tests; no real network requests."""

import hashlib
import io
import ssl
from dataclasses import replace
from urllib.error import HTTPError, URLError

import pandas as pd
import pytest

import emotion_detection.datasets.prepare as preparation


@pytest.fixture(autouse=True)
def prevent_real_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not access the real network")

    monkeypatch.setattr(preparation, "urlopen", forbidden)
    monkeypatch.setattr(preparation.time, "sleep", lambda _: None)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    sources, payloads = [], {}
    for track in ("a", "b"):
        for split in preparation.SPLITS:
            frame = pd.DataFrame(
                [
                    {
                        "id": f"{track}-{split}",
                        "text": f"{split} text" if track == "a" else f"{split} TEXT",
                        "disgust": None,
                        **{emotion: 0 for emotion in preparation.EMOTIONS},
                    }
                ]
            )
            stream = io.BytesIO()
            frame.to_parquet(stream, index=False)
            payload = stream.getvalue()
            source = preparation.SourceFile(
                track, split, hashlib.sha256(payload).hexdigest()
            )
            sources.append(source)
            payloads[source.url] = payload
    monkeypatch.setattr(preparation, "DATASET_FILES", tuple(sources))
    return raw, sources, payloads


def save_source(raw, source, payloads):
    path = raw / source.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payloads[source.url])
    return path


def fake_downloads(monkeypatch, payloads, *, unavailable=()):
    calls = []

    def open_url(request, *, timeout):
        assert timeout > 0
        calls.append(request.full_url)
        if request.full_url in unavailable:
            raise HTTPError(request.full_url, 404, "Not Found", {}, None)
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(preparation, "urlopen", open_url)
    return calls


def cli_args(raw, tmp_path):
    return [
        "--raw-dir",
        str(raw),
        "--output-dir",
        str(tmp_path / "processed"),
        "--analysis-dir",
        str(tmp_path / "analysis"),
        "--non-interactive",
    ]


def test_manifest_contains_six_pinned_english_files():
    assert len(preparation.DATASET_FILES) == 6
    assert len({source.relative_path for source in preparation.DATASET_FILES}) == 6
    for source in preparation.DATASET_FILES:
        assert len(source.sha256) == 64
        assert preparation.SOURCES[source.track]["revision"] in source.url
        assert "/resolve/main/" not in source.url
        assert "/eng/" in source.url


def test_complete_local_inputs_never_use_network(inputs):
    raw, sources, payloads = inputs
    for source in sources:
        save_source(raw, source, payloads)
    preparation.ensure_raw_data(raw, interactive=False)
    for source in sources:
        assert (raw / source.relative_path).read_bytes() == payloads[source.url]


def test_first_run_downloads_and_verifies_all_files(inputs, monkeypatch):
    raw, sources, payloads = inputs
    calls = fake_downloads(monkeypatch, payloads)
    preparation.ensure_raw_data(raw, interactive=False)
    assert calls == [source.url for source in sources]
    assert not list(raw.rglob("*.part"))
    for source in sources:
        preparation.verify_source_file(raw / source.relative_path, source)


def test_partial_local_dataset_downloads_only_missing_files(inputs, monkeypatch):
    raw, sources, payloads = inputs
    for source in sources[:3]:
        save_source(raw, source, payloads)
    calls = fake_downloads(monkeypatch, payloads)
    preparation.ensure_raw_data(raw, interactive=False)
    assert calls == [source.url for source in sources[3:]]


def test_invalid_existing_file_is_not_overwritten(inputs, capsys):
    raw, sources, payloads = inputs
    for source in sources:
        save_source(raw, source, payloads)
    path = raw / sources[0].relative_path
    path.write_bytes(b"user supplied different version")
    with pytest.raises(
        preparation.DatasetAcquisitionError, match="rerun the same command"
    ):
        preparation.ensure_raw_data(raw, interactive=False)
    assert path.read_bytes() == b"user supplied different version"
    error = capsys.readouterr().err
    assert "Checksum mismatch" in error
    assert str(path) in error
    assert "Existing files have not been overwritten" in error


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_permanent_http_errors_are_not_retried(inputs, monkeypatch, status):
    raw, sources, _ = inputs
    calls = []

    def fail(request, *, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, status, "Unavailable", {}, None)

    monkeypatch.setattr(preparation, "urlopen", fail)
    with pytest.raises(preparation.DatasetAcquisitionError, match=f"HTTP {status}"):
        preparation.download_source_file(raw, sources[0])
    assert len(calls) == 1
    assert not (raw / sources[0].relative_path).exists()


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timeout"), URLError("offline"), ssl.SSLError("TLS interrupted")],
)
def test_network_errors_have_bounded_retries(inputs, monkeypatch, failure):
    raw, sources, _ = inputs
    calls = []

    def fail(request, *, timeout):
        calls.append(request.full_url)
        raise failure

    monkeypatch.setattr(preparation, "urlopen", fail)
    with pytest.raises(preparation.DatasetAcquisitionError, match="Unable to download"):
        preparation.download_source_file(raw, sources[0])
    assert len(calls) == 3
    assert not list(raw.rglob("*.part"))


@pytest.mark.parametrize("status", [429, 503])
def test_temporary_http_failure_can_recover(inputs, monkeypatch, status):
    raw, sources, payloads = inputs
    calls = []

    def open_url(request, *, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise HTTPError(request.full_url, status, "Try later", {}, None)
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(preparation, "urlopen", open_url)
    preparation.download_source_file(raw, sources[0])
    assert len(calls) == 2
    preparation.verify_source_file(raw / sources[0].relative_path, sources[0])


def test_interrupted_download_never_becomes_final_input(inputs, monkeypatch):
    raw, sources, payloads = inputs

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            if self.tell():
                raise ConnectionResetError("connection lost")
            return super().read(10)

    monkeypatch.setattr(
        preparation,
        "urlopen",
        lambda request, **kwargs: InterruptedResponse(payloads[request.full_url]),
    )
    with pytest.raises(preparation.DatasetAcquisitionError, match="connection lost"):
        preparation.download_source_file(raw, sources[0])
    assert not (raw / sources[0].relative_path).exists()
    assert not list(raw.rglob("*.part"))


def test_wrong_download_checksum_is_not_accepted(inputs, monkeypatch):
    raw, sources, _ = inputs
    monkeypatch.setattr(preparation, "urlopen", lambda *a, **kw: io.BytesIO(b"wrong"))
    with pytest.raises(preparation.DatasetAcquisitionError, match="Checksum mismatch"):
        preparation.download_source_file(raw, sources[0])
    assert not (raw / sources[0].relative_path).exists()
    assert not list(raw.rglob("*.part"))


def test_file_supplied_during_download_is_never_overwritten(inputs, monkeypatch):
    raw, sources, payloads = inputs
    source = sources[0]
    path = raw / source.relative_path

    def supply_during_download(request, *, timeout):
        path.write_bytes(b"user supplied file")
        return io.BytesIO(payloads[request.full_url])

    monkeypatch.setattr(preparation, "urlopen", supply_during_download)
    with pytest.raises(preparation.DatasetAcquisitionError, match="Checksum mismatch"):
        preparation.download_source_file(raw, source)
    assert path.read_bytes() == b"user supplied file"
    assert not list(raw.rglob("*.part"))


@pytest.mark.parametrize("invalid_parquet", [False, True])
def test_matching_hash_still_requires_readable_parquet_and_schema(
    inputs,
    invalid_parquet,
):
    raw, sources, _ = inputs
    if invalid_parquet:
        payload = b"not a parquet file"
    else:
        stream = io.BytesIO()
        pd.DataFrame({"id": ["a"], "text": ["text"]}).to_parquet(stream)
        payload = stream.getvalue()
    source = replace(sources[0], sha256=hashlib.sha256(payload).hexdigest())
    path = raw / source.relative_path
    path.parent.mkdir(parents=True)
    path.write_bytes(payload)
    with pytest.raises(preparation.DatasetFileError):
        preparation.verify_source_file(path, source)


def test_manual_placement_rechecks_until_valid_without_retrying_network(
    inputs,
    monkeypatch,
    capsys,
):
    raw, sources, payloads = inputs
    missing = sources[0]
    for source in sources[1:]:
        save_source(raw, source, payloads)
    calls = fake_downloads(monkeypatch, payloads, unavailable=[missing.url])
    prompts = []

    def supply_file(prompt):
        prompts.append(prompt)
        path = raw / missing.relative_path
        path.write_bytes(b"wrong" if len(prompts) == 1 else payloads[missing.url])
        return ""

    monkeypatch.setattr("builtins.input", supply_file)
    preparation.ensure_raw_data(raw, interactive=True)
    assert len(prompts) == 2
    assert calls == [missing.url]
    preparation.verify_source_file(raw / missing.relative_path, missing)
    error = capsys.readouterr().err
    assert "Checksum mismatch" in error
    assert "Required revision:" in error
    assert str(raw / missing.relative_path) in error


@pytest.mark.parametrize("end_of_input", [False, True])
def test_user_can_stop_and_keep_completed_downloads(inputs, monkeypatch, end_of_input):
    raw, sources, payloads = inputs
    missing = sources[0]
    fake_downloads(monkeypatch, payloads, unavailable=[missing.url])

    def stop(prompt):
        if end_of_input:
            raise EOFError
        return "q"

    monkeypatch.setattr("builtins.input", stop)
    with pytest.raises(
        preparation.DatasetAcquisitionError, match="Existing files were kept"
    ):
        preparation.ensure_raw_data(raw, interactive=True)
    for source in sources[1:]:
        preparation.verify_source_file(raw / source.relative_path, source)


def test_rerun_resumes_only_the_missing_file(inputs, monkeypatch):
    raw, sources, payloads = inputs
    missing = sources[0]
    fake_downloads(monkeypatch, payloads, unavailable=[missing.url])
    with pytest.raises(preparation.DatasetAcquisitionError):
        preparation.ensure_raw_data(raw, interactive=False)
    calls = fake_downloads(monkeypatch, payloads)
    preparation.ensure_raw_data(raw, interactive=False)
    assert calls == [missing.url]


def test_noninteractive_cli_fails_with_instructions_without_preparing(
    inputs,
    monkeypatch,
    tmp_path,
    capsys,
):
    raw, sources, payloads = inputs
    missing = sources[0]
    fake_downloads(monkeypatch, payloads, unavailable=[missing.url])
    code = preparation.main(cli_args(raw, tmp_path))
    assert code == 2
    assert not (tmp_path / "processed").exists()
    assert not (tmp_path / "analysis").exists()
    error = capsys.readouterr().err
    assert "[ERROR]" in error and "HTTP 404" in error
    assert str(raw / missing.relative_path) in error
    assert "Traceback" not in error


def test_no_tty_does_not_prompt(inputs, monkeypatch):
    raw, sources, payloads = inputs
    fake_downloads(monkeypatch, payloads, unavailable=[sources[0].url])
    monkeypatch.setattr(preparation.sys, "stdin", io.StringIO())

    def forbidden(prompt):
        raise AssertionError("Noninteractive runs must never prompt")

    monkeypatch.setattr("builtins.input", forbidden)
    with pytest.raises(
        preparation.DatasetAcquisitionError, match="interactive input is unavailable"
    ):
        preparation.ensure_raw_data(raw)


def test_first_run_cli_downloads_and_integrates(inputs, monkeypatch, tmp_path):
    raw, _, payloads = inputs
    fake_downloads(monkeypatch, payloads)
    assert preparation.main(cli_args(raw, tmp_path)) == 0
    for split in preparation.SPLITS:
        output = pd.read_parquet(tmp_path / f"processed/{split}.parquet")
        assert len(output) == 1
        assert output.iloc[0]["text"] == f"{split} TEXT"
    assert (tmp_path / "analysis/summary.csv").is_file()
    assert not list((tmp_path / "analysis").glob("*.json*"))


def test_keyboard_interrupt_cleans_partial_download_and_returns_130(
    inputs,
    monkeypatch,
    tmp_path,
    capsys,
):
    raw, _, _ = inputs

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            raise KeyboardInterrupt

    monkeypatch.setattr(preparation, "urlopen", lambda *a, **kw: InterruptedResponse())
    assert preparation.main(cli_args(raw, tmp_path)) == 130
    assert not list(raw.rglob("*.part"))
    assert not list(raw.rglob("*.parquet"))
    assert "Interrupted" in capsys.readouterr().err


def test_unexpected_shards_are_not_silently_mixed_into_pinned_data(inputs):
    raw, sources, payloads = inputs
    path = save_source(raw, sources[0], payloads)
    extra = path.with_name("train-extra.parquet")
    extra.write_bytes(b"user file")
    with pytest.raises(
        preparation.DatasetAcquisitionError, match="Unexpected input shard"
    ):
        preparation.ensure_raw_data(raw, interactive=False)
    assert extra.read_bytes() == b"user file"


def test_manual_recovery_cannot_introduce_unverified_shards(inputs, monkeypatch):
    raw, sources, payloads = inputs
    missing = sources[0]
    for source in sources[1:]:
        save_source(raw, source, payloads)
    fake_downloads(monkeypatch, payloads, unavailable=[missing.url])

    def supply_files(prompt):
        path = save_source(raw, missing, payloads)
        path.with_name("train-extra.parquet").write_bytes(b"another version")
        return ""

    monkeypatch.setattr("builtins.input", supply_files)
    with pytest.raises(
        preparation.DatasetAcquisitionError, match="Unexpected input shard"
    ):
        preparation.ensure_raw_data(raw, interactive=True)


def test_cli_rejects_overlapping_paths_before_downloading(tmp_path, capsys):
    raw = tmp_path / "raw"
    assert (
        preparation.main(
            [
                "--raw-dir",
                str(raw),
                "--output-dir",
                str(raw),
                "--analysis-dir",
                str(tmp_path / "analysis"),
                "--non-interactive",
            ]
        )
        == 2
    )
    assert not raw.exists()
    assert "must not overlap" in capsys.readouterr().err
