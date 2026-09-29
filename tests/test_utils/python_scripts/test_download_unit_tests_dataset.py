# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import multiprocessing
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, call
from zipfile import ZipFile

import pytest

from tests.test_utils.python_scripts import download_unit_tests_dataset


def _archive_bytes(directory: str, content: str) -> bytes:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(f"{directory}/fixture.txt", content)
    return buffer.getvalue()


def test_download_and_extract_asset_prefers_staged_assets(monkeypatch, tmp_path):
    staged_root = tmp_path / "staged"
    staged_dir = staged_root / download_unit_tests_dataset.STAGED_RELEASE_ASSET_DIR
    staged_dir.mkdir(parents=True)
    for asset in download_unit_tests_dataset.ASSETS:
        asset_path = staged_dir / asset["name"]
        asset_path.write_bytes(_archive_bytes(asset_path.stem, asset_path.name))

    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(staged_root))
    get = MagicMock(side_effect=AssertionError("GitHub fallback should not be used"))
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    assert download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    get.assert_not_called()


def test_download_and_extract_asset_falls_back_without_github_token(monkeypatch, tmp_path):
    archives = {
        asset["url"]: _archive_bytes(Path(asset["name"]).stem, asset["name"])
        for asset in download_unit_tests_dataset.ASSETS
    }

    class Response:
        def __init__(self, content: bytes):
            self.content = content

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size: int):
            yield self.content

    get = MagicMock(side_effect=lambda url, **_: Response(archives[url]))
    monkeypatch.setenv(download_unit_tests_dataset.TEST_DATA_ROOT_ENV, str(tmp_path / "missing"))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(download_unit_tests_dataset.requests, "get", get)

    output_dir = tmp_path / "output"
    assert download_unit_tests_dataset.download_and_extract_asset(output_dir)
    assert (output_dir / "datasets" / "fixture.txt").read_text() == "datasets.zip"
    assert (output_dir / "tokenizers" / "fixture.txt").read_text() == "tokenizers.zip"
    assert get.call_args_list == [
        call(asset["url"], stream=True, timeout=60) for asset in download_unit_tests_dataset.ASSETS
    ]


def test_ensure_test_data_preserves_provisioned_data(monkeypatch, tmp_path):
    output = tmp_path / "data"
    output.mkdir()
    (output / "fixture").write_text("ready")
    prepare = MagicMock()
    monkeypatch.setattr(download_unit_tests_dataset, "download_and_extract_asset", prepare)
    download_unit_tests_dataset.ensure_test_data(output)
    prepare.assert_not_called()


def test_ensure_test_data_retries_incomplete_preparation(monkeypatch, tmp_path):
    output = tmp_path / "data"

    def partial(assets_dir):
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "partial").touch()
        return False

    monkeypatch.setattr(download_unit_tests_dataset, "download_and_extract_asset", partial)
    with pytest.raises(RuntimeError, match="Failed to prepare test data"):
        download_unit_tests_dataset.ensure_test_data(output)
    assert (tmp_path / ".data.incomplete").exists()
    prepare = MagicMock(return_value=True)
    monkeypatch.setattr(download_unit_tests_dataset, "download_and_extract_asset", prepare)
    download_unit_tests_dataset.ensure_test_data(output)
    prepare.assert_called_once_with(output)
    assert not (tmp_path / ".data.incomplete").exists()


def test_ensure_test_data_serializes_workers(monkeypatch, tmp_path):
    ctx = multiprocessing.get_context("fork")
    started = ctx.Event()
    release = ctx.Event()
    second_started = ctx.Event()
    second_done = ctx.Event()
    calls = ctx.Value("i", 0)
    output = tmp_path / "data"

    def prepare(assets_dir):
        with calls.get_lock():
            calls.value += 1
        assets_dir.mkdir(exist_ok=True)
        (assets_dir / "partial").touch()
        started.set()
        assert release.wait(10)
        (assets_dir / "ready").touch()
        return True

    def second_worker():
        second_started.set()
        download_unit_tests_dataset.ensure_test_data(output)
        assert (output / "ready").exists()
        second_done.set()

    monkeypatch.setattr(download_unit_tests_dataset, "download_and_extract_asset", prepare)
    first = ctx.Process(target=download_unit_tests_dataset.ensure_test_data, args=(output,))
    second = ctx.Process(target=second_worker)
    first.start()
    try:
        assert started.wait(10)
        second.start()
        assert second_started.wait(10)
        assert not second_done.wait(0.2)
        release.set()
        first.join(10)
        second.join(10)
        assert first.exitcode == second.exitcode == 0
        assert second_done.is_set()
        assert calls.value == 1
    finally:
        release.set()
        for worker in (first, second):
            if worker.pid is not None:
                if worker.is_alive():
                    worker.terminate()
                worker.join(10)
