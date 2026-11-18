"""Tests for the FTR object-storage wrapper (multi-replica artifact storage)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.services.storage_service import StorageService


class TestConfiguredBackendHonoured:
    """The pre-fix bug: the code forced the ``local`` backend and IGNORED
    ``STORAGE_BACKEND=minio``, so prod wrote artifacts to pod-local disk and the
    download 404'd cross-pod. These lock in that the CONFIGURED backend is used."""

    def test_minio_backend_is_used_when_configured(self, monkeypatch):
        from src.config import clear_settings_cache

        monkeypatch.setenv("STORAGE_BACKEND", "minio")
        monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.test:8389")
        monkeypatch.setenv("STORAGE_MINIO_BUCKET", "w2v-frontend-test-runner")
        clear_settings_cache()
        try:
            store = StorageService()
            # Honoured, not silently forced to local (the pre-fix regression).
            assert store.backend == "minio"
        finally:
            clear_settings_cache()

    def test_local_backend_when_configured_local(self, monkeypatch, tmp_path: Path):
        from src.config import clear_settings_cache

        monkeypatch.setenv("STORAGE_BACKEND", "local")
        monkeypatch.setenv("STORAGE_LOCAL_PATH", str(tmp_path))
        clear_settings_cache()
        try:
            store = StorageService()
            assert store.backend == "local"
        finally:
            clear_settings_cache()


class TestSafeKey:
    """Traversal guard confining every key to the ``{job_id}/`` prefix."""

    def _store(self, tmp_path: Path) -> StorageService:
        from tests.conftest import make_local_storage

        return make_local_storage(tmp_path / "store")

    def test_composes_key(self, tmp_path: Path):
        store = self._store(tmp_path)
        assert store._safe_key("job-1", "clips/clip-1.mp4") == "job-1/clips/clip-1.mp4"

    def test_rejects_parent_traversal(self, tmp_path: Path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="outside job prefix"):
            store._safe_key("job-1", "../outside.txt")

    def test_rejects_nested_traversal_escape(self, tmp_path: Path):
        store = self._store(tmp_path)
        with pytest.raises(ValueError, match="outside job prefix"):
            store._safe_key("job-1", "clips/../../etc/passwd")


class TestRoundTrip:
    """Store then read back on the local backend (the CI/dev path)."""

    async def test_store_bytes_and_retrieve(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        key = await store.store_bytes("job-1", "playwright-report/index.html", b"<html>x</html>")
        assert key == "job-1/playwright-report/index.html"
        assert await store.retrieve("job-1", "playwright-report/index.html") == b"<html>x</html>"

    async def test_store_file_streams_and_reads_back(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        src = tmp_path / "clip.mp4"
        src.write_bytes(b"\x00mp4-bytes")
        key = await store.store_file("job-1", "clips/clip-1.mp4", src, content_type="video/mp4")
        assert key == "job-1/clips/clip-1.mp4"
        assert await store.retrieve("job-1", "clips/clip-1.mp4") == b"\x00mp4-bytes"

    async def test_retrieve_absent_returns_none(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        assert await store.retrieve("job-1", "clips/nope.mp4") is None

    async def test_open_stream_serves_by_key(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        key = await store.store_bytes("job-1", "clips/clip-1.mp4", b"streamed")
        stream = await store.open_stream(key)
        try:
            assert stream.status == 200
            chunks = b"".join([chunk async for chunk in stream.iter_chunks()])
            assert chunks == b"streamed"
        finally:
            await stream.aclose()

    async def test_open_stream_absent_is_404(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        stream = await store.open_stream("job-1/clips/missing.mp4")
        try:
            assert stream.status == 404
        finally:
            await stream.aclose()
