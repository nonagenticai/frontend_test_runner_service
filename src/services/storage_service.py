"""Object storage for frontend_test_runner artifacts (reports + scenario clips).

Thin domain wrapper over the fleet-shared
:class:`w2v_common.storage.StorageService` facade. The raw object I/O
(``put_bytes`` / ``put_file`` / ``get_bytes`` / ``get_object_stream``) is
delegated to the shared facade so every service shares one storage
implementation; this wrapper keeps only the FTR-specific concerns:

* the ``{job_id}/…`` **object-key layout** (``playwright-report/…`` for the HTML
  report tree, ``clips/…`` for per-scenario recordings), and
* the **path-traversal boundary** (``_safe_key``) that confines every write
  target and read source to the ``{job_id}/`` prefix — the facade does not
  sandbox arbitrary ``..`` segments in caller-/LLM-supplied filenames (a report
  asset name, a ``clip_key``, or the ``artifacts/{path}`` request param), so the
  guard lives here, ahead of every delegated call.

**Multi-replica safety (the point of this module).** FTR previously wrote every
report + clip to a pod-local ``/app/data/jobs`` disk and served it via a
``FileResponse``. It also *ignored* the configured ``STORAGE_BACKEND`` — the DB
row said ``completed`` but the polling/download route 404'd whenever it landed on
a replica other than the one that produced the job. This service now honours the
*configured* backend (``STORAGE_BACKEND=minio`` in prod, bucket
``w2v-frontend-test-runner``), so every artifact lives in the shared object store
and ``artifact_paths`` holds object keys, not pod-local paths. Any replica serves
any job's artifacts, unblocking ``replicas > 1``. Local scratch (the temp package
dir) is retained only as the on-pod workspace where clips are transcoded/probed
before being streamed up.
"""

from __future__ import annotations

import asyncio
import logging
import posixpath
from pathlib import Path
from typing import TYPE_CHECKING

from w2v_common.storage import StorageService as SharedStorageService

from src.config import get_settings

if TYPE_CHECKING:
    from w2v_common.storage.service import StreamedObject

logger = logging.getLogger(__name__)


class StorageService:
    """Manages object storage for FTR artifacts.

    Honours the configured backend (``minio`` in prod, ``local`` in dev/CI).
    Every ``store_*`` returns the traversal-safe object **key** it wrote; that
    same key is what :meth:`open_stream` / :meth:`retrieve` read back, on either
    backend (the local backend joins the key onto ``local_path``; MinIO uses it
    verbatim). Keys are therefore portable across replicas and safe to persist in
    the job's ``artifact_paths``.
    """

    def __init__(self) -> None:
        settings = get_settings()
        # Honour the configured backend — closing the pre-fix inert-config gap
        # (the code forced local and ignored STORAGE_BACKEND=minio). Prod points
        # at bucket w2v-frontend-test-runner so artifacts are shared across
        # replicas; dev/CI default to the local backend.
        self._storage = SharedStorageService(settings.storage)
        self._backend = settings.storage.backend
        self._local_path = Path(settings.storage.local_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def backend(self) -> str:
        """The active storage backend (``local`` or ``minio``)."""
        return self._backend

    # --- guards --------------------------------------------------------------

    def _safe_key(self, job_id: str, relpath: str) -> str:
        """Compose ``{job_id}/{relpath}`` into a normalized, traversal-safe key.

        Rejects any ``..`` segment that would escape the ``{job_id}/`` prefix (a
        report asset name, ``clip_key`` filename, or the ``artifacts/{path}``
        request param could carry one). Operates purely on the object-key string
        — MinIO has no filesystem to resolve against — so the same guard holds on
        both backends.
        """
        raw = f"{job_id}/{relpath}"
        normalized = posixpath.normpath(raw)
        prefix = posixpath.normpath(job_id)
        if normalized != prefix and not normalized.startswith(prefix + "/"):
            raise ValueError(f"refusing to access outside job prefix: {job_id}/{relpath}")
        return normalized

    async def _ensure_ready(self) -> None:
        """Lazily initialize the backend once per instance (idempotent).

        Every entry point funnels through here so a freshly-constructed
        ``StorageService`` (the module-level route singleton, the engine's copy)
        can reach MinIO without the caller remembering to call
        :meth:`initialize`. Local → ensures the root dir exists; MinIO →
        constructs the client + verifies the bucket.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            await self._storage.initialize()
            self._initialized = True

    async def initialize(self) -> None:
        """Ensure the backend is ready (create local dir / verify MinIO bucket)."""
        await self._ensure_ready()
        logger.info("Storage initialized (backend=%s, root=%s)", self._backend, self._local_path)

    # --- writes --------------------------------------------------------------

    async def store_bytes(self, job_id: str, relpath: str, data: bytes) -> str:
        """Store raw bytes under ``{job_id}/{relpath}``. Returns the object key."""
        key = self._safe_key(job_id, relpath)
        await self._ensure_ready()
        await self._storage.put_bytes(key, data)
        logger.debug("Stored %d bytes to %s", len(data), key)
        return key

    async def store_file(
        self,
        job_id: str,
        relpath: str,
        local_path: str | Path,
        *,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Stream a file already on local disk into the store WITHOUT buffering it
        in memory — the streaming path for large artifacts (video clips, trace
        zips). Uses the facade's ``put_file`` (``fput_object`` multipart for
        MinIO, buffered copy for local). Returns the object key.
        """
        key = self._safe_key(job_id, relpath)
        await self._ensure_ready()
        await self._storage.put_file(key, local_path, content_type=content_type)
        logger.debug("Streamed file %s to %s", local_path, key)
        return key

    # --- deletes -------------------------------------------------------------

    async def delete_job_objects(self, job_id: str) -> bool:
        """Delete every object under this job's ``{job_id}/`` prefix.

        The counterpart to the ``store_*`` writers, and the object-store half of
        artifact retention: nothing else in this service ever removed a job's
        report tree or its clips, so every completed run leaked its objects
        forever — and reaping the DB row destroyed the only id that could find
        them again.

        ``job_id`` is routed through :meth:`_safe_key` so a traversal-carrying id
        cannot widen the target beyond the job (``_safe_key(job_id, "")``
        normalizes to exactly ``job_id`` and raises otherwise), and the prefix
        keeps its trailing ``/`` so a MinIO listing cannot also match a sibling
        job whose id merely starts with this one. Returns True if anything was
        deleted. A storage error is NOT swallowed here — the caller decides, and
        the cleanup caller deliberately logs and continues so the DB reap runs.
        """
        prefix = self._safe_key(job_id, "")
        await self._ensure_ready()
        return await self._storage.delete_prefix(f"{prefix}/")

    # --- reads ---------------------------------------------------------------

    async def retrieve(self, job_id: str, relpath: str) -> bytes | None:
        """Read raw bytes for ``{job_id}/{relpath}``, or ``None`` if absent."""
        key = self._safe_key(job_id, relpath)
        await self._ensure_ready()
        return await self._storage.get_bytes(key)

    async def open_stream(
        self, key: str, *, request_headers: dict[str, str] | None = None
    ) -> StreamedObject:
        """Open a streaming, range-aware read of an object ``key``.

        Never buffers the whole object in memory — the clip/artifact routes stream
        straight from the shared store to the client, so a seekable ``<video>``
        works and a large trace zip never lands in RAM. Forwards ``Range`` /
        ``If-*`` headers for partial/conditional GETs. The caller MUST
        ``aclose()`` the result (or use it as an ``async with``).
        """
        await self._ensure_ready()
        return await self._storage.get_object_stream(key, request_headers=request_headers)
