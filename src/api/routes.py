"""API routes for the frontend test runner service."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import uuid
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Final

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError
from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse
from w2v_common.auth import LICENSE_GATE, clear_identity_context

from src.api.schemas import (
    AsyncJobResponse,
    JobStatusResponse,
    TestExecutionRequest,
    TestExecutionResult,
    without_auth_secrets,
)
from src.cache.redis_client import get_redis_client
from src.config import get_settings
from src.database.models import TestExecutionJob
from src.database.session import get_db_session
from src.middleware.rate_limiter import concurrent_jobs_key
from src.services.drain import JobDrain, get_job_drain
from src.services.execution_engine import ExecutionEngine
from src.services.job_manager import JobManager, JobStateConflictError
from src.services.storage_service import StorageService
from src.services.task_identity import TaskIdentity, bind_task_identity, capture_task_identity

logger = logging.getLogger(__name__)
# The entitlement gate is mounted router-wide, not per-route. This service is a
# BILLED/metered destination (``w2v_common.auth.outbound.METERED_DESTINATION_SERVICES``)
# and until now carried no gate at all: ``AuthMiddleware`` establishes tenant
# IDENTITY ("who are you"), never ENTITLEMENT ("may you"). A tenant whose licence
# had expired or been revoked still presented a syntactically valid token, so
# every route below — including ``POST /execute``, which spends real browser/CPU
# minutes and is metered — served them.
#
# Router-wide is deliberate: a per-route list is a thing to forget on the next
# route added. Every route on this router is tenant-scoped (each one calls
# ``_require_tenant``), so there is no identity-less caller to strand — a request
# with no tenant already 401s here and never reaches the gate's 402.
#
# The health router (``src/api/health.py``) is NOT gated: probes carry no tenant,
# and a 402/503 on ``/health/live`` or ``/health/ready`` CrashLoops the pod.
# w2v_common additionally exempts those paths by value, so this holds even if a
# future refactor mounts the gate more broadly.
#
# Dormant on merge: ``require_license`` is a no-op unless
# ``W2V_LICENSE_GATE_ENABLED=true``, which gitops arms separately.
router = APIRouter(prefix="/api/test-runner", tags=["test-runner"], dependencies=list(LICENSE_GATE))

_engine = ExecutionEngine()
# Shared object store for artifact reads. Multi-replica safety: the poll/download
# route can land on ANY pod, so reports + clips are streamed from the shared
# store by object key — never a pod-local FileResponse (which 404'd cross-pod).
_storage = StorageService()

# Response headers forwarded from the store's streamed object (keeps the
# range/caching contract intact); X-Amz-*/server headers are dropped.
_FORWARD_RESPONSE_HEADERS = frozenset(
    {"content-length", "content-range", "accept-ranges", "etag", "last-modified"}
)
# Request headers forwarded to the store so partial/conditional GETs work.
_FORWARD_REQUEST_HEADERS = frozenset({"range", "if-range", "if-none-match", "if-modified-since"})


async def _stream_object(
    request: Request,
    key: str,
    *,
    not_found_detail: str,
    media_type: str | None = None,
) -> StreamingResponse:
    """Stream an object-store ``key`` to the client, forwarding range/conditional
    headers. Any replica can serve any job's artifact this way. 404s (via
    HTTPException) if the object is absent."""
    req_headers = {
        k: v for k, v in request.headers.items() if k.lower() in _FORWARD_REQUEST_HEADERS
    }
    stream = await _storage.open_stream(key, request_headers=req_headers or None)
    if stream.status == 404:
        await stream.aclose()
        raise HTTPException(status_code=404, detail=not_found_detail)

    headers = {k: v for k, v in stream.headers.items() if k.lower() in _FORWARD_RESPONSE_HEADERS}
    resolved_media_type = media_type or stream.headers.get("content-type")

    async def _body() -> AsyncIterator[bytes]:
        async for chunk in stream.iter_chunks():
            yield chunk

    return StreamingResponse(
        _body(),
        status_code=stream.status,
        media_type=resolved_media_type,
        headers=headers,
        background=BackgroundTask(stream.aclose),
    )


def _require_tenant(request: Request) -> str:
    """Return the authenticated tenant id (canonical UUID string), or 401.

    Until w2v_common 0.67.0 every call site here read
    ``getattr(request.state, "tenant_id", "anonymous")``. That sentinel was not
    a safe default: it became the tenant FILTER on every by-id read and the
    ``tenant_id`` written by ``/execute``, so a request that reached a route
    with no identity did not fail — it silently transacted as a fictional
    "anonymous" tenant, and two such requests shared one tenant's data.

    0.67.0 made an absent tenant an ERROR rather than a substitution
    (``MissingTenantError`` / ``InvalidTenantIdError``, or a 401 from the
    middleware). This mirrors that at the route boundary, the same way
    ``slide_service/src/api/routes.py`` does: 401, because for a read/execute
    route "no tenant" is exactly "not authenticated" — there is no partial
    answer to give. (``video_compiler_service``'s 400 fits a body-supplied
    tenant; here the tenant only ever comes from auth.)

    The nil uuid is rejected BY VALUE (``.int == 0``), so every spelling —
    dashed, bare hex, braced, ``urn:uuid:`` — is caught rather than the one
    literal string.
    """
    raw = getattr(request.state, "tenant_id", None)
    text_value = str(raw).strip() if raw is not None else ""
    if not text_value:
        raise HTTPException(status_code=401, detail="unauthenticated: no tenant bound to request")
    try:
        parsed = uuid.UUID(text_value)
    except ValueError as exc:
        raise HTTPException(
            status_code=401, detail="unauthenticated: tenant id is not a UUID"
        ) from exc
    if parsed.int == 0:
        raise HTTPException(status_code=401, detail="unauthenticated: the nil uuid is not a tenant")
    return str(parsed)


# Buffer added to ``max_run_seconds`` for a running job's slot deadline, so a
# slot never expires WHILE the job is still legitimately running (the reaper's
# stale cutoff is 30 min ≈ max_run + this buffer). A hard-killed job's slot
# self-expires this long after its deadline instead of leaking forever.
_CONCURRENT_JOB_BUFFER_SECONDS: Final[int] = 300


async def _register_concurrent_job(tenant_id: str, job_id: str) -> bool:
    """Add ``job_id`` to the per-tenant running-jobs sorted set, scored by its
    expiry DEADLINE (``now + max_run_seconds + buffer``).

    Returns True if the slot was registered, False if Redis is unavailable.
    The deadline score is what makes the slot LEAK-PROOF: the admission script
    prunes members whose deadline has passed, so a worker HARD-KILLED before its
    ``finally`` ``ZREM`` cannot hold a slot beyond its own deadline (unlike the
    retired integer counter, whose 1h TTL was reset on every increment and so
    leaked permanently under load).

    ⚠️ The return value gates the ``finally`` ``ZREM`` in ``_process_job``, so it
    must answer "is there a member to remove", NOT "did every Redis call
    succeed". The ``ZADD`` and the housekeeping ``EXPIRE`` are two round trips:
    with both under one handler, a connection reset BETWEEN them (a Redis
    restart/failover — the exact blip this is guarded for) reported False for a
    slot that was in fact ADDED, the release was skipped, and the tenant lost
    that concurrency slot until its own deadline expired it — up to
    ``max_run_seconds`` + 5 min (~30 min) after a job that may have run for two.
    Three such blips silently zero out the default cap of 3.
    """
    try:
        redis_client = get_redis_client()
        client = redis_client._ensure_connected()
        key = concurrent_jobs_key(tenant_id)
        ttl = get_settings().execution.max_run_seconds + _CONCURRENT_JOB_BUFFER_SECONDS
        deadline = time.time() + ttl
        await client.zadd(key, {job_id: deadline})
    except OSError, RuntimeError, RedisError:
        logger.warning("Failed to register concurrent job for tenant=%s", tenant_id)
        return False

    # The member EXISTS from here on — every exit below must return True so the
    # caller's ``finally`` releases it.
    try:
        # Bound an idle tenant's key so it clears once all members have expired.
        await client.expire(key, ttl + 60)
    except OSError, RuntimeError, RedisError:
        # Housekeeping only: every member still carries its own deadline score and
        # the admission script prunes by it, so a missing key TTL costs nothing
        # but an empty zset lingering in Redis.
        logger.warning(
            "Registered concurrent job for tenant=%s but could not set the key TTL", tenant_id
        )
    return True


async def _release_concurrent_job(tenant_id: str, job_id: str) -> None:
    """Remove ``job_id`` from the per-tenant running-jobs sorted set.

    Idempotent: ``ZREM`` of an already-pruned/absent member is a no-op, so a
    late release after the slot self-expired cannot drive the count negative.
    """
    try:
        redis_client = get_redis_client()
        client = redis_client._ensure_connected()
        await client.zrem(concurrent_jobs_key(tenant_id), job_id)
    except OSError, RuntimeError, RedisError:
        logger.warning("Failed to release concurrent job for tenant=%s", tenant_id)


async def _heartbeat_loop(job_id: str) -> None:
    """Periodically touch a running job's ``updated_at`` so the lease-gated reaper
    never fails a genuinely live job.

    A running job holds no lease other than a fresh ``updated_at``; without this,
    a legitimately long run (up to ``max_run_seconds`` ≈ 25 min) would drift past
    the 30-min stale cutoff and be falsely reaped. The heartbeat keeps ``updated_at``
    within one interval of now, so only a crashed/stalled pod (which stops
    heartbeating) ever goes stale enough to be reaped.
    """
    interval = get_settings().execution.heartbeat_interval_seconds
    while True:
        await asyncio.sleep(interval)
        try:
            async with get_db_session() as db:
                await JobManager.touch_heartbeat(db, job_id)
        except (OSError, SQLAlchemyError) as exc:
            logger.warning("[%s] heartbeat touch failed: %s", job_id, exc)


async def _process_job(
    job_id: str,
    request: TestExecutionRequest,
    tenant_id: str,
    license_token: str | None = None,
    *,
    drain: JobDrain,
    identity: TaskIdentity | None = None,
) -> None:
    """Background task: execute tests and persist results.

    ``identity`` is the originating request's identity, captured in request
    scope by ``start_execution`` (``capture_task_identity``). It is bound
    explicitly at task entry, and it is what every in-cell call of this run (the
    bdd package download) mints its ``X-W2V-Cell-Token`` from — fresh per send,
    so a run of any length never holds an expired token. ``None`` (direct
    callers, tests) binds the tenant with NO provenance: RLS only, nothing is
    signed or delegated. ``license_token`` is the dual-send licence fallback;
    the cell binding does not read it (on the cell lane it is ``None``).

    ``drain`` is the app's in-flight registry, injected by ``start_execution``
    from ``request.app.state``. It is keyword-only and has NO default on purpose:
    a background task that forgot to register itself is invisible to shutdown,
    and a default would let that mistake ship green.

    Maintains the per-tenant running-jobs set that the rate-limit middleware
    reads.  ``ZADD`` (scored by expiry deadline) happens at job start; ``ZREM``
    is in a ``finally`` so it runs on success, exception, AND mark_processing
    state-conflict. Even if this worker is HARD-KILLED before ``finally``, its
    slot self-expires at its deadline (the admission script prunes it), so it
    cannot leak. A heartbeat task touches ``updated_at`` while the job runs so
    the lease-gated reaper never fails this live job (see ``_heartbeat_loop``).

    RLS tenant context (Phase D): this is a DETACHED ``BackgroundTasks`` task. It
    does copy the request's context (measured 2026-10-09 behind the real
    ``AuthMiddleware``: tenant, provenance, run and project arrive), but nothing
    may rely on that inheritance — the binding below is explicit. Under ``TENANT_RLS_MODE=enforce`` the ``w2v_common`` ``get_db_session``
    auto-binds ``app.current_tenant`` from the identity context, so without a bind
    here every ``mark_processing`` / ``mark_completed`` / ``mark_failed`` /
    ``touch_heartbeat`` write on ``test_execution_jobs`` (a strict RLS table) would
    be scoped to the EMPTY tenant → 0 rows → the job would stick at ``pending``
    forever. We therefore bind the identity context from ``identity`` / ``tenant_id``
    at task entry (before opening any session AND before spawning
    ``_heartbeat_loop``, whose ``asyncio.create_task`` copies the current context,
    so it inherits the binding), and clear it in ``finally`` so the value never
    leaks into a sibling task on the shared event loop.
    """
    start = time.monotonic()
    settings = get_settings()
    persist_dir = Path(settings.execution.artifacts_root_dir) / job_id

    # Bind the originating request's identity for the detached task: RLS-scoped writes
    # bind this tenant under enforce, and in-cell calls mint cell tokens for exactly
    # this tenant / provenance / run / project / actor. Before P7 this re-seeded the
    # tenant alone, which cleared the human actor (cell tokens went out actor=null)
    # and left provenance/run/project to whatever the task happened to inherit.
    bind_task_identity(
        identity if identity is not None else TaskIdentity.unvouched(tenant_id),
        caller_license_token=license_token,
    )

    # Make this run visible to the shutdown drain. Registered BEFORE any await
    # that can yield, and released in the outermost ``finally`` (below) so every
    # exit path — success, exception, mark_processing conflict, cancellation —
    # clears the slot. Without this the drain has nothing to wait on and the pod
    # tears Redis/DB down underneath a live Playwright run.
    drain.register(job_id)

    incremented = await _register_concurrent_job(tenant_id, job_id)
    heartbeat: asyncio.Task[None] | None = None
    try:
        async with get_db_session() as db:
            try:
                await JobManager.mark_processing(db, job_id)
            except JobStateConflictError:
                logger.info("[%s] mark_processing lost race — another worker owns this job", job_id)
                return

        heartbeat = asyncio.create_task(_heartbeat_loop(job_id))

        try:
            result = await _engine.execute(
                request,
                persist_dir=persist_dir,
                public_artifact_base=f"/api/test-runner/{job_id}/artifacts",
                license_token=license_token,
            )
            elapsed = time.monotonic() - start

            async with get_db_session() as db:
                await JobManager.mark_completed(
                    db,
                    job_id,
                    summary=result["summary"],
                    test_results=result["test_results"],
                    report_url=result.get("report_url", ""),
                    artifact_paths=result.get("artifact_paths"),
                    scenario_clips=result.get("scenario_clips"),
                    capture_warnings=result.get("capture_warnings"),
                    execution_time_seconds=elapsed,
                )

            logger.info(
                "[%s] Test execution completed: %s in %.1fs",
                job_id,
                result["summary"],
                elapsed,
            )

        except Exception as exc:
            elapsed = time.monotonic() - start
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error("[%s] Test execution failed: %s", job_id, error_msg, exc_info=True)

            async with get_db_session() as db:
                await JobManager.mark_failed(
                    db,
                    job_id,
                    error_message=error_msg[:2000],
                    execution_time_seconds=elapsed,
                )
    finally:
        try:
            if heartbeat is not None:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            try:
                if incremented:
                    await _release_concurrent_job(tenant_id, job_id)
            finally:
                # ALWAYS clear the re-seeded identity — even if the release above raised — so
                # the tenant ContextVar can never leak into a sibling task sharing this event
                # loop's context (which under TENANT_RLS_MODE=enforce would bind a later DB
                # write to the WRONG tenant's RLS predicate). This nested finally makes the
                # clear independent of _release_concurrent_job's failure mode.
                clear_identity_context()
        finally:
            # OUTERMOST: the drain must be released whatever the teardown above did.
            # A leaked registration would make every subsequent shutdown burn the
            # full drain timeout and then ERROR about a job that finished long ago.
            drain.unregister(job_id)


@router.post("/execute", operation_id="execute_test_run")
async def start_execution(
    request_body: TestExecutionRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> AsyncJobResponse:
    """Submit a test execution job. Returns immediately with a job_id for polling."""
    # Readiness has already flipped to 503, but endpoint removal is eventually
    # consistent: a caller that resolved the endpoint moments before SIGTERM can
    # still land here. Refuse with a RETRYABLE 503 rather than accepting a run
    # this process will abandon within seconds. No row is created, so the caller
    # may retry with the same Idempotency-Key against a healthy replica.
    drain = get_job_drain(request.app)
    if drain is None:
        # Same fail-closed rule as readiness: no registry means the lifespan has
        # not run, so a job accepted here would be tracked by nothing.
        raise HTTPException(
            status_code=503,
            detail="service is still starting — retry shortly",
            headers={"Retry-After": "5"},
        )
    if drain.draining:
        raise HTTPException(
            status_code=503,
            detail="service is shutting down — retry against another replica",
            headers={"Retry-After": "5"},
        )
    if not request_body.bdd_job_id and not request_body.package_url:
        raise HTTPException(
            status_code=422,
            detail="Either bdd_job_id or package_url must be provided",
        )

    job_id = str(uuid.uuid7())
    tenant_id = _require_tenant(request)
    # The run's identity (tenant + provenance, run, project, actor), captured from
    # AuthMiddleware's binding while it is in scope and bound explicitly by the
    # background task — what its in-cell calls mint cell tokens from. Captured
    # BEFORE the job row exists: no authenticated tenant → MissingTenantError, not a
    # 'pending' job whose calls would all 401.
    identity = capture_task_identity(tenant_id)
    # Caller dedup key: a re-submit of the SAME execution (e.g. demo_video_record
    # retrying after a resumable timeout) returns the ORIGINAL job instead of
    # spawning a second full bddgen+Playwright recording (duplicate paid compute).
    idempotency_key = request.headers.get("Idempotency-Key")
    # Captured request-scoped so the detached background task can authenticate
    # the S2S package download from bdd_playwright_service. Dual-send fallback only:
    # the download's cell token is minted from ``identity`` above, never from this
    # (on the cell lane there is no licence and this is None).
    #
    # ⚠️ Read from ``request.state``, NEVER re-parsed from headers here. This used
    # to be a local copy of w2v_common's extraction rule that read the licence
    # ONLY from ``X-Tenant-Id: Bearer <token>``. w2v_common 0.74.0 widened the
    # canonical rule to ALSO accept an opaque licence on ``Authorization: Bearer
    # <token>`` — the only shape the ``license-authz`` ext-authz gate can read.
    # Once this service sits behind that gate, a request whose licence rides
    # ``Authorization`` authenticated fine but the local copy returned None, so
    # the background job forwarded an EMPTY token and the S2S package download
    # from bdd_playwright_service 401'd — with no failing test anywhere.
    # ``AuthMiddleware`` sets ``request.state.license_token`` to the introspected
    # opaque licence on the licence lane (and to None on the JWT and
    # trusted-network lanes, where there is no opaque licence to forward), which
    # is exactly what belongs here. Using it deletes the drift permanently.
    license_token = getattr(request.state, "license_token", None)

    async with get_db_session() as db:
        job, created = await JobManager.create_job(
            db,
            job_id=job_id,
            tenant_id=tenant_id,
            target_url=request_body.target_url,
            bdd_job_id=request_body.bdd_job_id,
            package_url=request_body.package_url,
            # Login secrets are NOT persisted: the row's `config` is returned verbatim by
            # GET /{job_id}/result, and the background job gets the live request in
            # memory. (Stored in plaintext until the merqi showcase spike, 2026-10-02.)
            config=without_auth_secrets(
                request_body.model_dump(exclude={"target_url", "bdd_job_id", "package_url"})
            ),
            idempotency_key=idempotency_key,
        )

    if not created:
        # Deduped by Idempotency-Key: an execution already exists — DON'T re-run it.
        logger.info("execute deduped on Idempotency-Key; returning existing job %s", job.job_id)
        return AsyncJobResponse(
            job_id=job.job_id,
            status=job.status,
            poll_url=f"/api/test-runner/{job.job_id}",
        )

    background_tasks.add_task(
        _process_job,
        job.job_id,
        request_body,
        tenant_id,
        license_token,
        drain=drain,
        identity=identity,
    )

    return AsyncJobResponse(
        job_id=job.job_id,
        status="pending",
        poll_url=f"/api/test-runner/{job.job_id}",
    )


def _enforce_job_tenant(job: TestExecutionJob | None, tenant_id: str) -> TestExecutionJob:
    """Return the job iff it belongs to the requesting tenant.

    Cross-tenant access returns 404 — same shape as a genuinely missing
    job, no enumeration side channel. Pre-fix every read route here was
    BOLA: job_id alone was enough, any authenticated caller could read
    / download any other tenant's test-runner job + artifacts.
    """
    if job is None or job.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@router.get("/{job_id}", operation_id="get_test_job_status")
async def get_job_status(request: Request, job_id: str) -> JobStatusResponse:
    """Poll job status."""
    tenant_id = _require_tenant(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id)
    job = _enforce_job_tenant(job, tenant_id)

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        error=job.error_message,
    )


@router.get("/{job_id}/result", operation_id="get_test_job_result")
async def get_job_result(request: Request, job_id: str) -> TestExecutionResult:
    """Get full test execution result."""
    tenant_id = _require_tenant(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id)
    job = _enforce_job_tenant(job, tenant_id)

    if job.status not in ("completed", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job {job_id} is still {job.status}",
        )

    from src.api.schemas import ScenarioClip, TestCaseResult, TestSummary

    summary_data = job.summary or {}
    test_results_data = job.test_results or []
    scenario_clips_data = job.scenario_clips or []

    return TestExecutionResult(
        job_id=job.job_id,
        status=job.status,
        target_url=job.target_url,
        summary=TestSummary(**summary_data),
        test_results=[TestCaseResult(**tc) for tc in test_results_data],
        scenario_clips=[ScenarioClip(**clip) for clip in scenario_clips_data],
        report_url=job.report_url,
        execution_time_seconds=job.execution_time_seconds or 0.0,
        config=job.config or {},
        capture_warnings=job.capture_warnings or [],
    )


@router.get("/{job_id}/report", operation_id="get_test_job_report")
async def get_report(request: Request, job_id: str) -> dict[str, Any]:
    """Get HTML report URL for a completed job."""
    tenant_id = _require_tenant(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id)
    job = _enforce_job_tenant(job, tenant_id)

    if job.status != "completed":
        raise HTTPException(status_code=409, detail=f"Job {job_id} is not completed")

    return {
        "job_id": job.job_id,
        "report_url": job.report_url,
        "artifact_paths": job.artifact_paths or {},
    }


@router.get("/{job_id}/clips/{clip_key}/video", operation_id="get_test_job_clip_video")
async def get_clip_video(request: Request, job_id: str, clip_key: str) -> StreamingResponse:
    """Stream a per-scenario video clip for a completed job FROM THE SHARED STORE.

    Tenant-scoped (cross-tenant → 404, no enumeration side channel). The clip
    filename is resolved from the job's ``artifact_paths['clip_files']`` map and
    read by object key ``{job_id}/clips/<filename>`` — so any replica serves any
    job's clip (multi-replica safety), and the caller-supplied ``clip_key`` never
    touches a filesystem path directly (traversal is rejected by ``_safe_key``).
    """
    tenant_id = _require_tenant(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id)
    job = _enforce_job_tenant(job, tenant_id)

    artifact_paths = job.artifact_paths or {}
    clip_files = artifact_paths.get("clip_files") or {}
    filename = clip_files.get(clip_key)
    if not isinstance(filename, str) or not filename:
        raise HTTPException(status_code=404, detail="clip not found")

    try:
        key = _storage._safe_key(job_id, f"clips/{filename}")
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="clip not found") from exc

    media_type = "video/mp4" if filename.endswith(".mp4") else "video/webm"
    return await _stream_object(
        request, key, not_found_detail="clip not found", media_type=media_type
    )


@router.get("/{job_id}/artifacts/{artifact_path:path}", operation_id="get_test_job_artifact")
async def get_artifact(request: Request, job_id: str, artifact_path: str) -> StreamingResponse:
    """Serve a persisted report artifact for a completed job FROM THE SHARED STORE.

    Reads object key ``{job_id}/{artifact_path}`` (e.g.
    ``playwright-report/index.html``) so any replica serves any job's report.
    Path traversal in ``artifact_path`` is rejected by ``_safe_key`` (400).
    """
    tenant_id = _require_tenant(request)
    async with get_db_session() as db:
        job = await JobManager.get_job(db, job_id)
    job = _enforce_job_tenant(job, tenant_id)

    artifact_paths = job.artifact_paths or {}
    if not artifact_paths.get("has_report") and not artifact_paths.get("report_index"):
        raise HTTPException(status_code=404, detail=f"No artifacts found for job {job_id}")

    try:
        key = _storage._safe_key(job_id, artifact_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid artifact path") from exc

    return await _stream_object(
        request, key, not_found_detail=f"Artifact {artifact_path} not found"
    )
