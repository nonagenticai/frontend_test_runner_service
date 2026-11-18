"""ASGI application for frontend_test_runner_service.

Built on the shared ``w2v_common.asgi.create_app`` factory. The canonical
middleware (CORS + SecurityHeaders + Auth + RequestId), the error envelope
(incl. ExceptionGroup-unwrapping), and the BearerAuth/TenantIdHeader OpenAPI
customization all come from the factory. Service-specific wiring that stays:
the per-tenant ``RateLimitMiddleware`` (must run INSIDE auth — it reads
``request.state.tenant_id``), the DB/Redis/cleanup-loop lifespan, the
health-check access-log filter, the root banner, and the ``RequestIdLogFilter``
on the root logger.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from collections.abc import Callable
from contextlib import suppress
from types import FrameType
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from w2v_common.asgi import ASGIMiddleware, create_app
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.observability import database_endpoint, redis_endpoint
from w2v_common.observability.request_context import RequestIdLogFilter

# Kept at module scope so tests can patch ``src.asgi.<name>`` (see conftest).
from src.cache.redis_client import close_redis, get_redis_client, init_redis
from src.config import get_settings
from src.database.connection import close_engine, get_engine
from src.middleware.rate_limiter import RateLimitMiddleware
from src.services.drain import JobDrain, get_job_drain

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger().addFilter(RequestIdLogFilter())
logger = logging.getLogger(__name__)

settings = get_settings()

# What ``signal.getsignal`` can hand back: a Python callable, one of the
# ``SIG_DFL``/``SIG_IGN`` constants, or None for a handler set outside Python.
_SignalHandler = Callable[[int, FrameType | None], object] | int | signal.Handlers | None


# ============================================================================
# Health Check Logging Filter (not provided by create_app)
# ============================================================================


class HealthCheckLoggingFilter(logging.Filter):
    """Suppress noisy health-check endpoint log lines."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(
            path in message
            for path in ["/health", "/health/ready", "/health/live", "/healthz", "/readyz"]
        )


uvicorn_access_logger = logging.getLogger("uvicorn.access")
uvicorn_access_logger.addFilter(HealthCheckLoggingFilter())


# ============================================================================
# Lifespan hooks
# ============================================================================


async def _startup_database(app: FastAPI) -> Any:
    """Connect to the database and ensure tables exist."""
    try:
        db_engine = await get_engine()
        app.state.db_engine = db_engine
        logger.info("Database connected: %s", database_endpoint(settings.database.url))
    except Exception as e:
        logger.error("Failed to connect to database: %s", e, exc_info=True)
        raise

    # Schema is owned by alembic -- the gitops Deployment runs
    # ``alembic upgrade head`` in an initContainer before this app
    # starts (see gitops-production
    # clusters/strato/services/w2v/deployments.yaml). Do NOT call
    # ``Base.metadata.create_all`` here: it bypasses migration history
    # and can drift the live schema away from what alembic expects.
    logger.info("Database connected (schema managed by alembic)")
    return db_engine


async def _startup_redis(app: FastAPI) -> None:
    """Connect to Redis. Failures are logged but non-fatal."""
    try:
        redis_client = await init_redis()
        await redis_client.ping()
        app.state.redis_client = get_redis_client()
        # NEVER log ``settings.redis.url`` -- a Redis URL is
        # redis://[user]:<password>@host:port/db, so the raw value writes the
        # live password to stdout on every pod start. ``redis_endpoint``
        # rebuilds host:port from urlsplit components that structurally cannot
        # carry userinfo, and fails closed with ``<unparseable>``.
        logger.info("Redis connected: %s", redis_endpoint(settings.redis.url))
    except Exception as e:
        logger.error("Failed to connect to Redis: %s", e, exc_info=True)
        logger.warning("Continuing without Redis -- caching will be disabled")


async def _startup_storage(app: FastAPI) -> None:
    """Verify the artifact object store (create local dir / check MinIO bucket).

    Non-fatal: a transient MinIO blip must not crashloop the pod — the store is
    lazily re-initialized on first artifact write. Surfacing it here verifies the
    configured backend (``STORAGE_BACKEND=minio`` in prod) at boot so a
    misconfiguration is visible in the logs immediately.
    """
    from src.api.routes import _storage

    try:
        await _storage.initialize()
        app.state.storage_service = _storage
        logger.info("Artifact storage ready (backend=%s)", _storage.backend)
    except Exception as e:
        logger.error("Failed to initialize artifact storage: %s", e, exc_info=True)
        logger.warning("Continuing — storage will be re-initialized on first artifact write")


async def _cleanup_loop_iteration(cleanup_service: Any) -> None:
    """Run a single cleanup pass against the DB.

    Advisory-lock-gated so only ONE replica reaps per tick (see
    ``CleanupService.run_cleanup_pass``); a pod that loses the lock skips quietly.
    """
    from src.database.session import get_db_session

    async with get_db_session() as db:
        outcome = await cleanup_service.run_cleanup_pass(
            db,
            stuck_timeout_minutes=settings.cleanup.stuck_job_timeout_minutes,
            never_started_timeout_hours=settings.cleanup.never_started_job_timeout_hours,
            retention_days=settings.cleanup.retention_days,
        )
    if outcome is None:
        logger.debug("Cleanup pass skipped — another replica holds the advisory lock")
        return
    stuck, never_started, old = outcome
    if stuck:
        logger.info("Cleaned up %d stuck jobs", stuck)
    if never_started:
        logger.warning(
            "Failed out %d job(s) that were accepted but NEVER started — the accepting "
            "process died between the create_job commit and mark_processing",
            never_started,
        )
    if old:
        logger.info("Cleaned up %d old jobs", old)


def _start_cleanup_task() -> asyncio.Task[None]:
    """Schedule the background cleanup loop."""
    from src.services.cleanup_service import CleanupService

    cleanup_service = CleanupService()

    async def _run_cleanup_loop() -> None:
        while True:
            try:
                await asyncio.sleep(300)
                await _cleanup_loop_iteration(cleanup_service)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Cleanup task error: %s", e, exc_info=True)

    task = asyncio.create_task(_run_cleanup_loop())
    logger.info("Cleanup background task started")
    return task


async def _on_startup(app: FastAPI) -> None:
    """create_app on_startup hook -- init DB, Redis, and the cleanup loop."""
    logger.info("=" * 70)
    logger.info("Starting Frontend Test Runner Service")
    logger.info("=" * 70)

    # FIRST, before anything that can fail: this app's own in-flight registry.
    # Everything downstream (readiness, /execute, _process_job, _on_shutdown,
    # the SIGTERM handler) reaches the drain through this attribute, so it must
    # exist before the app can serve a single request. One per app instance —
    # NOT a module singleton (see src/services/drain.py).
    drain = JobDrain()
    app.state.job_drain = drain

    # Phase D RLS defense-in-depth: re-bind the ``app.current_tenant`` GUC at the
    # start of EVERY transaction (from the identity ContextVar), so the multi-commit
    # background path (mark_processing → execute → mark_completed, each its own
    # session/transaction) stays tenant-bound after any intra-task commit. Inert
    # unless ``TENANT_RLS_MODE=enforce`` AND a tenant is in context. Idempotent.
    from w2v_common.database.rls import install_rls_tenant_rebind

    install_rls_tenant_rebind()

    await _startup_database(app)
    await _startup_redis(app)
    await _startup_storage(app)
    app.state.cleanup_task = _start_cleanup_task()
    _install_drain_signal_handlers(app, drain)

    logger.info("=" * 70)
    logger.info("Frontend Test Runner Service started successfully")
    logger.info("Environment: %s", settings.environment)
    logger.info("API: %s:%d", settings.api.host, settings.api.port)
    logger.info("=" * 70)


def _install_drain_signal_handlers(app: FastAPI, drain: JobDrain) -> None:
    """Flip readiness to 503 the INSTANT SIGTERM lands, before uvicorn drains.

    Ordering matters and the lifespan hook is too late for this half of the job:
    uvicorn's own SIGTERM handler stops accepting connections and then waits for
    the in-flight ones; ``lifespan.shutdown`` (i.e. ``_on_shutdown``) only runs
    AFTER that wait. If the readiness flip lived there, this pod would keep
    answering ``/health/ready`` with 200 for the whole drain and kubelet would
    keep it in the Service endpoints — routing brand new jobs into a process
    that is seconds from exiting.

    So we chain onto uvicorn's handler with ``signal.signal``: our handler sets
    the drain flag (pure, non-blocking) and then delegates to whatever was
    installed before us, so uvicorn's shutdown proceeds exactly as it would have.

    HOW SIGTERM REACHES THE DRAIN. ``signal.signal`` hands the handler nothing
    but ``(signum, frame)`` — no app, no request, no context. That is the only
    reason the registry was ever process-global. It does not have to be: this
    installer is called from ``_on_startup``, which HAS the app, so the drain is
    bound into the handler's closure (as a default argument, so the late-binding
    trap cannot bite) at install time. The handler is process-wide; the state it
    touches is not. On shutdown we put the previous handlers back, so a process
    that runs several lifespans (every test that uses the ``app`` fixture) does
    not accumulate a chain of handlers pointing at dead drains.
    """
    restore: dict[int, _SignalHandler] = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            previous = signal.getsignal(sig)
        except ValueError, OSError:  # pragma: no cover - not the main thread
            continue

        def _handler(
            signum: int,
            frame: FrameType | None,
            _previous: _SignalHandler = previous,
            _drain: JobDrain = drain,
        ) -> None:
            _drain.begin_drain()
            # SIG_DFL / SIG_IGN are ints (signal.Handlers is an IntEnum) and None
            # means the handler was installed outside Python — neither is callable.
            if _previous is not None and not isinstance(_previous, int):
                _previous(signum, frame)

        try:
            signal.signal(sig, _handler)
        except ValueError:  # pragma: no cover - signal handlers need the main thread
            logger.warning("Could not install drain handler for %s (not main thread)", sig)
        else:
            restore[sig] = previous
    app.state.drain_previous_signal_handlers = restore


def _restore_drain_signal_handlers(app: FastAPI) -> None:
    """Put back the handlers this app displaced, releasing its drain reference."""
    restore: dict[int, _SignalHandler] = getattr(app.state, "drain_previous_signal_handlers", {})
    for sig, previous in restore.items():
        with suppress(ValueError, OSError):  # pragma: no cover - needs a non-main thread
            signal.signal(sig, previous)
    app.state.drain_previous_signal_handlers = {}


async def _hand_back_abandoned_jobs(job_ids: list[str]) -> None:
    """Mark jobs this process could not finish as ``failed``, naming the cause.

    Left alone, an abandoned row sits at ``processing`` until the reaper rewrites
    it 30 minutes later as "Job stuck (no heartbeat — producing pod likely
    crashed)" — a diagnosis that is simply wrong and that costs the caller half
    an hour of polling. Writing the real cause here turns that into an immediate,
    correctly-named terminal failure the orchestrator sees on its next poll.

    NOT a re-queue, deliberately: there is no queue. A job is executed only by
    the process that accepted it, so resetting the row to ``pending`` would park
    it in a state nothing ever picks it up from — strictly worse than failing it.
    (``pending`` is no longer a permanent strand: ``reap_never_started_test_execution_jobs``
    now fails such rows out after ``cleanup.never_started_job_timeout_hours``. But
    that is a 6-hour backstop for rows nobody could hand back, not a substitute for
    naming the real cause here and now.) Genuine re-execution is the caller's
    decision; note that a re-submit under the SAME ``Idempotency-Key`` returns this
    failed row rather than re-running, which is the existing paid-compute dedup
    contract and out of scope to change here.
    """
    from src.database.session import get_db_session
    from src.services.job_manager import JobManager

    for job_id in job_ids:
        try:
            async with get_db_session() as db:
                await JobManager.mark_failed(
                    db,
                    job_id,
                    error_message=(
                        "Abandoned: the runner pod was terminated and the drain "
                        "timeout expired before this run finished. Re-submit with "
                        "a new Idempotency-Key to re-run."
                    ),
                )
            logger.error("[%s] marked FAILED — abandoned by shutdown drain", job_id)
        except Exception as e:
            logger.error(
                "[%s] could not mark abandoned job failed: %s — it will be reaped as "
                "'stuck' in ~%d minutes",
                job_id,
                e,
                settings.cleanup.stuck_job_timeout_minutes,
                exc_info=True,
            )


async def _drain_in_flight_jobs(drain: JobDrain) -> None:
    """Wait for in-flight test runs, then report loudly on anything abandoned."""
    drain.begin_drain()  # idempotent; covers non-signal shutdowns (tests, reload)
    pending = drain.in_flight()
    if not pending:
        logger.info("No in-flight test runs — draining immediately")
        return

    timeout = settings.shutdown.drain_timeout_seconds
    logger.info(
        "Draining %d in-flight test run(s) (timeout=%ss): %s", len(pending), timeout, pending
    )
    abandoned = await drain.wait_for_idle(timeout)
    if not abandoned:
        logger.info("Drain complete — all in-flight test runs finished")
        return

    # LOUD on purpose. Silence here is how this failure mode stayed invisible:
    # the pod exited, the row stayed 'processing', and the only signal anyone
    # ever saw was a wrong reaper message half an hour later.
    logger.error(
        "SHUTDOWN DRAIN TIMED OUT after %ss — ABANDONING %d in-flight test run(s): %s. "
        "These runs are being killed mid-execution. If this recurs, either "
        "SHUTDOWN_DRAIN_TIMEOUT_SECONDS is below EXECUTION_MAX_RUN_SECONDS (%ss) or the "
        "Deployment's terminationGracePeriodSeconds is shorter than this drain bound.",
        timeout,
        len(abandoned),
        abandoned,
        settings.execution.max_run_seconds,
    )
    await _hand_back_abandoned_jobs(abandoned)


async def _on_shutdown(app: FastAPI) -> None:
    """create_app on_shutdown hook -- drain in-flight jobs, then close Redis/DB.

    ORDER IS THE FIX. This hook used to close Redis and the DB engine straight
    away, which broke the very runs it should have been waiting for: every
    ``mark_completed`` / ``mark_failed`` / heartbeat write in ``_process_job``
    goes through that engine. Drain FIRST, tear down SECOND.
    """
    logger.info("Shutting down Frontend Test Runner Service")
    cleanup_task = getattr(app.state, "cleanup_task", None)
    if cleanup_task is not None:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task
        logger.info("Cleanup background task stopped")

    # Must precede close_redis/close_engine: in-flight jobs still need both.
    drain = get_job_drain(app)
    if drain is None:
        # Startup never got as far as creating the registry (it is the very
        # first thing _on_startup does), so there is nothing that could have
        # accepted a job. Say so rather than pretending a drain happened.
        logger.error("No job drain on app.state — startup did not complete; skipping drain")
    else:
        try:
            await _drain_in_flight_jobs(drain)
        except Exception as e:
            logger.error("Error draining in-flight jobs: %s", e, exc_info=True)
    _restore_drain_signal_handlers(app)

    try:
        await close_redis()
    except Exception as e:
        logger.error("Error closing Redis: %s", e, exc_info=True)
    try:
        await close_engine()
    except Exception as e:
        logger.error("Error closing database: %s", e, exc_info=True)
    logger.info("Frontend Test Runner Service shutdown complete")


# ============================================================================
# Create App — shared factory
# ============================================================================

# Mount routers
from src.api.health import router as health_router  # noqa: E402
from src.api.routes import router as api_router  # noqa: E402

# RateLimitMiddleware reads ``request.state.tenant_id`` (set by AuthMiddleware),
# so it MUST run INSIDE auth. create_app's ``extra_middleware`` registers
# OUTERMOST, so we disable the factory's auth registration and register both
# ourselves in order [RateLimit, Auth] → Auth becomes the outer layer, sets the
# tenant, then RateLimit reads it. Auth is gated on settings.auth.enabled, same
# as the previous hand-rolled wiring.
_extra_middleware: list[tuple[type[ASGIMiddleware], dict[str, Any]]] = [(RateLimitMiddleware, {})]
if settings.auth.enabled:
    _extra_middleware.append((AuthMiddleware, {}))
    logger.info("Authentication middleware enabled")
else:
    logger.warning("Running WITHOUT authentication -- development mode only!")

app = create_app(
    settings,
    routers=[health_router, api_router],
    title=settings.app_name,
    description="Playwright test execution microservice — runs BDD-generated tests against live sites",
    version=settings.app_version,
    on_startup=[_on_startup],
    on_shutdown=[_on_shutdown],
    auth_enabled=False,  # registered via extra_middleware to sit outside the rate limiter
    extra_middleware=_extra_middleware,
    redirect_slashes=False,
    swagger_ui_parameters={"persistAuthorization": True},
    openapi_tags=[
        {"name": "health", "description": "Liveness, readiness, dependency probes."},
        {
            "name": "test-runner",
            "description": "Playwright test execution jobs — submit, poll, results, reports, artifacts.",
        },
    ],
)


@app.get(
    "/",
    operation_id="service_root",
    tags=["health"],
    summary="Service banner + endpoint index",
)
async def root() -> dict[str, Any]:
    """Return service metadata and available endpoints."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "description": "Playwright test execution — runs BDD-generated tests against live sites",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "execute": "/api/test-runner/execute",
            "jobs": "/api/test-runner/{job_id}",
            "result": "/api/test-runner/{job_id}/result",
            "report": "/api/test-runner/{job_id}/report",
        },
        "status": "running",
        "environment": settings.environment,
    }


# ============================================================================
# Main Entry Point
# ============================================================================


def main() -> None:
    """Main entry point for running the ASGI application."""
    import uvicorn

    uvicorn.run(
        "src.asgi:app",
        host=settings.api.host,
        port=settings.api.port,
        log_level=settings.logging.level.lower(),
        reload=settings.api.reload,
        # Uvicorn's own connection drain must outlast the job drain, not cut it
        # short. A BackgroundTasks job keeps its request "in flight" as far as
        # uvicorn is concerned, so a bound shorter than the job drain would let
        # uvicorn CANCEL the run before _on_shutdown ever got the chance to wait
        # for it. +60s covers the lifespan teardown that follows.
        timeout_graceful_shutdown=settings.shutdown.drain_timeout_seconds + 60,
    )


if __name__ == "__main__":
    main()
