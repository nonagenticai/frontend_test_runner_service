"""Health check endpoints."""

from __future__ import annotations

import inspect
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from redis.exceptions import RedisError
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(tags=["health"])

# The library exception trees below are NOT optional and must never be
# "simplified" away. None of these inherit from the builtins whose names they
# shadow::
#
#     sqlalchemy.exc.OperationalError.__mro__
#         -> DatabaseError -> DBAPIError -> StatementError -> SQLAlchemyError
#            -> HasDescriptionCode -> Exception
#     issubclass(sqlalchemy.exc.OperationalError, OSError | RuntimeError) -> False
#     issubclass(redis.exceptions.RedisError,     OSError)                -> False
#
# A ``psycopg.OperationalError`` (the real "postgres is unreachable" fault)
# surfaces as ``sqlalchemy.exc.OperationalError``, so a builtins-only
# ``except (OSError, RuntimeError)`` NEVER fired: the exception escaped to ASGI
# and FastAPI turned it into a 500. On /health/ready (and its /readyz alias)
# that breaks the Kubernetes readiness contract — a transient DB fault must be
# a 503 ("not ready"), never a 500 ("the app is broken").
#
# Probe failure = the dependency is down, not the process. The builtins stay in
# the tuples because a socket-level error can still surface as one.
_DB_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, SQLAlchemyError)
_REDIS_PROBE_ERRORS = (OSError, ConnectionError, TimeoutError, RuntimeError, RedisError)


@router.get("/health", operation_id="health_check")
async def health() -> dict[str, Any]:
    """Liveness probe."""
    return {"status": "ok", "service": "frontend_test_runner_service"}


@router.get("/health/ready", response_model=None, operation_id="health_ready")
async def readiness(request: Request) -> dict[str, Any] | JSONResponse:
    """Readiness probe — checks DB and Redis.

    Fails CLOSED: if any dependency is down the response carries HTTP 503 so
    kubelet actually pulls this replica out of the Service endpoints. This
    service runs at ``replicas: 1`` and EXECUTES BDD runs, so a pod that
    cannot reach postgres/redis must stop receiving traffic rather than
    accept work it can never persist. The ``checks`` detail is returned in
    both the ready and the not-ready body.
    """
    checks: dict[str, str] = {}

    # Drain gate — FIRST, and short-circuiting. Once SIGTERM has landed this pod
    # must leave the Service endpoints immediately so no NEW job is routed to a
    # process that is about to stop. Reported before the dependency probes
    # because DB/Redis are still perfectly healthy during a drain: probing them
    # would return "ready" and keep sending work into the kill window.
    #
    # The registry is per-app state (``app.state.job_drain``), reached through
    # the request — NOT a module singleton. A missing one means the lifespan has
    # not run, which is "not ready", not "not draining": fail CLOSED so a wiring
    # regression shows up as a red probe instead of silently deleting this gate.
    from src.services.drain import get_job_drain

    drain = get_job_drain(request.app)
    if drain is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "checks": {"drain": "lifespan has not initialised the in-flight registry"},
            },
        )
    if drain.draining:
        return JSONResponse(
            status_code=503,
            content={
                "status": "draining",
                "checks": {"drain": "shutting down — not accepting new work"},
                "in_flight_jobs": drain.in_flight(),
            },
        )

    try:
        from src.database.connection import health_check

        await health_check()
        checks["database"] = "ok"
    except _DB_PROBE_ERRORS as e:
        checks["database"] = f"error: {e}"

    try:
        from src.cache.redis_client import get_redis_client

        redis = get_redis_client()
        client = redis._ensure_connected()
        pong = client.ping()
        if inspect.isawaitable(pong):
            await pong
        checks["redis"] = "ok"
    except _REDIS_PROBE_ERRORS as e:
        checks["redis"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    if not all_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "checks": checks},
        )
    return {"status": "ready", "checks": checks}


@router.get("/health/live", operation_id="health_live")
async def liveness() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "alive"}


@router.get("/healthz", operation_id="health.live")
async def healthz() -> dict[str, str]:
    """Modern K8s liveness probe path (alias of /health/live)."""
    return await liveness()


@router.get("/readyz", response_model=None, operation_id="health.ready")
async def readyz(request: Request) -> dict[str, Any] | JSONResponse:
    """Modern K8s readiness probe path (alias of /health/ready)."""
    return await readiness(request)
