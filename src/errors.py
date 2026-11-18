"""Shared error-envelope helpers.

Single source of truth for the standardized error-response shape
``{error: {code, message, status, request_id, timestamp}}`` used by the
ASGI exception handlers and any middleware that emits a non-2xx
response without raising ``HTTPException`` (e.g. the rate limiter).
"""

from __future__ import annotations

import time as _time
import uuid as _uuid
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse


def request_id_from(request: Request) -> str:
    """Return the per-request id; falls back to a fresh uuid7 if unset.

    The fallback covers middlewares that run before ``RequestIDMiddleware``
    has attached ``state.request_id`` (registration order is not
    contractually fixed for non-asgi middleware stacks).
    """
    rid = getattr(request.state, "request_id", "") if hasattr(request, "state") else ""
    return rid or str(_uuid.uuid7())


def error_envelope(
    code: str,
    message: str,
    http_status: int,
    *,
    request_id: str,
    details: str | None = None,
) -> dict[str, Any]:
    """Build the standardized error envelope payload (without wrapping JSONResponse)."""
    payload: dict[str, Any] = {
        "code": code,
        "message": message,
        "status": http_status,
        "request_id": request_id,
        "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime()),
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def envelope_response(
    message: str,
    http_status: int,
    *,
    request: Request,
    code: str | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    """Return a JSONResponse carrying the standard envelope.

    Convenience wrapper for middlewares that emit a 4xx/5xx response
    without going through the HTTPException → handler path.

    ``headers`` are attached verbatim (e.g. ``Retry-After`` on a 429); the
    envelope BODY is unaffected, so callers that parse it are unchanged.
    """
    return JSONResponse(
        status_code=http_status,
        content=error_envelope(
            code or f"HTTP_{http_status}",
            message,
            http_status,
            request_id=request_id_from(request),
        ),
        headers=headers,
    )
