"""Guardrail tests for the auth middleware integration.

frontend_test_runner_service uses the shared
``w2v_common.auth.middleware.AuthMiddleware``, wired via
``w2v_common.asgi.create_app``. Because the per-tenant RateLimitMiddleware
must run inside auth, asgi.py registers AuthMiddleware through the factory's
``extra_middleware`` (gated on ``settings.auth.enabled``) rather than the
factory's own auth path. These tests exercise that the middleware is **wired**
on the asgi app and that it behaves correctly for the standard public/protected
token matrix.

The intent is to catch the "false-merge" pattern: a PR that says it added auth
but stopped wiring it (or registered it unconditionally / behind an off flag).
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

import httpx
import pytest

# Re-export from the shared module — same import path the asgi entry uses.
import w2v_common.auth.middleware as w2v_auth
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from w2v_common.auth.middleware import AuthMiddleware

from tests.conftest import TEST_TENANT_ID


@pytest.fixture(autouse=True)
def _clear_license_cache() -> None:
    cache = w2v_auth._get_license_cache()
    cache.clear()


def _make_minimal_app() -> FastAPI:
    """Tiny app for behaviour tests — independent of asgi lifespan side-effects."""
    app = FastAPI()
    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    async def _health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/test-runner/echo")
    async def _echo(request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "tenant_id": getattr(request.state, "tenant_id", None),
                "license_present": getattr(request.state, "license", None) is not None,
            }
        )

    return app


@pytest.fixture
async def client() -> AsyncGenerator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Wiring check — the regression-against-false-merge assertion
# ---------------------------------------------------------------------------


def test_auth_middleware_is_actually_registered_on_asgi_app() -> None:
    """The asgi entry must wire the shared AuthMiddleware, gated on the setting.

    asgi.py adopts ``w2v_common.asgi.create_app`` and — because the per-tenant
    RateLimitMiddleware must run INSIDE auth — registers AuthMiddleware via
    ``extra_middleware`` (with ``auth_enabled=False``) rather than the factory's
    own auth path. Tests run with ``AUTH_ENABLED=false`` (conftest) for
    fixture-built apps, so the wiring is asserted at the source level to catch
    the false-merge pattern (auth dropped, or registered unconditionally).
    """
    asgi_src = (
        __import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "asgi.py"
    ).read_text()
    assert "create_app(" in asgi_src, "asgi.py must build the app via create_app"
    assert "if settings.auth.enabled:" in asgi_src, (
        "AuthMiddleware registration must stay gated on settings.auth.enabled"
    )
    assert "(AuthMiddleware, {})" in asgi_src, (
        "asgi.py must register the shared AuthMiddleware via extra_middleware"
    )


# ---------------------------------------------------------------------------
# Public bypass
# ---------------------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Protected paths
# ---------------------------------------------------------------------------


async def test_protected_route_with_valid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    introspect_payload: dict[str, Any] = {
        "active": True,
        "tenant_id": TEST_TENANT_ID,
        "subject_id": "subj-1",
        "license_id": "lic-1",
        "roles": ["user"],
    }

    async def _fake(token: str) -> dict[str, Any]:
        assert token == "abc.def.ghi"
        return introspect_payload

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/test-runner/echo",
            headers={"X-Tenant-Id": "Bearer abc.def.ghi"},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["tenant_id"] == TEST_TENANT_ID
    assert body["license_present"] is True


async def test_protected_route_with_invalid_license_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake(_token: str) -> dict[str, Any]:
        return {"active": False, "reason": "expired"}

    monkeypatch.setattr(w2v_auth, "_introspect_license", _fake)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get(
            "/api/test-runner/echo",
            headers={"X-Tenant-Id": "Bearer dud-token"},
        )

    assert resp.status_code == 401
    body = resp.json()
    assert body["error"]["code"] == "INVALID_TOKEN"
    assert body["error"]["status"] == 401
    assert "Invalid or expired license token" in body["error"]["message"]


async def test_protected_route_no_token_untrusted_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Untrusted source IP + no token must 401, regardless of TRUST_INCLUSTER_IDENTITY."""
    # Force the trust-incluster path off so the test is deterministic.
    monkeypatch.setattr(w2v_auth, "_trust_incluster_identity", lambda: False)

    transport = httpx.ASGITransport(app=_make_minimal_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        resp = await ac.get("/api/test-runner/echo")
    # ASGITransport peer = 'testclient' which is NOT in the trusted-CIDR list,
    # so this should hit the 401 branch.
    assert resp.status_code == 401
