"""The entitlement gate on the tenant-scoped router actually refuses an
unlicensed tenant.

This service is a BILLED/metered destination — it is listed in
``w2v_common.auth.outbound.METERED_DESTINATION_SERVICES`` — yet it carried no
gate code at all. ``AuthMiddleware`` answers "who are you" (tenant IDENTITY),
never "may you" (ENTITLEMENT). A tenant whose licence was ``expired`` or
``revoked`` still presented a syntactically valid token, so every route under
``/api/test-runner`` served them, including ``POST /execute``, which spends real
browser/CPU minutes and is metered. ``LICENSE_GATE`` is mounted router-wide on
``src/api/routes.py`` to close that.

Every test here drives a REAL HTTP request through the REAL routers, because
the interesting question is not whether ``require_license`` works — w2v_common
owns that — but whether THIS service's routes actually carry it. Static
introspection of ``route.dependant.dependencies`` is NOT an acceptable
substitute: this FastAPI build includes sub-routers lazily (``_IncludedRouter``)
and applies a parent router's ``dependencies=`` when a request is matched, not
at import time, so introspection can report a live-gated route as ungated — a
false negative. A request is the only honest probe.

The paired 200s (entitled tenant, and flag-off) are what stop this file from
passing vacuously: if the gate were mounted wrongly, or the fixtures were
broken, those cases would not be 200s either.

This module deliberately does NOT use the shared ``app``/``client`` fixtures
from ``tests/conftest.py``: those bind a live postgres engine, and the gate
runs strictly before any handler body, so a DB is irrelevant to what is under
test here. ``get_db_session`` and ``JobManager.get_job`` are patched instead,
which makes the gated handler's own work free — a non-200 can then only be the
gate, which is the whole point.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.middleware.base import BaseHTTPMiddleware
from w2v_common.auth import clear_license_status_cache
from w2v_common.auth.licenses_read_client import LicensesClientError

from src.api.health import router as health_router
from src.api.routes import router as api_router
from src.database.models import TestExecutionJob
from src.services.drain import JobDrain
from tests.conftest import TEST_TENANT_ID

SUBJECT = "user-under-test"

JOB_ID = "job-under-test"

#: The cheapest genuinely tenant-scoped route on the gated router: with
#: ``JobManager.get_job`` faked it touches no DB, no Redis and no object store,
#: so the only thing that can turn its 200 into something else is the gate.
GATED_PATH = f"/api/test-runner/{JOB_ID}"

#: The metered route that motivated this change. It is asserted separately so a
#: future refactor that moves ``/execute`` onto its own router cannot silently
#: drop the gate from the one endpoint that actually spends money.
GATED_EXECUTE_PATH = "/api/test-runner/execute"

#: A probe. Must answer 200 for a revoked tenant: a 402/503 on a liveness probe
#: CrashLoops the pod, and an unlicensed tenant cannot be billed by killing our
#: own deployment.
PROBE_PATH = "/health/live"

#: The readiness probe the fleet actually polls. Same argument, and this one is
#: the path that pulls the replica out of the Service endpoints when it fails.
READY_PROBE_PATH = "/health/ready"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Record:
    """A licences wire record. Only ``model_dump`` is on the gate's path."""

    def __init__(self, **fields: Any) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, Any]:
        return dict(self._fields)


class _Detail(_Record):
    """An ACTIVE licence detail (the 200 from ``GET /licenses/active``)."""

    conflict = False


class _FakeLicenses:
    """Stands in for ``LicensesReader``.

    ``active`` None models the authoritative 404 ("no active licence"), which
    sends the gate to the subject inventory to refine none/expired/revoked.
    ``error`` models an upstream failure, which must 503, never allow.
    """

    def __init__(
        self,
        *,
        active: _Detail | None = None,
        inventory: list[_Record] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._active = active
        self._inventory = inventory or []
        self._error = error

    async def get_active_license(self, *, tenant_id: str, app_key: str) -> _Detail | None:
        if self._error is not None:
            raise self._error
        return self._active

    async def list_user_licenses(self, subject_id: str, *, tenant_id: str) -> list[_Record]:
        if self._error is not None:
            raise self._error
        return list(self._inventory)


def _revoked() -> list[_Record]:
    return [_Record(app={"key": "w2v"}, tenant_id=TEST_TENANT_ID, status="revoked")]


def _expired() -> list[_Record]:
    return [
        _Record(
            app={"key": "w2v"},
            tenant_id=TEST_TENANT_ID,
            status="expired",
            expires_at="2020-01-01T00:00:00+00:00",
        )
    ]


def _entitled() -> _Detail:
    return _Detail(
        app={"key": "w2v"},
        tenant_id=TEST_TENANT_ID,
        status="active",
        expires_at="2099-01-01T00:00:00+00:00",
        plan={"code": "pro"},
    )


@asynccontextmanager
async def _fake_db_session():
    """A stand-in for ``get_db_session`` — nothing here reaches postgres."""
    yield MagicMock(name="fake_db_session")


def _job() -> TestExecutionJob:
    """A completed job owned by the test tenant, built in memory."""
    return TestExecutionJob(
        job_id=JOB_ID,
        tenant_id=TEST_TENANT_ID,
        status="completed",
        target_url="https://example.com",
        error_message=None,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_cached_verdicts():
    """The gate caches its verdict per (tenant, app, subject) for 60s. Two tests
    in one process would otherwise share a verdict and the second would assert
    nothing."""
    clear_license_status_cache()
    yield
    clear_license_status_cache()


@pytest.fixture(autouse=True)
def _gate_disarmed(monkeypatch):
    """Default the flag OFF, exactly as it ships. Each test arms it explicitly."""
    monkeypatch.delenv("W2V_LICENSE_GATE_ENABLED", raising=False)


@pytest.fixture(autouse=True)
def _free_handler_work():
    """Make the gated handler's own body free: no DB, no job lookup."""
    with (
        patch("src.api.routes.get_db_session", _fake_db_session),
        patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=_job())),
    ):
        yield


@pytest.fixture
def app() -> FastAPI:
    """The REAL routers, with identity stamped the way AuthMiddleware stamps it."""
    application = FastAPI()

    class _StampIdentity(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            request.state.tenant_id = TEST_TENANT_ID
            request.state.subject_id = SUBJECT
            return await call_next(request)

    application.add_middleware(_StampIdentity)
    application.include_router(health_router)
    application.include_router(api_router)
    # Readiness fails CLOSED with no in-flight registry (that means the lifespan
    # never ran), so the stand-in app is wired like the real one.
    application.state.job_drain = JobDrain()
    return application


async def _get(app: FastAPI, path: str) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.get(path)


async def _post(app: FastAPI, path: str, json: dict[str, Any]) -> Any:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        return await client.post(path, json=json)


def _licenses(monkeypatch, fake: _FakeLicenses) -> None:
    """Patch the client the gate resolves through, in the module that calls it."""
    monkeypatch.setattr(
        "w2v_common.auth.entitlement.get_licenses_read_client", lambda *a, **k: fake
    )


# ---------------------------------------------------------------------------
# Armed: an unentitled tenant is refused
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("inventory", "code"),
    [
        ([], "license_none"),
        (_revoked(), "license_revoked"),
        (_expired(), "license_expired"),
    ],
    ids=["none", "revoked", "expired"],
)
async def test_armed_gate_402s_an_unentitled_tenant(app, monkeypatch, inventory, code):
    """402 with the state-specific code — the tenant's licence is the problem."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=inventory))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == code


async def test_armed_gate_402s_the_metered_execute_route(app, monkeypatch):
    """``POST /execute`` is the route that actually spends browser/CPU minutes
    and is metered downstream. A revoked tenant must never reach it — and it
    must be refused BEFORE the request body is even validated, so a 422 here
    would be a failure, not a pass."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _post(app, GATED_EXECUTE_PATH, {})

    assert response.status_code == 402, response.text
    assert response.json()["detail"]["error"] == "license_revoked"


async def test_armed_gate_503s_when_entitlement_cannot_be_established(app, monkeypatch):
    """An upstream failure is NOT an allow, and NOT a 402 either: we do not know."""
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(error=LicensesClientError("licenses unreachable")))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 503, response.text
    assert response.json()["detail"]["error"] == "license_unavailable"
    assert response.headers["Retry-After"]


# ---------------------------------------------------------------------------
# Armed: an entitled tenant is served (the non-vacuity pair)
# ---------------------------------------------------------------------------


async def test_armed_gate_serves_an_entitled_tenant(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=_entitled()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text
    assert response.json()["job_id"] == JOB_ID


# ---------------------------------------------------------------------------
# Dormant on merge
# ---------------------------------------------------------------------------


async def test_gate_is_dormant_while_the_flag_is_unset(app, monkeypatch):
    """Merging this changes NOTHING until gitops arms the flag. A revoked tenant
    is still served — deliberately — so the rollout is a flag flip, not a
    deploy."""
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


async def test_gate_is_dormant_when_the_flag_is_explicitly_false(app, monkeypatch):
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "false")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, GATED_PATH)

    assert response.status_code == 200, response.text


# ---------------------------------------------------------------------------
# Probes are never gated
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("probe", [PROBE_PATH, READY_PROBE_PATH], ids=["live", "ready"])
async def test_probes_are_served_for_a_revoked_tenant(app, monkeypatch, probe):
    """A gated probe path CrashLoops the pod. The fleet polls ``/health/live``
    and ``/health/ready``; neither may ever answer 402/503-because-licence.

    ``/health/ready`` is allowed to be 503 for a DEPENDENCY reason (that is its
    contract), so this asserts only that the answer is not the gate's refusal.
    """
    monkeypatch.setenv("W2V_LICENSE_GATE_ENABLED", "true")
    _licenses(monkeypatch, _FakeLicenses(active=None, inventory=_revoked()))

    response = await _get(app, probe)

    assert response.status_code != 402, response.text
    body = response.json()
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, dict):
        assert detail.get("error") != "license_unavailable", response.text


async def test_the_services_own_probe_paths_are_exempt_by_path():
    """Belt and braces: even if a future refactor mounted the gate on the health
    router, w2v_common would still refuse to enforce on these paths. This is the
    guard against hand-rolling an exemption list in this repo — we do not own
    one, and must not."""
    from w2v_common.auth import is_license_exempt_path

    for path in ("/health", "/health/live", "/health/ready", "/healthz", "/readyz"):
        assert is_license_exempt_path(path), path
    assert not is_license_exempt_path(GATED_PATH)
    assert not is_license_exempt_path(GATED_EXECUTE_PATH)
