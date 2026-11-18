"""P7: a background test run's in-cell calls carry a FRESH cell token.

``POST /execute`` hands the run to a ``BackgroundTasks`` task that downloads the
package from ``bdd_playwright_service`` (in-cell) and may run for many minutes.
w2v_common's ``identity_event_hook`` mints an ``X-W2V-Cell-Token`` per send, but
only from what the task's context holds: a tenant with VOUCHED provenance, plus
run / project / actor. These tests drive the real route behind the real
``AuthMiddleware`` cell lane and assert what reaches the wire — and that none of
it depends on the forwarded licence (the cell lane carries none).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import uuid
from collections.abc import AsyncGenerator, Generator, Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi import FastAPI
from jwt.algorithms import OKPAlgorithm
from w2v_common.auth import cell_jwt, clear_identity_context
from w2v_common.auth.cell_jwt import (
    CELL_TOKEN_HEADER,
    CellClaims,
    mint_cell_token,
    reset_cell_state,
    verify_cell_token,
)
from w2v_common.auth.identity_context import set_identity_context
from w2v_common.auth.middleware import AuthMiddleware
from w2v_common.discovery import service_url
from w2v_common.tenant import MissingTenantError, TenantSource

from src.api import routes
from src.api.schemas import TestExecutionRequest
from src.services.drain import JobDrain
from src.services.execution_engine import ExecutionEngine
from src.services.task_identity import TaskIdentity, capture_task_identity

ORCH = "w2v-orchestrator"
SELF = "frontend-test-runner-service"
CALLEE = "bdd-playwright-service"
DEPLOYMENT = "strato-w2v-test"
TENANT = uuid.UUID("0198f1d2-0000-7000-8000-0000000c3111")
ACTOR = "0198f1d2-0000-7000-8000-0000000ac701"
PROJECT = "0198f1d2-0000-7000-8000-0000000b0001"
RUN = "0198f1d2-0000-7000-8000-00000000a0a1"
SUBJECT = "0198f1d2-0000-7000-8000-0000000c5501"
LICENSE = "0198f1d2-0000-7000-8000-0000000d1c01"
#: What the originating request is attributed to. On w2v_common 0.92.x the cell
#: lane takes these from the forwarded headers; from 0.93.0 they are SIGNED
#: claims of the inbound token and the outbound token must re-sign them.
_ATTRIBUTION_HEADERS = {"X-Subject-Id": SUBJECT, "X-License-Id": LICENSE}
_SIGNS_ATTRIBUTION = "license_id" in inspect.signature(mint_cell_token).parameters
_RESULT = {
    "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "flaky": 0, "pass_rate": 0},
    "test_results": [],
}


def _jwk(private: Ed25519PrivateKey, kid: str, iss: str) -> dict[str, Any]:
    entry = json.loads(OKPAlgorithm.to_jwk(private.public_key()))
    entry.update({"kid": kid, "iss": iss})
    return entry


def _pem(private: Ed25519PrivateKey) -> str:
    return private.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _sign_as(monkeypatch: pytest.MonkeyPatch, key: Ed25519PrivateKey, kid: str, iss: str) -> None:
    monkeypatch.setenv("CELL_JWT_SIGNING_KEY", _pem(key))
    monkeypatch.setenv("CELL_JWT_KID", kid)
    monkeypatch.setenv("CELL_SERVICE_NAME", iss)


@pytest.fixture
def keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, Ed25519PrivateKey]]:
    """Orchestrator + this service keys in a JWKS file; this process signs as ftr."""
    orch, own = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
    jwks = tmp_path / "jwks.json"
    jwks.write_text(json.dumps({"keys": [_jwk(orch, "orch-1", ORCH), _jwk(own, "ftr-1", SELF)]}))
    for name in ("W2V_LOCAL", "CELL_JWKS_RELOAD_SECONDS", "W2V_LICENSE_GATE_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CELL_JWKS_PATH", str(jwks))
    monkeypatch.setenv("CELL_DEPLOYMENT_ID", DEPLOYMENT)
    monkeypatch.setenv("CELL_ALLOWED_CALLERS", "*")
    monkeypatch.setenv("CELL_AUTH_MODE", "enforce")
    _sign_as(monkeypatch, own, "ftr-1", SELF)
    reset_cell_state()
    clear_identity_context()
    yield {"orch": orch, "own": own}
    clear_identity_context()
    reset_cell_state()


def _orchestrator_token(monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey]) -> str:
    _sign_as(monkeypatch, keys["orch"], "orch-1", ORCH)
    signed: dict[str, Any] = (
        {"subject_id": SUBJECT, "license_id": LICENSE} if _SIGNS_ATTRIBUTION else {}
    )
    token = mint_cell_token(SELF, tenant=TENANT, run=RUN, actor=ACTOR, project=PROJECT, **signed)
    _sign_as(monkeypatch, keys["own"], "ftr-1", SELF)
    return token


def _verify_at_callee(monkeypatch: pytest.MonkeyPatch, token: str, now: float) -> CellClaims:
    """Verify as the CALLEE would: aud must be the callee's own cell name."""
    with monkeypatch.context() as m:
        m.setenv("CELL_SERVICE_NAME", CALLEE)
        return verify_cell_token(token, now=now)


@contextlib.asynccontextmanager
async def _fake_db_session() -> AsyncGenerator[MagicMock]:
    yield MagicMock()


def _recording_client(sent: list[httpx.Request]) -> type[httpx.AsyncClient]:
    """``httpx.AsyncClient`` that keeps every hook but answers from a recorder."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"PK\x05\x06" + b"\x00" * 18)

    class _Recording(httpx.AsyncClient):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["transport"] = httpx.MockTransport(handler)
            super().__init__(*args, **kwargs)

    return _Recording


async def _download_from_bdd(tmp_path: Path, license_token: str | None) -> None:
    """The run's real in-cell call: ``ExecutionEngine._download_zip`` against bdd."""
    url = f"{service_url('bdd_playwright')}/api/bdd/bdd-1/download"
    await ExecutionEngine()._download_zip(
        url, tmp_path, allow_private=True, license_token=license_token
    )


@contextlib.contextmanager
def _job_plumbing_patched() -> Generator[None]:
    with (
        patch("src.api.routes.get_db_session", _fake_db_session),
        patch("src.api.routes._register_concurrent_job", AsyncMock(return_value=False)),
        patch("src.api.routes.JobManager.mark_processing", new_callable=AsyncMock),
        patch("src.api.routes.JobManager.mark_completed", new_callable=AsyncMock),
        patch("src.api.routes.JobManager.mark_failed", new_callable=AsyncMock) as mark_failed,
    ):
        yield
    assert not mark_failed.await_count, mark_failed.await_args


async def test_background_run_outbound_carries_cell_token_for_the_originating_request(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey], tmp_path: Path
) -> None:
    """Inbound cell token → POST /execute → background run → bdd download: same identity."""
    sent: list[httpx.Request] = []
    seen: dict[str, Any] = {}

    async def execute(_request: TestExecutionRequest, **kwargs: Any) -> dict[str, Any]:
        seen["license_token"] = kwargs.get("license_token")
        await _download_from_bdd(tmp_path, kwargs.get("license_token"))
        return _RESULT

    created = MagicMock(job_id="job-1", status="pending")
    app = FastAPI()
    app.add_middleware(AuthMiddleware)
    app.include_router(routes.router)
    app.state.job_drain = JobDrain()

    inbound = _orchestrator_token(monkeypatch, keys)
    with (
        _job_plumbing_patched(),
        patch("src.api.routes.JobManager.create_job", AsyncMock(return_value=(created, True))),
        patch("src.api.routes._engine.execute", AsyncMock(side_effect=execute)),
    ):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://ftr") as client:
            # Only clients built from here on (the run's download) are recorded.
            monkeypatch.setattr(httpx, "AsyncClient", _recording_client(sent))
            resp = await client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-1", "target_url": "https://example.com"},
                headers={CELL_TOKEN_HEADER: inbound, **_ATTRIBUTION_HEADERS},
            )
    assert resp.status_code == 200, resp.text

    # The cell lane forwards no licence: the cell binding must stand on its own.
    assert seen["license_token"] is None
    assert len(sent) == 1
    token = sent[0].headers.get(CELL_TOKEN_HEADER)
    assert token, "the background run's in-cell call went out with no cell token (enforce → 401)"
    assert token != inbound, "the inbound token was replayed instead of a fresh mint"
    claims = _verify_at_callee(monkeypatch, token, now=cell_jwt.time.time())
    assert claims.issuer == SELF
    assert claims.audience == CALLEE
    assert claims.tenant_id == TENANT
    assert claims.run == RUN
    assert claims.project == PROJECT
    assert claims.actor == ACTOR, "the human actor was dropped on the way into the run"
    # Who pays: the request's licence + subject reach the wire (headers on every
    # version; signed claims from 0.93.0), never NULL — a dropped licence is an
    # unattributable cost once receivers enforce.
    assert sent[0].headers.get("X-License-Id") == LICENSE
    assert sent[0].headers.get("X-Subject-Id") == SUBJECT
    if _SIGNS_ATTRIBUTION:
        assert getattr(claims, "license_id", None) == LICENSE
        assert getattr(claims, "subject", None) == SUBJECT


async def test_run_past_token_ttl_mints_fresh_per_call(
    monkeypatch: pytest.MonkeyPatch, keys: dict[str, Ed25519PrivateKey], tmp_path: Path
) -> None:
    """A run still going 61 s later mints a NEW token; nothing minted at start is held."""
    clock = {"now": 1_900_000_000.0}
    monkeypatch.setattr(cell_jwt, "time", SimpleNamespace(time=lambda: clock["now"]))
    sent: list[httpx.Request] = []
    monkeypatch.setattr(httpx, "AsyncClient", _recording_client(sent))

    async def execute(_request: TestExecutionRequest, **_kwargs: Any) -> dict[str, Any]:
        await _download_from_bdd(tmp_path, None)
        clock["now"] += cell_jwt.CELL_TOKEN_TTL_SECONDS + 1
        await _download_from_bdd(tmp_path, None)
        return _RESULT

    identity = TaskIdentity(
        tenant_id=TENANT,
        tenant_source=TenantSource.CELL_JWT,
        run=RUN,
        project=PROJECT,
        actor=ACTOR,
        subject_id=SUBJECT,
        license_id=LICENSE,
    )
    with (
        _job_plumbing_patched(),
        patch("src.api.routes._engine.execute", AsyncMock(side_effect=execute)),
    ):
        await routes._process_job(
            "job-1",
            TestExecutionRequest(bdd_job_id="bdd-1", target_url="https://example.com"),
            str(TENANT),
            drain=JobDrain(),
            identity=identity,
        )

    first, second = (r.headers[CELL_TOKEN_HEADER] for r in sent)
    assert first != second
    # Verified a little later still (past the callee's clock-skew allowance): the
    # token minted at run start is dead, the one minted on the late call is live.
    verify_at = clock["now"] + 10
    late = _verify_at_callee(monkeypatch, second, now=verify_at)
    assert late.tenant_id == TENANT
    assert late.run == RUN
    assert late.issued_at == int(clock["now"])
    with pytest.raises(cell_jwt.CellTokenError):
        _verify_at_callee(monkeypatch, first, now=verify_at)


def test_capture_without_a_bound_tenant_fails_loud(keys: dict[str, Ed25519PrivateKey]) -> None:
    """No tenant in request scope → refuse before the run is scheduled."""
    clear_identity_context()
    with pytest.raises(MissingTenantError):
        capture_task_identity(str(TENANT))


def test_capture_refuses_a_tenant_that_is_not_the_authenticated_one(
    keys: dict[str, Ed25519PrivateKey],
) -> None:
    other = "0198f1d2-0000-7000-8000-0000000c3222"
    set_identity_context(TENANT, None, None, tenant_source=TenantSource.CELL_JWT)
    with pytest.raises(MissingTenantError):
        capture_task_identity(other)
