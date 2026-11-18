"""Tests for API routes."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI

from src.database.models import TestExecutionJob
from src.services.drain import JobDrain
from tests.conftest import OTHER_TENANT_ID, TEST_LICENSE_TOKEN, TEST_TENANT_ID


@pytest.fixture
async def health_client():
    """HTTP client over a bare app mounting ONLY the health router.

    The shared `client` fixture drags in a live postgres engine, which the
    probe endpoints must never require -- readiness has to be able to answer
    (with 503) precisely when the database is unreachable. Mounting the router
    standalone keeps these tests honest and runnable without a DB.
    """
    from src.api.health import router

    app = FastAPI()
    app.include_router(router)
    # Readiness fails CLOSED when the app has no in-flight registry (that means
    # the lifespan never ran), so this stand-in app has to be wired like the
    # real one — one JobDrain on app.state, no module singleton.
    app.state.job_drain = JobDrain()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@asynccontextmanager
async def _fake_db_session():
    """Async context manager yielding a mock AsyncSession for background-task tests."""
    yield MagicMock(name="fake_db_session")


def _make_job(
    *,
    job_id: str = "job-1",
    status: str = "completed",
    summary: dict | None = None,
    test_results: list | None = None,
    report_url: str | None = "/reports/job-1",
    artifact_paths: dict | None = None,
    error_message: str | None = None,
    execution_time_seconds: float | None = 12.5,
    config: dict | None = None,
    target_url: str = "https://example.com",
) -> TestExecutionJob:
    """Build an in-memory TestExecutionJob without touching the DB."""
    return TestExecutionJob(
        job_id=job_id,
        tenant_id=TEST_TENANT_ID,
        status=status,
        target_url=target_url,
        summary=summary
        or {"total": 2, "passed": 2, "failed": 0, "skipped": 0, "flaky": 0, "pass_rate": 100.0},
        test_results=test_results
        if test_results is not None
        else [
            {"name": "01950000-0000-7000-8000-000000000001", "status": "passed", "duration_ms": 100}
        ],
        report_url=report_url,
        artifact_paths=artifact_paths or {},
        error_message=error_message,
        execution_time_seconds=execution_time_seconds,
        config=config or {},
    )


class TestHealthEndpoints:
    async def test_health(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["service"] == "frontend_test_runner_service"

    async def test_liveness(self, client):
        resp = await client.get("/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    @staticmethod
    def _patch_probes(*, db_ok: bool, redis_ok: bool):
        """Patch the two readiness dependencies at their import sites.

        ``readiness()`` imports both lazily inside the handler, so patching the
        re-export modules is what the running probe actually resolves.
        """
        db = AsyncMock(return_value=True) if db_ok else AsyncMock(side_effect=OSError("db down"))

        def _redis_factory():
            if not redis_ok:
                raise OSError("redis down")
            client_ = MagicMock()
            client_.ping.return_value = True
            redis = MagicMock()
            redis._ensure_connected.return_value = client_
            return redis

        return (
            patch("src.database.connection.health_check", db),
            patch("src.cache.redis_client.get_redis_client", side_effect=_redis_factory),
        )

    @pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
    async def test_readiness_ok_returns_200_ready(self, health_client, path):
        """All dependencies up -> 200 with status ``ready`` and both checks."""
        db_patch, redis_patch = self._patch_probes(db_ok=True, redis_ok=True)
        with db_patch, redis_patch:
            resp = await health_client.get(path)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["checks"] == {"database": "ok", "redis": "ok"}

    @pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
    @pytest.mark.parametrize(("db_ok", "redis_ok"), [(False, True), (True, False), (False, False)])
    async def test_readiness_failure_returns_503_and_keeps_checks(
        self, health_client, path, db_ok, redis_ok
    ):
        """Any dependency down must FAIL CLOSED with 503 -- a 200 here would keep
        this replicas:1 BDD executor in the Service endpoints while it cannot
        persist a single result. The ``checks`` detail is preserved in the body.
        """
        db_patch, redis_patch = self._patch_probes(db_ok=db_ok, redis_ok=redis_ok)
        with db_patch, redis_patch:
            resp = await health_client.get(path)
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert set(data["checks"].keys()) == {"database", "redis"}
        assert data["checks"]["database"] == ("ok" if db_ok else "error: db down")
        assert data["checks"]["redis"] == ("ok" if redis_ok else "error: redis down")

    @pytest.mark.parametrize("path", ["/health/ready", "/readyz"])
    async def test_readiness_sqlalchemy_operational_error_is_503_not_500(self, health_client, path):
        """The REAL "postgres is unreachable" fault.

        ``psycopg.OperationalError`` surfaces as ``sqlalchemy.exc.OperationalError``,
        whose MRO is ``... -> SQLAlchemyError -> Exception`` -- it is NOT an
        ``OSError``/``RuntimeError``. With a builtins-only except tuple the
        exception escaped the handler to ASGI and FastAPI answered **500**, which
        tells kubelet "the app is broken" instead of "not ready".
        """
        from sqlalchemy.exc import OperationalError

        assert not issubclass(OperationalError, OSError | RuntimeError)

        db = AsyncMock(side_effect=OperationalError("SELECT 1", None, Exception("db down")))
        _, redis_patch = self._patch_probes(db_ok=True, redis_ok=True)
        with patch("src.database.connection.health_check", db), redis_patch:
            resp = await health_client.get(path)

        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["checks"]["database"].startswith("error:")
        assert data["checks"]["redis"] == "ok"

    @pytest.mark.parametrize("path", ["/healthz", "/health/live"])
    async def test_liveness_stays_unconditional_when_deps_are_down(self, health_client, path):
        """Liveness must NOT fail closed -- restarting the pod does not fix a
        down postgres/redis, it only loses in-flight BDD runs."""
        db_patch, redis_patch = self._patch_probes(db_ok=False, redis_ok=False)
        with db_patch, redis_patch:
            resp = await health_client.get(path)
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    async def test_root(self, client):
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert "Frontend Test Runner" in data["name"]
        assert "endpoints" in data


class TestAuthEnforcement:
    """Verify the prod fail-closed default (w2v_common >= 0.6.9) is enforced.

    When no `X-Tenant-Id: Bearer <token>` header is sent and the
    in-cluster trusted-network fallback is disabled (the prod default),
    protected routes must return 401. This guards against silently
    re-opening the legacy fail-open path in tests or in production.
    """

    async def test_protected_route_rejects_missing_token_when_fail_closed(self, unauth_client):
        # No auth header, trusted-network fallback off (prod default).
        # Must NOT fall through to a 422 / 200 — middleware must 401.
        resp = await unauth_client.post(
            "/api/test-runner/execute",
            json={"bdd_job_id": "bdd-1", "target_url": "https://example.com"},
        )
        assert resp.status_code == 401, resp.text
        assert "Missing authentication token" in resp.text

    async def test_protected_route_rejects_invalid_license_token(self, unauth_client):
        # A bogus license token must introspect as inactive -> 401.
        resp = await unauth_client.post(
            "/api/test-runner/execute",
            headers={"X-Tenant-Id": "Bearer not-a-real-token"},
            json={"bdd_job_id": "bdd-1", "target_url": "https://example.com"},
        )
        assert resp.status_code == 401, resp.text
        assert "Invalid or expired license token" in resp.text


class TestExecuteEndpoint:
    async def test_execute_requires_source(self, client):
        """Must provide either bdd_job_id or package_url."""
        resp = await client.post(
            "/api/test-runner/execute",
            json={"target_url": "https://example.com"},
        )
        assert resp.status_code == 422

    async def test_execute_creates_job(self, client):
        with patch("src.api.routes._process_job", new_callable=AsyncMock):
            resp = await client.post(
                "/api/test-runner/execute",
                json={
                    "bdd_job_id": "bdd-123",
                    "target_url": "https://example.com",
                },
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert "/api/test-runner/" in data["poll_url"]

    async def test_execute_never_persists_the_login_password(self, client):
        """The job row's `config` is returned verbatim by GET /{job_id}/result, so
        the login secret must not be in it; the live request still reaches the job."""
        from src.api.routes import JobManager

        original = JobManager.create_job
        stored: dict = {}

        async def _capture(*args, **kwargs):
            stored.update(kwargs["config"])
            return await original(*args, **kwargs)

        secret = "pw-not-for-the-row"  # pragma: allowlist secret
        with (
            patch("src.api.routes._process_job", new_callable=AsyncMock) as proc,
            patch("src.api.routes.JobManager.create_job", side_effect=_capture),
        ):
            resp = await client.post(
                "/api/test-runner/execute",
                json={
                    "bdd_job_id": "bdd-auth",
                    "target_url": "https://example.com",
                    "auth_config": [
                        {"login_url": "https://example.com/login", "password": secret},
                        {"login_url": "https://other.example.com/", "password": secret},
                    ],
                },
            )
        assert resp.status_code == 200, resp.text
        assert secret not in str(stored)
        assert [e["login_url"] for e in stored["auth_config"]] == [
            "https://example.com/login",
            "https://other.example.com/",
        ]
        live_request = proc.call_args.args[1]
        assert live_request.auth_config[0]["password"] == secret

    async def test_execute_dedupes_on_idempotency_key(self, client):
        """Two /execute POSTs with the same Idempotency-Key spawn the recording job
        ONCE and both return the SAME job_id — so demo_video_record retrying after a
        resumable timeout can't trigger a second paid bddgen+Playwright run."""
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            body = {"bdd_job_id": "bdd-idem", "target_url": "https://example.com"}
            headers = {"Idempotency-Key": "run-xyz:demo_video_record"}
            r1 = await client.post("/api/test-runner/execute", json=body, headers=headers)
            r2 = await client.post("/api/test-runner/execute", json=body, headers=headers)
        assert r1.status_code == 200 and r2.status_code == 200
        assert r1.json()["job_id"] == r2.json()["job_id"]  # same original job
        assert proc.call_count == 1  # second request did NOT re-spawn the run

    async def test_execute_with_package_url(self, client):
        with patch("src.api.routes._process_job", new_callable=AsyncMock):
            resp = await client.post(
                "/api/test-runner/execute",
                json={
                    "package_url": "https://example.com/package.zip",
                    "target_url": "https://example.com",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"


class TestExecuteForwardsTheCallersLicenceToken:
    """The licence forwarded into the background job must follow w2v_common's
    CANONICAL extraction rule — both accepted credential shapes, right precedence.

    The background task re-authenticates the S2S package download from
    bdd_playwright_service; AuthMiddleware's identity ContextVars do not survive
    into a detached BackgroundTask, so the token is captured request-scoped and
    passed as the 4th positional arg to ``_process_job``.

    SILENT DEFECT this covers: routes.py used to re-implement the extraction
    rule locally, as a copy of w2v_common's PRE-0.74.0 version that read the
    licence ONLY from ``X-Tenant-Id: Bearer``. 0.74.0 widened the canonical rule
    to also accept an opaque licence on ``Authorization: Bearer`` — the only
    shape the ``license-authz`` ext-authz gate can read. Behind that gate the
    request authenticated (the middleware had the wide rule) but the local copy
    returned None, so the job forwarded an EMPTY token and the download 401'd.
    Nothing failed: every existing test sent the legacy header.
    """

    @staticmethod
    def _forwarded_token(proc) -> str | None:
        """The ``license_token`` positional handed to the background task."""
        assert proc.call_count == 1, f"expected one background job, got {proc.call_count}"
        return proc.call_args.args[3]

    async def test_opaque_authorization_bearer_licence_is_forwarded(self, unauth_client):
        """An opaque licence on ``Authorization: Bearer`` reaches the background job.

        FAILS against the pre-fix local extractor: it never looked at
        ``Authorization``, so the job was spawned with ``license_token=None``
        even though the very same token had just authenticated the request.
        """
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            resp = await unauth_client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-authz", "target_url": "https://example.com"},
                headers={"Authorization": f"Bearer {TEST_LICENSE_TOKEN}"},
            )
        assert resp.status_code == 200, resp.text
        assert self._forwarded_token(proc) == TEST_LICENSE_TOKEN

    async def test_legacy_x_tenant_id_bearer_still_forwarded(self, unauth_client):
        """The legacy w2v shape keeps working — every current caller sends it."""
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            resp = await unauth_client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-legacy", "target_url": "https://example.com"},
                headers={"X-Tenant-Id": f"Bearer {TEST_LICENSE_TOKEN}"},
            )
        assert resp.status_code == 200, resp.text
        assert self._forwarded_token(proc) == TEST_LICENSE_TOKEN

    async def test_x_tenant_id_wins_when_both_headers_present(self, unauth_client):
        """Precedence guard: ``X-Tenant-Id: Bearer`` is checked FIRST.

        BaseServiceClient puts the CALLER's delegated licence on ``X-Tenant-Id``
        while its constructor-baked SERVICE-ACCOUNT token sits on
        ``Authorization``. Preferring ``Authorization`` would silently
        re-attribute every delegated call to the service account's tenant — so
        the caller's token is the one that must be forwarded onward.
        """
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            resp = await unauth_client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-both", "target_url": "https://example.com"},
                headers={
                    "X-Tenant-Id": f"Bearer {TEST_LICENSE_TOKEN}",
                    "Authorization": "Bearer service-account-opaque-token",
                },
            )
        assert resp.status_code == 200, resp.text
        assert self._forwarded_token(proc) == TEST_LICENSE_TOKEN

    async def test_jwt_shaped_authorization_is_never_a_licence(self, unauth_client):
        """A JWT-shaped ``Authorization`` bearer is NOT an opaque licence.

        It takes the Keycloak JWT lane, which (with no validator configured
        here) rejects outright — it must never be introspected as a licence nor
        forwarded as one. The job is never spawned.
        """
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            resp = await unauth_client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-jwt", "target_url": "https://example.com"},
                headers={"Authorization": "Bearer header.payload.signature"},
            )
        assert resp.status_code == 401, resp.text
        assert proc.call_count == 0

    async def test_no_licence_forwards_none_not_a_stray_header_value(self, client):
        """A bare (non-Bearer) ``X-Tenant-Id`` is an identity assertion, not a
        credential — it must never be forwarded as a licence token."""
        with patch("src.api.routes._process_job", new_callable=AsyncMock) as proc:
            resp = await client.post(
                "/api/test-runner/execute",
                json={"bdd_job_id": "bdd-bare", "target_url": "https://example.com"},
                headers={"X-Tenant-Id": TEST_TENANT_ID},
            )
        # Bare header alone is not a credential: fail-closed (no trusted-network
        # fallback in tests), so the job is never spawned with it as a token.
        assert resp.status_code == 401, resp.text
        assert proc.call_count == 0


class TestJobStatusEndpoint:
    async def test_get_nonexistent_job(self, client):
        resp = await client.get("/api/test-runner/nonexistent-job")
        assert resp.status_code == 404

    async def test_get_existing_job(self, client):
        job = _make_job(status="processing", error_message=None)
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "processing"
        assert data["error"] is None

    async def test_get_failed_job_carries_error(self, client):
        job = _make_job(status="failed", error_message="boom")
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1")
        assert resp.status_code == 200
        assert resp.json()["error"] == "boom"


class TestJobResultEndpoint:
    async def test_get_nonexistent_result(self, client):
        resp = await client.get("/api/test-runner/nonexistent-job/result")
        assert resp.status_code == 404

    async def test_get_result_happy_path(self, client):
        job = _make_job(status="completed")
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1/result")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["status"] == "completed"
        assert data["summary"]["total"] == 2
        assert len(data["test_results"]) == 1
        assert data["test_results"][0]["name"] == "01950000-0000-7000-8000-000000000001"
        assert data["execution_time_seconds"] == pytest.approx(12.5)

    async def test_get_result_conflict_when_still_processing(self, client):
        job = _make_job(status="processing")
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1/result")
        assert resp.status_code == 409
        body = resp.json()
        assert body["error"]["status"] == 409
        assert "still processing" in body["error"]["message"]


class TestReportEndpoint:
    async def test_get_report_happy_path(self, client):
        job = _make_job(status="completed", artifact_paths={"root_dir": "/tmp/foo"})
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1/report")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == "job-1"
        assert data["report_url"] == "/reports/job-1"
        assert data["artifact_paths"] == {"root_dir": "/tmp/foo"}

    async def test_get_report_nonexistent(self, client):
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=None)):
            resp = await client.get("/api/test-runner/missing/report")
        assert resp.status_code == 404

    async def test_get_report_conflict_when_not_completed(self, client):
        job = _make_job(status="processing")
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1/report")
        assert resp.status_code == 409


class TestArtifactEndpoint:
    """GET /{job_id}/artifacts/{path} — streams the report from the shared store,
    so any replica serves any job's report (multi-replica safety)."""

    async def test_artifact_nonexistent_job(self, client):
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=None)):
            resp = await client.get("/api/test-runner/missing/artifacts/report.html")
        assert resp.status_code == 404

    async def test_artifact_no_report(self, client):
        job = _make_job(artifact_paths={})
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)):
            resp = await client.get("/api/test-runner/job-1/artifacts/report.html")
        assert resp.status_code == 404

    async def test_artifact_happy_path(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        await store.store_bytes("job-1", "playwright-report/index.html", b"<html>ok</html>")

        job = _make_job(artifact_paths={"has_report": True})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/artifacts/playwright-report/index.html")
        assert resp.status_code == 200
        assert resp.text == "<html>ok</html>"

    async def test_artifact_path_traversal_rejected(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        # An object outside the job prefix that traversal would try to reach.
        await store.store_bytes("outside.txt", "x", b"secret")

        job = _make_job(artifact_paths={"has_report": True})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/artifacts/..%2Foutside.txt")
        # 400 (rejected by _safe_key) or 404 — both are safe; escape must not succeed.
        assert resp.status_code in (400, 404)
        assert resp.text != "secret"

    async def test_artifact_missing_object(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        job = _make_job(artifact_paths={"has_report": True})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/artifacts/does-not-exist.html")
        assert resp.status_code == 404


class TestClipVideoEndpoint:
    """GET /{job_id}/clips/{clip_key}/video — tenant-scoped, streamed from the
    shared object store (any replica serves any job's clip)."""

    async def test_clip_nonexistent_job(self, client):
        with patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=None)):
            resp = await client.get("/api/test-runner/missing/clips/clip-1/video")
        assert resp.status_code == 404

    async def test_clip_unknown_key(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        job = _make_job(artifact_paths={"clip_files": {}})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/clips/clip-99/video")
        assert resp.status_code == 404

    async def test_clip_happy_path(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        await store.store_bytes("job-1", "clips/clip-1.mp4", b"\x00\x00mp4bytes")
        job = _make_job(artifact_paths={"clip_files": {"clip-1": "clip-1.mp4"}})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/clips/clip-1/video")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "video/mp4"
        assert resp.content == b"\x00\x00mp4bytes"

    async def test_clip_missing_object(self, client, tmp_path: Path):
        from tests.conftest import make_local_storage

        store = make_local_storage(tmp_path / "store")
        job = _make_job(artifact_paths={"clip_files": {"clip-1": "clip-1.mp4"}})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", store),
        ):
            resp = await client.get("/api/test-runner/job-1/clips/clip-1/video")
        assert resp.status_code == 404

    async def test_clip_served_by_a_different_replica(self, client, tmp_path: Path):
        """Multi-replica regression guard: a clip WRITTEN by 'pod A' (one store
        instance) is served by 'pod B' (a DIFFERENT store instance over the same
        shared store) with NO pod-local job dir — resolution is by object KEY, not
        a per-pod persist_dir. This is the cross-pod 404 the migration fixes."""
        from tests.conftest import make_local_storage

        shared_root = tmp_path / "shared_store"
        pod_a = make_local_storage(shared_root)
        await pod_a.store_bytes("job-1", "clips/clip-1.mp4", b"\x00\x00cross-pod")

        pod_b = make_local_storage(shared_root)  # a fresh instance = a different pod
        job = _make_job(artifact_paths={"clip_files": {"clip-1": "clip-1.mp4"}})
        with (
            patch("src.api.routes.JobManager.get_job", AsyncMock(return_value=job)),
            patch("src.api.routes._storage", pod_b),
        ):
            resp = await client.get("/api/test-runner/job-1/clips/clip-1/video")
        assert resp.status_code == 200
        assert resp.content == b"\x00\x00cross-pod"


class TestProcessJobBackground:
    """Exercise the _process_job background task path via a mocked engine."""

    async def test_process_job_success_path(self):
        """Successful execution should mark_completed with parsed results."""
        from src.api.routes import _process_job
        from src.api.schemas import TestExecutionRequest

        engine_result = {
            "summary": {
                "total": 1,
                "passed": 1,
                "failed": 0,
                "skipped": 0,
                "flaky": 0,
                "pass_rate": 100.0,
            },
            "test_results": [
                {
                    "name": "01950000-0000-7000-8000-000000000001",
                    "status": "passed",
                    "duration_ms": 100,
                }
            ],
            "report_url": "/reports/foo",
            "artifact_paths": {"root_dir": "/tmp/foo"},
            "scenario_clips": [
                {"clip_key": "clip-1", "video_url": "/api/test-runner/job-1/clips/clip-1/video"}
            ],
        }

        with (
            patch("src.api.routes.get_db_session", _fake_db_session),
            patch(
                "src.api.routes._engine.execute",
                new_callable=AsyncMock,
                return_value=engine_result,
            ),
            patch("src.api.routes.JobManager.mark_processing", new_callable=AsyncMock) as mark_p,
            patch("src.api.routes.JobManager.mark_completed", new_callable=AsyncMock) as mark_c,
            patch("src.api.routes.JobManager.mark_failed", new_callable=AsyncMock) as mark_f,
        ):
            await _process_job(
                "job-1",
                TestExecutionRequest(bdd_job_id="bdd-1", target_url="https://example.com"),
                TEST_TENANT_ID,
                drain=JobDrain(),
            )

        mark_p.assert_awaited_once()
        mark_c.assert_awaited_once()
        mark_f.assert_not_awaited()
        assert mark_c.await_args is not None
        kwargs = mark_c.await_args.kwargs
        assert kwargs["summary"]["total"] == 1
        assert kwargs["report_url"] == "/reports/foo"
        assert kwargs["scenario_clips"][0]["clip_key"] == "clip-1"

    async def test_process_job_failure_path(self):
        """Engine exception should mark_failed with truncated error."""
        from src.api.routes import _process_job
        from src.api.schemas import TestExecutionRequest

        with (
            patch("src.api.routes.get_db_session", _fake_db_session),
            patch(
                "src.api.routes._engine.execute",
                new_callable=AsyncMock,
                side_effect=RuntimeError("playwright crashed"),
            ),
            patch("src.api.routes.JobManager.mark_processing", new_callable=AsyncMock),
            patch("src.api.routes.JobManager.mark_completed", new_callable=AsyncMock) as mark_c,
            patch("src.api.routes.JobManager.mark_failed", new_callable=AsyncMock) as mark_f,
        ):
            await _process_job(
                "job-2",
                TestExecutionRequest(bdd_job_id="bdd-2", target_url="https://example.com"),
                TEST_TENANT_ID,
                drain=JobDrain(),
            )

        mark_c.assert_not_awaited()
        mark_f.assert_awaited_once()
        assert mark_f.await_args is not None
        kwargs = mark_f.await_args.kwargs
        assert "RuntimeError" in kwargs["error_message"]
        assert "playwright crashed" in kwargs["error_message"]

    async def test_process_job_bails_on_state_conflict(self):
        """If mark_processing raises JobStateConflictError, the task exits cleanly."""
        from src.api.routes import _process_job
        from src.api.schemas import TestExecutionRequest
        from src.services.job_manager import JobStateConflictError

        with (
            patch("src.api.routes.get_db_session", _fake_db_session),
            patch("src.api.routes._engine.execute", new_callable=AsyncMock) as engine_exec,
            patch(
                "src.api.routes.JobManager.mark_processing",
                new_callable=AsyncMock,
                side_effect=JobStateConflictError("lost race"),
            ),
            patch("src.api.routes.JobManager.mark_completed", new_callable=AsyncMock) as mark_c,
            patch("src.api.routes.JobManager.mark_failed", new_callable=AsyncMock) as mark_f,
        ):
            await _process_job(
                "job-3",
                TestExecutionRequest(bdd_job_id="bdd-3", target_url="https://example.com"),
                TEST_TENANT_ID,
                drain=JobDrain(),
            )

        engine_exec.assert_not_awaited()
        mark_c.assert_not_awaited()
        mark_f.assert_not_awaited()


# ---------------------------------------------------------------------------
# BOLA regression (#bug 2026-05-11)
# ---------------------------------------------------------------------------


class TestCrossTenantBolaProtection:
    """Cross-tenant access to job-by-id routes must 404.

    Pre-fix every read route in `routes.py` looked the job up by URL
    `{job_id}` alone — any authenticated tenant could read any other
    tenant's job status / result / report / artifacts.
    """

    async def test_cross_tenant_get_routes_return_404(
        self,
        client,
        db_session,
    ):
        from src.database.models import TestExecutionJob

        # Create a job belonging to a tenant that is NOT the request's.
        other_tenant_job = TestExecutionJob(
            job_id="job-other-tenant-fixed-id",
            tenant_id=OTHER_TENANT_ID,
            target_url="https://example.com",
            status="completed",
            summary={"total": 0, "passed": 0, "failed": 0, "duration_ms": 0},
            test_results=[],
            artifact_paths={"root_dir": "/tmp/never-served"},
            report_url=None,
        )
        db_session.add(other_tenant_job)
        await db_session.commit()
        await db_session.refresh(other_tenant_job)
        jid = other_tenant_job.job_id

        for path in (
            f"/api/test-runner/{jid}",
            f"/api/test-runner/{jid}/result",
            f"/api/test-runner/{jid}/report",
            f"/api/test-runner/{jid}/artifacts/index.html",
        ):
            resp = await client.get(path)
            assert resp.status_code == 404, (
                f"BOLA: {path} returned {resp.status_code}, expected 404"
            )
