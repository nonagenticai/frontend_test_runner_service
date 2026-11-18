"""Shared test fixtures for frontend_test_runner_service."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.database.models import Base


def make_local_storage(root: Path) -> Any:
    """Build a local-backend ``StorageService`` rooted at ``root`` — the shared
    object store stand-in for tests. Bypasses ``get_settings()`` so a test can
    point the store at a tmp dir without touching global config. The same store
    (any instance rooted at ``root``) serves any key written under it, which is
    exactly the cross-replica property the routes rely on.
    """
    from w2v_common.config import StorageSettings
    from w2v_common.storage import StorageService as SharedStorageService

    from src.services.storage_service import StorageService

    store = StorageService.__new__(StorageService)
    store._storage = SharedStorageService(StorageSettings(backend="local", local_path=str(root)))
    store._backend = "local"
    store._local_path = root
    store._initialized = False
    store._init_lock = asyncio.Lock()
    return store


# Capture the test DB URL at module load. CI injects DATABASE_URL via the
# ephemeral postgres sidecar in .github/workflows/ci-cd.yaml; locally falls
# back to the config default (postgresql+psycopg://postgres:postgres@
# localhost:5450/frontend_test_runner). Captured here so any later fixture
# that monkeypatches DATABASE_URL for app-config tests cannot affect the
# real engine the db_engine fixture connects to.
_TEST_DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://postgres:postgres@localhost:5450/frontend_test_runner",  # pragma: allowlist secret
)

# w2v_common >= 0.15.0 dropped the localhost defaults on DatabaseSettings.url,
# RedisSettings.url, and StorageSettings.minio_endpoint. Settings() is
# instantiated at module-import time (e.g. when test files import from src.*
# during pytest collection), which happens BEFORE any fixture runs. Seed
# module-level so collection succeeds.
os.environ.setdefault("DATABASE_URL", _TEST_DATABASE_URL)
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("STORAGE_MINIO_ENDPOINT", "localhost:9000")
# Keep the artifact store on the local backend rooted at a scratch dir for tests
# (prod uses STORAGE_BACKEND=minio); avoids the startup init touching /app/data.
os.environ.setdefault("STORAGE_BACKEND", "local")
os.environ.setdefault("STORAGE_LOCAL_PATH", str(Path(tempfile.gettempdir()) / "ftr_test_store"))

# Default test license token. Route fixtures send this in the
# `X-Tenant-Id: Bearer <token>` header so the auth middleware exercises
# its real prod path (license introspection), not the legacy in-cluster
# trusted-network fallback. The fallback (now controlled by
# `AuthSettings.trust_incluster_identity`, fed via the
# `AUTH_TRUST_INCLUSTER_IDENTITY` env var) defaults to fail-closed
# since w2v_common 0.6.9, and we deliberately keep that default in
# tests so route auth is genuinely covered.
TEST_LICENSE_TOKEN = "test-license-token"

# w2v_common >= 0.67.0 made a tenant id a real ``uuid.UUID``: the auth
# middleware now 401s on a slug tenant (``InvalidTenantIdError``) instead of
# substituting a sentinel. These are the ONE shared set of test tenants — a
# route test that authenticates uses TEST_TENANT_ID, and anything proving
# cross-tenant isolation must use a DIFFERENT one (OTHER_TENANT_ID /
# TENANT_A_ID / TENANT_B_ID), never collapse them.
TEST_TENANT_ID = "0198f7a0-0000-7000-8000-000000000001"
OTHER_TENANT_ID = "0198f7a0-0000-7000-8000-000000000002"
TENANT_A_ID = "0198f7a0-0000-7000-8000-00000000000a"
TENANT_B_ID = "0198f7a0-0000-7000-8000-00000000000b"


def _fake_introspect_result() -> dict[str, Any]:
    """Shape returned by the license-authz introspection endpoint for a valid token."""
    return {
        "active": True,
        "tenant_id": TEST_TENANT_ID,
        "subject_id": "test-subject",
        "scopes": ["frontend-test-runner.execute"],
    }


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Headers that authenticate as the test tenant via license introspection."""
    return {"X-Tenant-Id": f"Bearer {TEST_LICENSE_TOKEN}"}


@pytest.fixture(autouse=True)
def mock_license_introspection() -> Iterator[AsyncMock]:
    """Patch w2v_common's license introspector to accept the test token.

    The middleware calls `w2v_common.auth.middleware._introspect_license`,
    which normally POSTs to the license-authz service. We replace that
    network call with an in-process AsyncMock so route tests can send a
    real `X-Tenant-Id: Bearer <token>` header and exercise the
    middleware's license path end-to-end. Tokens not equal to
    `TEST_LICENSE_TOKEN` introspect as inactive (-> 401), preserving
    negative-path coverage for tests that send a bogus or missing token.
    """

    async def _introspect(token: str) -> dict[str, Any] | None:
        if token == TEST_LICENSE_TOKEN:
            return _fake_introspect_result()
        return {"active": False, "reason": "token_not_found"}

    with patch(
        "w2v_common.auth.middleware._introspect_license",
        new=AsyncMock(side_effect=_introspect),
    ) as mock:
        yield mock


@pytest.fixture
async def db_engine():
    """Per-test async postgres engine with TRUNCATE-based isolation.

    Connects to _TEST_DATABASE_URL captured at module load (CI postgres in
    CI, localhost:5450 locally). Runs Base.metadata.create_all (idempotent)
    then TRUNCATEs every model table so each test starts from a clean slate.
    """
    from src.services.cleanup_service import create_reap_function_statements

    engine = create_async_engine(_TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # The cross-tenant cleanup reaper functions live in the alembic migration in
        # prod; tests build the schema via create_all, so install them here from the
        # same single source (src/services/cleanup_service.py) that migration 003 uses.
        for stmt in create_reap_function_statements():
            await conn.execute(text(stmt))
        if Base.metadata.sorted_tables:
            tables = ", ".join(f'"{t.name}"' for t in reversed(Base.metadata.sorted_tables))
            await conn.execute(text(f"TRUNCATE TABLE {tables} RESTART IDENTITY CASCADE"))
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session_factory(db_engine):
    """Async session factory bound to the test engine.

    Used by the RLS ``after_begin`` re-bind regression test, which needs to open
    multiple transactions on distinct sessions from the SAME engine.
    """
    return async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture
async def db_session(db_session_factory) -> AsyncIterator[AsyncSession]:
    """Create a test database session."""
    async with db_session_factory() as session:
        yield session


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock = AsyncMock()
    mock.ping = AsyncMock(return_value=True)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    return mock


@pytest.fixture
async def app(db_engine, mock_redis) -> AsyncIterator[FastAPI]:
    """Create a test FastAPI app with mocked dependencies."""
    from src.config import clear_settings_cache

    clear_settings_cache()

    # Create a session factory that uses the test SQLite engine
    test_session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

    @asynccontextmanager
    async def _test_get_db_session():
        async with test_session_factory() as session:
            yield session

    with (
        patch("src.asgi.get_engine", return_value=db_engine),
        patch("src.asgi.init_redis", return_value=mock_redis),
        patch("src.asgi.get_redis_client", return_value=mock_redis),
        patch("src.asgi.close_redis"),
        patch("src.asgi.close_engine"),
        # Patch get_db_session where routes import it from
        patch("src.api.routes.get_db_session", _test_get_db_session),
    ):
        from src.asgi import app

        async with app.router.lifespan_context(app):
            yield app


@pytest.fixture
async def client(app: FastAPI, auth_headers: dict[str, str]) -> AsyncIterator[AsyncClient]:
    """Test HTTP client that authenticates via X-Tenant-Id Bearer by default.

    Tests that need to exercise the unauthenticated or legacy
    trusted-network path should use the `unauth_client` fixture instead.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=auth_headers) as ac:
        yield ac


@pytest.fixture
async def unauth_client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """Test HTTP client without auth headers.

    Use this only for tests that explicitly verify auth-failure
    behavior or that opt into the legacy trusted-network fallback via
    `AUTH_TRUST_INCLUSTER_IDENTITY=true` (now wired through
    `AuthSettings.trust_incluster_identity` since w2v_common 0.6.10).
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
