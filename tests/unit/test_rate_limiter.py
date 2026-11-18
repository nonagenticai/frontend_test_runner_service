"""Tests for the atomic rate-limit middleware.

Covers the two HIGH-severity bugs from the 2026-04-27 audit plus the
2026-07-24 counter-leak fix:
- H1: dead concurrent-job counter (wired up in routes._process_job).
- H2: zcard/zadd race (replaced by a single Lua EVAL).
- Leak: the integer counter leaked a slot when a worker was hard-killed before
  its DECR (permanent under load — TTL reset on every INCR). Now a per-job
  sorted set scored by expiry deadline, so a killed job's slot self-prunes.

Tests use ``fakeredis`` because (a) it supports the Lua subset we need
(ZREMRANGEBYSCORE / ZCARD / ZADD / EXPIRE / GET) and (b) it lets us
assert real concurrent semantics without spinning up a Redis server.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from typing import Any, cast
from unittest.mock import patch

import fakeredis.aioredis
import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse

from src.middleware.rate_limiter import (
    _WINDOW_SECONDS,
    RateLimitMiddleware,
    concurrent_jobs_key,
    retry_after_for_window,
    window_key,
)
from src.services.drain import JobDrain
from tests.conftest import OTHER_TENANT_ID, TEST_TENANT_ID

# A deadline comfortably in the future — a live running job's slot.
_FUTURE = time.time() + 3600.0


class _YieldingRedis:
    """fakeredis proxy for the code paths that take a raw Redis handle.

    ⛔ DO NOT DELETE — and do NOT cite it as the reason the burst tests work.
    An earlier docstring called it "load-bearing for the burst tests". It is
    not, but it is not unused either; both halves are measured below.

    IT IS EXERCISED. Instrumented on 2026-09-06, the module instantiates it
    **5 times** and runs **8 Redis commands** through it (3 ``expire``, 3
    ``zadd``, 2 ``zrem``), from three tests:
    ``TestProcessJobMaintainsCounter::test_incr_then_decr_around_success``,
    ``TestProcessJobMaintainsCounter::test_decr_runs_even_on_exception`` and
    ``TestIdentityClearedEvenIfReleaseRaises::test_clear_identity_runs_when_release_raises``.
    Those drive ``routes._process_job`` directly, which takes a raw handle via
    ``_StubRedisClient._ensure_connected``. Making ``_ensure_connected`` return
    ``None`` fails exactly those 3 tests (23 pass), so removing this class
    breaks the concurrent-counter and identity-cleanup coverage.

    ITS OWN SUSPENSION IS INERT, THOUGH. Deleting the ``await asyncio.sleep(0)``
    from ``_call`` leaves all 26 tests GREEN: the three tests that reach it
    issue their requests sequentially, so there is never another coroutine
    waiting to interleave at that suspension point.

    THE BURST TESTS NEVER REACH THIS CLASS. The middleware under test
    (``src/middleware/rate_limiter.py:225``) goes exclusively through
    ``eval_script``; ``_ensure_connected`` has callers only in
    ``src/api/routes.py`` and ``src/api/health.py``. The interleaving that makes
    the H2 read-then-write race *reachable* comes from
    ``_StubRedisClient.eval_script``'s own ``await asyncio.sleep(0)`` — the
    single suspension point on the middleware's path. Measured 2026-09-06 by
    restoring the H2 race (moving the window ``ZADD`` out of the Lua script into
    a second, separately-awaited round-trip):

    * with ``eval_script``'s ``sleep(0)``    -> ``test_burst_admits_exactly_n``
      FAILS, ``admitted=100, rejected=0`` (2 failed / 24 passed overall) — the
      race is CAUGHT.
    * without it -> the same burst test PASSES — the race is MISSED.

    So that ``sleep(0)`` is the load-bearing line; keep it. (An earlier note
    dated 2026-09-04 claimed the split-``ZADD`` mutation "left all 26 tests
    GREEN". It does not — it is 2 failed / 24 passed — so that premise is
    dropped rather than repeated.)

    ``fakeredis`` resolves a command without ever yielding, whereas a real Redis
    command is a network round-trip that always suspends; this class models that
    for the raw-handle paths above.
    """

    def __init__(self, fake: fakeredis.aioredis.FakeRedis) -> None:
        self._fake = fake

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._fake, name)

        async def _call(*args: Any, **kwargs: Any) -> Any:
            await asyncio.sleep(0)
            return await attr(*args, **kwargs)

        return _call


class _StubRedisClient:
    """Stub for ``w2v_common.RedisClient`` that delegates to a fakeredis instance."""

    def __init__(self, fake: fakeredis.aioredis.FakeRedis) -> None:
        self._fake = fake

    def _ensure_connected(self) -> Any:
        return _YieldingRedis(self._fake)

    async def eval_script(
        self,
        script: str,
        keys: list[str] | None = None,
        args: list[Any] | None = None,
    ) -> Any:
        # One suspension for the round-trip, then the WHOLE script executes
        # without interleaving — exactly Redis's EVAL guarantee.
        #
        # ⛔ THIS ``sleep(0)`` IS THE LOAD-BEARING LINE FOR THE BURST TESTS, not
        # the _YieldingRedis proxy above (see its docstring for the measurement).
        # It is the only suspension point on the middleware's path, so it is what
        # lets ``asyncio.gather`` interleave 100 requests. Remove it and a
        # restored check-then-act race passes ``test_burst_admits_exactly_n``
        # unnoticed.
        await asyncio.sleep(0)
        return await cast(
            Any,
            self._fake.eval(script, len(keys or []), *(keys or []), *(args or [])),
        )


@asynccontextmanager
async def _build_app(
    fake: fakeredis.aioredis.FakeRedis,
    *,
    rpm: int,
    concurrent: int,
) -> AsyncGenerator[FastAPI]:
    """Yield a FastAPI app with rate-limit middleware wired to ``fake``."""

    class _StubRateLimit:
        requests_per_minute = rpm
        concurrent_jobs_per_tenant = concurrent

    class _StubExecution:
        # Grounds the concurrent-case Retry-After hint (see
        # ``retry_after_for_concurrency``); mirrors the real default.
        heartbeat_interval_seconds = 60

    class _StubSettings:
        rate_limit = _StubRateLimit()
        execution = _StubExecution()

    def _stub_get_settings() -> _StubSettings:
        return _StubSettings()

    def _stub_get_redis_client() -> _StubRedisClient:
        return _StubRedisClient(fake)

    app = FastAPI()
    app.add_middleware(RateLimitMiddleware)

    async def _execute(_request: Request) -> JSONResponse:
        _ = _request
        # The middleware reads ``request.state.tenant_id``; defaults to the
        # fixture tenant uuid if the dispatch path didn't set it (no upstream
        # auth middleware here).
        return JSONResponse({"ok": True})

    app.post("/api/test-runner/execute")(_execute)

    async def _attach_tenant(request: Request, call_next: Any) -> Any:
        request.state.tenant_id = request.headers.get(
            "x-tenant", "01950000-0000-7000-8000-000000000001"
        )
        return await call_next(request)

    app.middleware("http")(_attach_tenant)

    with (
        patch("src.middleware.rate_limiter.get_settings", _stub_get_settings),
        patch("src.middleware.rate_limiter.get_redis_client", _stub_get_redis_client),
    ):
        yield app


@pytest.fixture
async def fake_redis() -> AsyncIterator[fakeredis.aioredis.FakeRedis]:
    fake = fakeredis.aioredis.FakeRedis()
    yield fake
    await fake.aclose()


class TestAtomicSlidingWindow:
    """H2: assert exactly N concurrent requests are admitted under burst."""

    async def test_burst_admits_exactly_n(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        rpm = 10
        async with _build_app(fake_redis, rpm=rpm, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:

                async def _hit() -> int:
                    resp = await ac.post("/api/test-runner/execute", json={})
                    return resp.status_code

                results = await asyncio.gather(*[_hit() for _ in range(100)])

        admitted = sum(1 for s in results if s == 200)
        rejected = sum(1 for s in results if s == 429)
        assert admitted + rejected == 100
        assert admitted == rpm, (
            f"Atomic sliding-window must admit exactly {rpm} of 100 bursting "
            f"requests; got admitted={admitted}, rejected={rejected}"
        )

    async def test_separate_tenants_have_independent_buckets(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        rpm = 5
        async with _build_app(fake_redis, rpm=rpm, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:

                async def _hit(tenant: str) -> int:
                    resp = await ac.post(
                        "/api/test-runner/execute",
                        json={},
                        headers={"x-tenant": tenant},
                    )
                    return resp.status_code

                t1 = await asyncio.gather(*[_hit("alpha") for _ in range(10)])
                t2 = await asyncio.gather(*[_hit("beta") for _ in range(10)])

        assert sum(1 for s in t1 if s == 200) == rpm
        assert sum(1 for s in t2 if s == 200) == rpm


class TestConcurrentLimit:
    """H1: assert the concurrent-job counter is read by the middleware."""

    async def test_concurrent_limit_trips_when_counter_at_cap(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Three LIVE running jobs (future deadlines) == the cap; next request 429s.
        await fake_redis.zadd(
            concurrent_jobs_key("01950000-0000-7000-8000-000000000001"),
            {"j1": _FUTURE, "j2": _FUTURE, "j3": _FUTURE},
        )
        async with _build_app(fake_redis, rpm=1000, concurrent=3) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})
        assert resp.status_code == 429
        body = resp.json()
        assert body["error"]["status"] == 429
        assert "Concurrent job limit" in body["error"]["message"]

    async def test_concurrent_below_cap_admits(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await fake_redis.zadd(
            concurrent_jobs_key("01950000-0000-7000-8000-000000000001"),
            {"j1": _FUTURE, "j2": _FUTURE},
        )
        async with _build_app(fake_redis, rpm=1000, concurrent=3) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})
        assert resp.status_code == 200

    async def test_leaked_slots_are_pruned_and_do_not_block(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # THE LEAK REGRESSION: 3 hard-killed jobs whose slots were never released
        # sit in the set with PAST deadlines. Admission must prune them and admit,
        # not 429 forever (the retired integer counter would have stayed at 3).
        past = time.time() - 1.0
        await fake_redis.zadd(
            concurrent_jobs_key("01950000-0000-7000-8000-000000000001"),
            {"dead1": past, "dead2": past, "dead3": past},
        )
        async with _build_app(fake_redis, rpm=1000, concurrent=3) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})
        assert resp.status_code == 200
        # The stale members were pruned from the set.
        assert (
            await fake_redis.zcard(concurrent_jobs_key("01950000-0000-7000-8000-000000000001")) == 0
        )

    async def test_live_jobs_counted_stale_pruned(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # 2 live + 5 stale → effective count 2 < cap 3 → admit (stale never block).
        past = time.time() - 1.0
        await fake_redis.zadd(
            concurrent_jobs_key("01950000-0000-7000-8000-000000000001"),
            {"live1": _FUTURE, "live2": _FUTURE, **{f"dead{i}": past for i in range(5)}},
        )
        async with _build_app(fake_redis, rpm=1000, concurrent=3) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})
        assert resp.status_code == 200
        assert (
            await fake_redis.zcard(concurrent_jobs_key("01950000-0000-7000-8000-000000000001")) == 2
        )

    async def test_admit_writes_window_entry(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        async with _build_app(fake_redis, rpm=10, concurrent=10) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})
        assert resp.status_code == 200
        assert await fake_redis.zcard(window_key("01950000-0000-7000-8000-000000000001")) == 1


class TestProcessJobMaintainsCounter:
    """_process_job must ZADD its slot on start and ZREM it in `finally`."""

    async def test_incr_then_decr_around_success(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.api import routes

        tenant = TEST_TENANT_ID
        # Stub the engine to return a minimal valid result.
        stub_engine = MagicMock()
        stub_engine.execute = AsyncMock(
            return_value={
                "summary": {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "flaky": 0,
                    "pass_rate": 0.0,
                },
                "test_results": [],
                "report_url": "",
                "artifact_paths": None,
            }
        )

        # Bypass DB: stub mark_processing/mark_completed; real DB tested elsewhere.
        with (
            patch.object(routes, "_engine", stub_engine),
            patch(
                "src.api.routes.get_redis_client",
                lambda: _StubRedisClient(fake_redis),
            ),
            patch("src.api.routes.JobManager.mark_processing", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_completed", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_failed", new=AsyncMock()),
            patch("src.api.routes.get_db_session", _noop_db_session),
        ):
            from src.api.schemas import TestExecutionRequest

            req = TestExecutionRequest(bdd_job_id="bdd-1", target_url="https://example.com")
            await routes._process_job("job-1", req, tenant, drain=JobDrain())

        # After completion, the job's slot is released → set is empty.
        assert await fake_redis.zcard(concurrent_jobs_key(tenant)) == 0

    async def test_decr_runs_even_on_exception(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock

        from src.api import routes

        tenant = OTHER_TENANT_ID
        stub_engine = MagicMock()
        stub_engine.execute = AsyncMock(side_effect=RuntimeError("boom"))

        with (
            patch.object(routes, "_engine", stub_engine),
            patch(
                "src.api.routes.get_redis_client",
                lambda: _StubRedisClient(fake_redis),
            ),
            patch("src.api.routes.JobManager.mark_processing", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_completed", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_failed", new=AsyncMock()),
            patch("src.api.routes.get_db_session", _noop_db_session),
        ):
            from src.api.schemas import TestExecutionRequest

            req = TestExecutionRequest(bdd_job_id="bdd-1", target_url="https://example.com")
            await routes._process_job("job-2", req, tenant, drain=JobDrain())

        # The slot is released in the `finally` even though execute() raised.
        assert await fake_redis.zcard(concurrent_jobs_key(tenant)) == 0


@asynccontextmanager
async def _noop_db_session() -> AsyncGenerator[Any]:
    """Async-context-manager that yields a sentinel — DB calls are mocked."""
    yield object()


class TestIdentityClearedEvenIfReleaseRaises:
    """CRITICAL isolation guarantee: `_process_job`'s `finally` must clear the re-seeded
    tenant identity even if `_release_concurrent_job` raises — else the tenant ContextVar
    leaks into a sibling task on the shared event loop and a later DB write binds to the
    WRONG tenant's RLS predicate."""

    async def test_clear_identity_runs_when_release_raises(
        self, fake_redis: fakeredis.aioredis.FakeRedis
    ) -> None:
        from unittest.mock import AsyncMock, MagicMock, patch

        from src.api import routes

        stub_engine = MagicMock()
        stub_engine.execute = AsyncMock(
            return_value={
                "summary": {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "flaky": 0,
                    "pass_rate": 0.0,
                },
                "test_results": [],
                "report_url": "",
                "artifact_paths": None,
            }
        )
        clear_spy = MagicMock()
        with (
            patch.object(routes, "_engine", stub_engine),
            patch("src.api.routes.get_redis_client", lambda: _StubRedisClient(fake_redis)),
            patch("src.api.routes.JobManager.mark_processing", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_completed", new=AsyncMock()),
            patch("src.api.routes.JobManager.mark_failed", new=AsyncMock()),
            patch("src.api.routes.get_db_session", _noop_db_session),
            # Release blows up with an UNEXPECTED error (past its own except) — the nested
            # finally must STILL clear identity.
            patch(
                "src.api.routes._release_concurrent_job",
                new=AsyncMock(side_effect=RuntimeError("boom")),
            ),
            patch("src.api.routes.clear_identity_context", clear_spy),
        ):
            from src.api.schemas import TestExecutionRequest

            req = TestExecutionRequest(bdd_job_id="bdd-1", target_url="https://example.com")
            # The raising release must not prevent identity clearing (it may re-raise out,
            # which is fine — the isolation invariant is that clear ran).
            with contextlib.suppress(RuntimeError):
                await routes._process_job("job-leak", req, TEST_TENANT_ID, drain=JobDrain())

        clear_spy.assert_called_once()


class TestRetryAfterHeader:
    """Both 429 branches must tell the client HOW LONG to wait.

    Regression for the 2026-07-20..24 burst: 22 ``demo_video_record`` runs died
    because the orchestrator client retried a bare 429 three times with 1s/2s
    linear backoff (~3s) against a 60s window — guaranteed exhaustion.
    """

    async def test_rate_429_carries_parsable_retry_after(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        async with _build_app(fake_redis, rpm=1, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                first = await ac.post("/api/test-runner/execute", json={})
                resp = await ac.post("/api/test-runner/execute", json={})

        assert first.status_code == 200
        assert resp.status_code == 429
        raw = resp.headers.get("retry-after")
        assert raw is not None, "rate-case 429 must carry Retry-After"
        seconds = int(raw)
        assert 1 <= seconds <= _WINDOW_SECONDS

    async def test_rate_retry_after_derived_from_window_not_hardcoded(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # The single window entry is 59.5s old, so it ages out in ~0.5s → the
        # honest hint is the 1s floor, NOT a blanket 60.
        await fake_redis.zadd(
            window_key("01950000-0000-7000-8000-000000000001"), {"old": time.time() - 59.5}
        )
        async with _build_app(fake_redis, rpm=1, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})

        assert resp.status_code == 429
        assert int(resp.headers["retry-after"]) == 1

    async def test_concurrent_429_carries_retry_after(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        await fake_redis.zadd(
            concurrent_jobs_key("01950000-0000-7000-8000-000000000001"),
            {"j1": _FUTURE, "j2": _FUTURE, "j3": _FUTURE},
        )
        async with _build_app(fake_redis, rpm=1000, concurrent=3) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})

        assert resp.status_code == 429
        raw = resp.headers.get("retry-after")
        assert raw is not None, "concurrent-case 429 must carry Retry-After"
        assert int(raw) >= 1

    async def test_rate_429_body_envelope_unchanged(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        # Callers parse the envelope; adding a header must not disturb it.
        async with _build_app(fake_redis, rpm=1, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                await ac.post("/api/test-runner/execute", json={})
                resp = await ac.post("/api/test-runner/execute", json={})

        body = resp.json()
        assert set(body) == {"error"}
        assert body["error"]["status"] == 429
        assert body["error"]["code"] == "HTTP_429"
        assert "Rate limit exceeded" in body["error"]["message"]
        assert body["error"]["request_id"]
        assert body["error"]["timestamp"]

    async def test_admitted_200_has_no_retry_after(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        async with _build_app(fake_redis, rpm=10, concurrent=10) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})

        assert resp.status_code == 200
        assert "retry-after" not in resp.headers


class TestRetryAfterForWindow:
    """Unit-level edges of the window arithmetic."""

    def test_rounds_up_and_floors_at_one(self) -> None:
        now = 1_000.0
        # Ages out 10.2s from now → 11 (round UP, never early).
        assert retry_after_for_window(now, str(now - _WINDOW_SECONDS + 10.2)) == 11
        # Already expired / racing → the 1s floor, never 0.
        assert retry_after_for_window(now, str(now - _WINDOW_SECONDS)) == 1
        assert retry_after_for_window(now, str(now - 999)) == 1

    def test_caps_at_window_and_survives_missing_or_garbage_score(self) -> None:
        now = 1_000.0
        # A clock-skewed future score cannot imply a wait longer than one window.
        assert retry_after_for_window(now, str(now + 500)) == _WINDOW_SECONDS
        assert retry_after_for_window(now, None) == _WINDOW_SECONDS
        assert retry_after_for_window(now, "") == _WINDOW_SECONDS
        assert retry_after_for_window(now, "not-a-number") == _WINDOW_SECONDS

    def test_accepts_bytes_score_from_undecoded_redis(self) -> None:
        now = 1_000.0
        score = str(now - _WINDOW_SECONDS + 5.0).encode()
        assert retry_after_for_window(now, score) == 5

    @pytest.mark.parametrize(
        "poison",
        ["inf", "-inf", "1e400", "nan", b"\xff\xfe", ["not", "a", "score"], {}],
        ids=["inf", "neg-inf", "overflow-literal", "nan", "invalid-utf8", "list", "dict"],
    )
    def test_poisoned_score_falls_back_never_raises(self, poison: object) -> None:
        """Anything can ``ZADD`` into the window key, so the score is hostile input.

        Each of these used to escape the ``ValueError``-only handler:
        ``inf``/``-inf``/``1e400`` → ``OverflowError`` from ``math.ceil``;
        ``nan`` → ``ValueError`` from ``ceil`` (raised OUTSIDE the old ``try``);
        invalid UTF-8 → ``UnicodeDecodeError`` from the ``.decode()`` (also
        outside); a non-str/bytes value → ``TypeError`` from ``float()``.
        """
        now = 1_000.0
        result = retry_after_for_window(now, cast(Any, poison))
        assert type(result) is int
        assert result == _WINDOW_SECONDS

    async def test_poisoned_window_score_still_429_not_500(
        self,
        fake_redis: fakeredis.aioredis.FakeRedis,
    ) -> None:
        """End-to-end: a corrupt score must degrade the HINT, never the STATUS.

        A 500 here would defeat the whole point of the Retry-After work — the
        orchestrator client treats 5xx and 429 completely differently.
        """
        await fake_redis.zadd(
            window_key("01950000-0000-7000-8000-000000000001"), {"old": float("inf")}
        )
        async with _build_app(fake_redis, rpm=1, concurrent=1000) as app:
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as ac:
                resp = await ac.post("/api/test-runner/execute", json={})

        assert resp.status_code == 429
        seconds = int(resp.headers["retry-after"])
        assert 1 <= seconds <= _WINDOW_SECONDS
