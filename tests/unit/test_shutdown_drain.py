"""Graceful-shutdown drain for in-flight test runs.

The bug these cover: ``_on_shutdown`` closed Redis and the DB engine
immediately, so a pod deletion (HPA scale-down, rollout, eviction) tore the
database out from under a live ``_process_job``. The row stayed ``processing``,
stopped heartbeating, and 30 minutes later the reaper mislabelled it
``"Job stuck (no heartbeat — producing pod likely crashed)"``.

Each test below is written so that REMOVING the drain flips it red — see the
per-test notes.
"""

from __future__ import annotations

import asyncio
import logging
from typing import cast

import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import src.asgi as asgi_module
from src.api.health import readiness
from src.api.schemas import TestExecutionRequest
from src.services.drain import JobDrain


@pytest.fixture
def drain_app() -> FastAPI:
    """A bare app carrying its own ``JobDrain`` — exactly how the real app is wired.

    No monkeypatching of module globals: the registry is per-app state, so a
    test isolates it by owning an app, not by rebinding an import. That is the
    whole point of the refactor this file guards.
    """
    app = FastAPI()
    app.state.job_drain = JobDrain()
    return app


@pytest.fixture
def fresh_drain(drain_app: FastAPI) -> JobDrain:
    """The registry belonging to ``drain_app``."""
    return cast(JobDrain, drain_app.state.job_drain)


@pytest.fixture
def teardown_order(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record the order of the shutdown teardown steps."""
    order: list[str] = []

    async def _close_redis() -> None:
        order.append("redis_closed")

    async def _close_engine() -> None:
        order.append("engine_closed")

    monkeypatch.setattr(asgi_module, "close_redis", _close_redis)
    monkeypatch.setattr(asgi_module, "close_engine", _close_engine)
    return order


_TENANT = "0195f3a1-0000-7000-8000-000000000001"


async def _anoop(*args: object, **kwargs: object) -> None:
    return None


async def _afalse(*args: object, **kwargs: object) -> bool:
    return False


async def _aforever(*args: object, **kwargs: object) -> None:
    await asyncio.Event().wait()


def _request_for(app: FastAPI) -> Request:
    """A minimal ASGI ``Request`` whose ``.app`` is ``app`` — the only thing the
    readiness gate and ``/execute`` read off it."""
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})


def _request_stub() -> TestExecutionRequest:
    """``_process_job`` only passes the request object through to the engine."""
    return TestExecutionRequest(target_url="https://example.test", bdd_job_id="bdd-1")


class TestDrainOrdering:
    async def test_shutdown_waits_for_inflight_job_before_closing_redis_and_db(
        self, drain_app: FastAPI, fresh_drain: JobDrain, teardown_order: list[str]
    ) -> None:
        """The load-bearing assertion: the job finishes BEFORE the engine closes.

        Without the drain, ``_on_shutdown`` returns while the job is still
        running, so the observed order is
        ``["redis_closed", "engine_closed", "job_finished"]`` — red.
        """
        fresh_drain.register("job-alpha")

        async def _slow_job() -> None:
            await asyncio.sleep(0.15)
            teardown_order.append("job_finished")
            fresh_drain.unregister("job-alpha")

        runner = asyncio.create_task(_slow_job())
        await asgi_module._on_shutdown(drain_app)
        await runner

        assert teardown_order == ["job_finished", "redis_closed", "engine_closed"]

    async def test_shutdown_is_immediate_when_nothing_is_in_flight(
        self, drain_app: FastAPI, fresh_drain: JobDrain, teardown_order: list[str]
    ) -> None:
        """An idle pod must not pay the drain timeout."""
        loop = asyncio.get_running_loop()
        started = loop.time()
        await asgi_module._on_shutdown(drain_app)
        assert loop.time() - started < 1.0
        assert teardown_order == ["redis_closed", "engine_closed"]


class TestDrainTimeoutIsLoud:
    async def test_timeout_logs_error_naming_the_abandoned_jobs(
        self,
        drain_app: FastAPI,
        fresh_drain: JobDrain,
        teardown_order: list[str],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A blown drain must be diagnosable NOW, not via the reaper in 30 min."""
        monkeypatch.setattr(
            asgi_module.settings.shutdown, "drain_timeout_seconds", 0.05, raising=False
        )
        handed_back: list[list[str]] = []

        async def _hand_back(job_ids: list[str]) -> None:
            handed_back.append(job_ids)

        monkeypatch.setattr(asgi_module, "_hand_back_abandoned_jobs", _hand_back)

        fresh_drain.register("job-stuck-1")
        fresh_drain.register("job-stuck-2")

        with caplog.at_level(logging.ERROR, logger="src.asgi"):
            await asgi_module._on_shutdown(drain_app)

        errors = [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors, "drain timeout was swallowed — no ERROR emitted"
        blob = "\n".join(errors)
        assert "job-stuck-1" in blob and "job-stuck-2" in blob
        # And the abandoned jobs are handed back so the caller sees a terminal
        # status immediately instead of polling a 'processing' row for 30 min.
        assert handed_back == [["job-stuck-1", "job-stuck-2"]]
        # Teardown still happens — a blown drain must not wedge the process.
        assert teardown_order == ["redis_closed", "engine_closed"]

    async def test_clean_drain_emits_no_error(
        self,
        drain_app: FastAPI,
        fresh_drain: JobDrain,
        teardown_order: list[str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Guards the negative: the ERROR above must mean something."""
        fresh_drain.register("job-quick")

        async def _quick_job() -> None:
            await asyncio.sleep(0)
            fresh_drain.unregister("job-quick")

        runner = asyncio.create_task(_quick_job())
        with caplog.at_level(logging.ERROR, logger="src.asgi"):
            await asgi_module._on_shutdown(drain_app)
        await runner

        assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


class TestReadinessGate:
    async def test_readiness_reports_503_while_draining(
        self, drain_app: FastAPI, fresh_drain: JobDrain
    ) -> None:
        """Endpoint removal is the whole point: no NEW job may be routed here.

        Short-circuits ahead of the DB/Redis probes, which are still healthy
        during a drain and would otherwise answer 200.
        """
        fresh_drain.register("job-running")
        fresh_drain.begin_drain()

        response = await readiness(_request_for(drain_app))

        assert isinstance(response, JSONResponse), "readiness answered 200 during a drain"
        assert response.status_code == 503
        assert b"draining" in response.body
        assert b"job-running" in response.body

    async def test_execute_refuses_new_work_while_draining(
        self, drain_app: FastAPI, fresh_drain: JobDrain
    ) -> None:
        """A caller that beat endpoint removal gets a retryable 503, not a doomed job."""
        from fastapi import BackgroundTasks, HTTPException

        from src.api.routes import start_execution

        fresh_drain.begin_drain()

        with pytest.raises(HTTPException) as exc:
            await start_execution(
                cast(TestExecutionRequest, None),
                _request_for(drain_app),
                cast(BackgroundTasks, None),
            )

        assert exc.value.status_code == 503
        assert "shutting down" in str(exc.value.detail)

    async def test_readiness_fails_closed_when_the_registry_is_missing(self) -> None:
        """No registry = the lifespan never ran, which is NOT ready.

        Guards the fail-closed choice that replaced the module singleton: if a
        future edit reached for ``getattr(..., None)`` and treated a missing
        drain as "not draining", the whole gate would vanish silently and every
        other test here would still pass.
        """
        response = await readiness(_request_for(FastAPI()))

        assert isinstance(response, JSONResponse), "readiness answered 200 with no registry"
        assert response.status_code == 503
        assert b"starting" in response.body


class TestProcessJobRegistration:
    """``_process_job`` must put itself in the registry, or the drain is blind."""

    @staticmethod
    def _stub_job_path(monkeypatch: pytest.MonkeyPatch, on_execute: object) -> None:
        """Neuter every I/O dependency of ``_process_job`` except the drain."""
        import contextlib

        import src.api.routes as routes

        @contextlib.asynccontextmanager
        async def _fake_session():  # type: ignore[no-untyped-def]
            yield object()

        monkeypatch.setattr(routes, "get_db_session", _fake_session)
        monkeypatch.setattr(routes.JobManager, "mark_processing", _anoop)
        monkeypatch.setattr(routes.JobManager, "mark_completed", _anoop)
        monkeypatch.setattr(routes.JobManager, "mark_failed", _anoop)
        monkeypatch.setattr(routes, "_register_concurrent_job", _afalse)
        monkeypatch.setattr(routes, "_release_concurrent_job", _anoop)
        monkeypatch.setattr(routes, "_heartbeat_loop", _aforever)
        monkeypatch.setattr(routes._engine, "execute", on_execute)

    async def test_job_is_registered_while_running_and_released_after(
        self, fresh_drain: JobDrain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.api.routes as routes

        seen: list[list[str]] = []

        async def _execute(*args: object, **kwargs: object) -> dict[str, object]:
            seen.append(fresh_drain.in_flight())
            return {"summary": {}, "test_results": [], "report_url": ""}

        self._stub_job_path(monkeypatch, _execute)
        await routes._process_job("job-reg", _request_stub(), _TENANT, drain=fresh_drain)

        assert seen == [["job-reg"]], "job was not registered while executing"
        assert fresh_drain.in_flight() == [], "job was not released after finishing"

    async def test_job_is_released_even_when_execution_raises(
        self, fresh_drain: JobDrain, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import src.api.routes as routes

        async def _boom(*args: object, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("playwright exploded")

        self._stub_job_path(monkeypatch, _boom)
        await routes._process_job("job-boom", _request_stub(), _TENANT, drain=fresh_drain)

        assert fresh_drain.in_flight() == []


class TestJobDrainRegistry:
    async def test_wait_for_idle_returns_remaining_ids_on_timeout(self) -> None:
        drain = JobDrain()
        drain.register("a")
        drain.register("b")
        assert await drain.wait_for_idle(0.01) == ["a", "b"]

    async def test_wait_for_idle_returns_empty_once_drained(self) -> None:
        drain = JobDrain()
        drain.register("a")
        drain.unregister("a")
        assert await drain.wait_for_idle(0.01) == []

    def test_begin_drain_is_idempotent(self) -> None:
        drain = JobDrain()
        assert drain.begin_drain() is True
        assert drain.begin_drain() is False
        assert drain.draining is True

    def test_unregister_of_unknown_job_is_a_noop(self) -> None:
        drain = JobDrain()
        drain.register("a")
        drain.unregister("ghost")
        assert drain.in_flight() == ["a"]
