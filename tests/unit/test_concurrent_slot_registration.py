"""``_register_concurrent_job`` must answer "is there a member to release".

The bug: the ``ZADD`` that creates the slot and the housekeeping ``EXPIRE`` that
bounds the key sat under ONE ``except`` that returned ``False``. Those are two
round trips, so a connection reset BETWEEN them — a Redis restart/failover,
exactly the blip the handler exists for — reported "not registered" for a slot
that HAD been added. ``_process_job`` gates its ``finally`` ``ZREM`` on that
return value, so the release was skipped and the member sat in the per-tenant
running-jobs zset until its own deadline score expired it: ``max_run_seconds``
+ 5 min (~30 min) after a job that may have finished in two.

The admission Lua counts that member, so the tenant is short one concurrency
slot for half an hour per blip; three of them zero out the default cap of 3 and
every ``POST /execute`` 429s with "Concurrent job limit exceeded" while nothing
is running. A leaked slot whose lease keeps renewing is the failure signature
this fleet has already paid for once.
"""

from __future__ import annotations

import contextlib
from typing import Any

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

import src.api.routes as routes

pytestmark = pytest.mark.unit


class _FakeRedis:
    """Records zadd/zrem; ``expire`` raises the way a mid-sequence blip does."""

    def __init__(self, *, expire_raises: bool = False, zadd_raises: bool = False) -> None:
        self.expire_raises = expire_raises
        self.zadd_raises = zadd_raises
        self.members: dict[str, float] = {}
        self.zrem_calls: list[str] = []

    async def zadd(self, key: str, mapping: dict[str, float]) -> int:
        if self.zadd_raises:
            raise RedisConnectionError("connection reset by peer")
        self.members.update(mapping)
        return len(mapping)

    async def expire(self, key: str, ttl: int) -> bool:
        if self.expire_raises:
            raise RedisConnectionError("connection reset by peer")
        return True

    async def zrem(self, key: str, member: str) -> int:
        self.zrem_calls.append(member)
        self.members.pop(member, None)
        return 1


class _FakeRedisClient:
    def __init__(self, fake: _FakeRedis) -> None:
        self._fake = fake

    def _ensure_connected(self) -> _FakeRedis:
        return self._fake


@pytest.fixture
def wire_redis(monkeypatch: pytest.MonkeyPatch):
    def _wire(fake: _FakeRedis) -> _FakeRedis:
        monkeypatch.setattr(routes, "get_redis_client", lambda: _FakeRedisClient(fake))
        return fake

    return _wire


class TestRegisterConcurrentJob:
    async def test_expire_failure_after_a_successful_zadd_still_reports_registered(
        self, wire_redis
    ) -> None:
        fake = wire_redis(_FakeRedis(expire_raises=True))
        registered = await routes._register_concurrent_job("tenant-a", "job-1")
        assert fake.members, "precondition: the ZADD landed, so a member exists"
        assert registered is True, (
            "returning False here skips the finally-ZREM and strands the member "
            "until its deadline score expires it (~30 min)"
        )

    async def test_zadd_failure_reports_not_registered(self, wire_redis) -> None:
        fake = wire_redis(_FakeRedis(zadd_raises=True))
        assert await routes._register_concurrent_job("tenant-a", "job-1") is False
        assert not fake.members

    async def test_healthy_path_registers(self, wire_redis) -> None:
        fake = wire_redis(_FakeRedis())
        assert await routes._register_concurrent_job("tenant-a", "job-1") is True
        assert "job-1" in fake.members


class TestSlotIsReleasedAfterAnExpireBlip:
    """The end-to-end property: the slot the job took is the slot it gives back."""

    @staticmethod
    def _stub_job_path(monkeypatch: pytest.MonkeyPatch) -> None:
        async def _anoop(*args: object, **kwargs: object) -> None:
            return None

        @contextlib.asynccontextmanager
        async def _fake_session():  # type: ignore[no-untyped-def]
            yield object()

        async def _forever(*args: object, **kwargs: object) -> None:
            import asyncio

            await asyncio.Event().wait()

        async def _execute(*args: object, **kwargs: object) -> dict[str, Any]:
            return {"summary": {}, "test_results": [], "report_url": ""}

        monkeypatch.setattr(routes, "get_db_session", _fake_session)
        monkeypatch.setattr(routes.JobManager, "mark_processing", _anoop)
        monkeypatch.setattr(routes.JobManager, "mark_completed", _anoop)
        monkeypatch.setattr(routes.JobManager, "mark_failed", _anoop)
        monkeypatch.setattr(routes, "_heartbeat_loop", _forever)
        monkeypatch.setattr(routes._engine, "execute", _execute)

    async def test_process_job_releases_the_slot_despite_the_expire_blip(
        self, monkeypatch: pytest.MonkeyPatch, wire_redis
    ) -> None:
        from src.api.schemas import TestExecutionRequest
        from src.services.drain import JobDrain

        fake = wire_redis(_FakeRedis(expire_raises=True))
        self._stub_job_path(monkeypatch)

        await routes._process_job(
            "job-1",
            TestExecutionRequest(target_url="https://example.com", package_url="https://e/p.zip"),
            "01950000-0000-7000-8000-000000000001",
            None,
            drain=JobDrain(),
        )

        assert fake.zrem_calls == ["job-1"], (
            "the finished job must give its concurrency slot back; leaving it "
            "costs the tenant that slot until the deadline score prunes it"
        )
        assert fake.members == {}
