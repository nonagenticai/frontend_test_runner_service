"""Tests for ``_heartbeat_loop``'s exception handling.

The bug: the guarded call, ``JobManager.touch_heartbeat``, is a SQLAlchemy DB
write — but the handler caught ``RedisError`` (irrelevant here) instead of
``sqlalchemy.exc.SQLAlchemyError``. A DB blip during a heartbeat tick would
escape the try/except entirely, kill the detached background heartbeat task,
and the job would stop heartbeating — making it eligible to be falsely reaped
as orphaned while still running. This proves the handler now catches what the
DB layer actually raises, and that the loop survives the tick and keeps
beating rather than dying.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy.exc import OperationalError

from src.api.routes import _heartbeat_loop


@pytest.mark.unit
class TestHeartbeatLoopRecoversFromDbErrors:
    async def test_operational_error_is_caught_and_loop_keeps_beating(self) -> None:
        db_session = AsyncMock()

        @asynccontextmanager
        async def _fake_get_db_session():
            yield db_session

        settings = MagicMock()
        settings.execution.heartbeat_interval_seconds = 0  # tick as fast as possible

        # First tick: DB blip (OperationalError, a SQLAlchemyError subclass).
        # Second tick: succeeds — proves the loop is still alive afterward.
        touch_heartbeat = AsyncMock(
            side_effect=[
                OperationalError("stmt", {}, Exception("db gone")),
                None,
            ]
        )

        with (
            patch("src.api.routes.get_db_session", _fake_get_db_session),
            patch("src.api.routes.get_settings", return_value=settings),
            patch("src.api.routes.JobManager") as mock_jm,
        ):
            mock_jm.touch_heartbeat = touch_heartbeat

            task = asyncio.create_task(_heartbeat_loop("job-1"))
            try:
                # Give the loop two full ticks to run.
                for _ in range(50):
                    await asyncio.sleep(0)
                    if touch_heartbeat.call_count >= 2:
                        break
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

        assert touch_heartbeat.call_count >= 2, (
            "loop died after the first DB error instead of surviving it"
        )
