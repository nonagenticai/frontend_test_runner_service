"""Tests for the ``src.asgi`` database startup hook.

── The startup "Database connected" line must never carry the DSN password ──

REGRESSION: ``_startup_database`` logged ``settings.database.url`` raw — the
full ``postgresql+psycopg://<user>:<password>@host:port/db`` DSN — at INFO on
the connect path, so the live app password landed in this pod's stdout and in
anything shipping those logs, indefinitely.

The test drives the REAL ``_startup_database`` rather than the redactor, because
the defect was at the CALL SITE: a test that only exercises
``w2v_common.observability.database_endpoint`` passes for the whole duration of
the leak. That helper has its own unit tests in w2v_common; this one exists to
fail if anyone hands the raw url back to the logger here.
"""

from __future__ import annotations

import io
import logging
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI

# Wholly synthetic and shares no prefix with any real credential — this string
# is committed, printed by pytest on failure, and ends up in CI logs, i.e.
# exactly the places this change exists to keep secrets out of.
_FAKE_PASSWORD = "not-a-real-password-xyzzy"  # pragma: allowlist secret
_FAKE_USER = "ftr_app"
_FAKE_ENDPOINT = "pgbouncer.pgbouncer:6432/frontend_test_runner_service"
_FAKE_DSN = f"postgresql+psycopg://{_FAKE_USER}:{_FAKE_PASSWORD}@{_FAKE_ENDPOINT}"


async def test_startup_db_log_does_not_leak_the_dsn_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The emitted startup log record must not contain the DSN password."""
    from src import asgi

    monkeypatch.setattr(asgi.settings.database, "url", _FAKE_DSN)

    # A handler attached directly to the module logger, NOT ``caplog``. The
    # shared logging setup re-points the root handlers, so caplog's root handler
    # can observe nothing and every assertion below would pass vacuously.
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    asgi.logger.addHandler(handler)
    # ``setLevel`` rather than assigning ``.level``: ``Logger`` memoizes
    # ``isEnabledFor`` in ``_cache``, and only ``setLevel`` invalidates it. Under
    # pytest the root logger sits at WARNING, so the cached answer for INFO is
    # already False by the time this test runs and a bare attribute assignment
    # leaves the record suppressed — which reads exactly like a missing log line.
    previous_level = asgi.logger.level
    asgi.logger.setLevel(logging.INFO)

    app = FastAPI()
    try:
        # ``get_engine`` is patched at the ``src.asgi`` module scope exactly as
        # the ``app`` fixture in tests/conftest.py does it; nothing else in this
        # hook opens a connection.
        with patch("src.asgi.get_engine", new_callable=AsyncMock, return_value=AsyncMock()):
            await asgi._startup_database(app)
    finally:
        asgi.logger.removeHandler(handler)
        asgi.logger.setLevel(previous_level)

    captured = buffer.getvalue()
    # Guard against a vacuous pass: if the line never emitted, the assertions
    # below would hold for the wrong reason. The trailing colon matters — the
    # hook also logs "Database connected (schema managed by alembic)", which
    # carries no DSN and would satisfy a looser guard on its own.
    assert "Database connected: " in captured, (
        "the 'Database connected: <dsn>' line never emitted; the assertions would be vacuous"
    )
    assert _FAKE_PASSWORD not in captured, "DSN password leaked into the startup log"
    assert _FAKE_USER not in captured, "DSN username leaked into the startup log"
    # Still useful, or the next person deletes the log line instead of keeping it.
    assert _FAKE_ENDPOINT in captured


# ── The startup "Redis connected" line must never carry the Redis password ──
#
# REGRESSION (same class as the DSN leak above): ``_startup_redis`` logged
# ``settings.redis.url`` raw. A Redis URL is ``redis://[user]:<password>@host:port/db``,
# so every pod start wrote the live Redis password to stdout and to log
# aggregation. Fixed by routing the argument through
# ``w2v_common.observability.redis_endpoint`` (w2v-common 0.67.25).
#
# As above, this drives the REAL ``_startup_redis`` and not ``redis_endpoint``:
# the defect is at the CALL SITE, and a helper-only test stays green for the
# whole duration of the leak.

_FAKE_REDIS_USER = "ftr_cache"
_FAKE_REDIS_ENDPOINT = "redis.w2v:6379"
_FAKE_REDIS_URL = f"redis://{_FAKE_REDIS_USER}:{_FAKE_PASSWORD}@{_FAKE_REDIS_ENDPOINT}/3"


async def test_startup_redis_log_does_not_leak_the_redis_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The emitted 'Redis connected' record must not contain the Redis password."""
    from src import asgi

    monkeypatch.setattr(asgi.settings.redis, "url", _FAKE_REDIS_URL)

    # Own handler on the module logger, NOT ``caplog`` -- the shared logging
    # setup re-points the root handlers, so caplog would capture nothing and
    # every assertion below would pass vacuously.
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setLevel(logging.INFO)
    asgi.logger.addHandler(handler)
    # ``setLevel``, never ``monkeypatch.setattr(logger, "level", ...)``:
    # ``Logger`` memoizes ``isEnabledFor`` in ``_cache`` and only ``setLevel``
    # invalidates it. A stale cache entry suppresses the record entirely, which
    # reads exactly like a clean redaction.
    previous_level = asgi.logger.level
    asgi.logger.setLevel(logging.INFO)

    app = FastAPI()
    try:
        with (
            patch("src.asgi.init_redis", new_callable=AsyncMock, return_value=AsyncMock()),
            patch("src.asgi.get_redis_client", return_value=AsyncMock()),
        ):
            await asgi._startup_redis(app)
    finally:
        asgi.logger.removeHandler(handler)
        asgi.logger.setLevel(previous_level)

    captured = buffer.getvalue()
    # Positive control FIRST: a line that never emitted would satisfy every
    # negative assertion below for the wrong reason.
    assert "Redis connected" in captured, (
        "the 'Redis connected' line never emitted; the assertions would be vacuous"
    )
    assert _FAKE_PASSWORD not in captured, "Redis password leaked into the startup log"
    assert _FAKE_REDIS_USER not in captured, "Redis username leaked into the startup log"
    # Still useful, or the next person deletes the log line instead of keeping it.
    assert _FAKE_REDIS_ENDPOINT in captured
