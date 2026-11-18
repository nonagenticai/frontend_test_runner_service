"""Redis-based sliding window rate limiter.

A single Lua script evaluates the sliding-window count, the concurrent-job
count, and (on success) writes the new window entry. This makes the
SLIDING-WINDOW (RPM) admission atomic — the zcard/zadd race that let two
concurrent requests both observe an under-limit window count and both pass
is eliminated for the window key.

⚠️ The CONCURRENCY cap is NOT yet fully atomic. The script only READS the
concurrent set (``ZCARD``) at admission; the actual ``ZADD`` of the job's slot
happens later, out-of-band, in the detached ``_process_job`` worker (after the
HTTP response returns). So a tight BURST of ``/execute`` calls from one tenant
that all admit before any worker registers its slot can transiently exceed
``concurrent_jobs_per_tenant`` (a soft over-admit, self-correcting as the
workers register + the deadline-prune runs). TODO: reserve the slot atomically
inside this Lua at admission time (a request-scoped placeholder member the
worker then adopts) to make the concurrency cap a hard limit under burst.

The concurrent-job set (``test_runner:concurrent:{tenant_id}:jobs``) is a
Redis SORTED SET of running ``job_id``s, each scored by its expiry DEADLINE
(``now + max_run_seconds + buffer``). The worker entry point ``ZADD``s a job
on start and ``ZREM``s it in a ``finally`` block. The admission script prunes
members whose deadline has passed (``ZREMRANGEBYSCORE … now``) before counting
(``ZCARD``) — so a worker that is HARD-KILLED (SIGKILL on pod eviction / OOM /
wall-timeout killpg) before its ``finally`` runs cannot leak a slot forever:
its member self-expires at its own deadline without wrongly pruning a live
job's slot. (An earlier integer ``INCR``/``DECR`` counter leaked permanently on
hard-kill because its 1h TTL was reset on every ``INCR``.)
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from typing import Any, Final, cast

from fastapi import Request
from redis.exceptions import RedisError
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from src.cache.redis_client import get_redis_client
from src.config import get_settings
from src.errors import envelope_response

logger = logging.getLogger(__name__)


# Lua script: atomic sliding-window + concurrent-cap admission control.
#
# KEYS[1] = window sorted-set key
# KEYS[2] = concurrent running-jobs sorted-set key (member=job_id, score=deadline)
# ARGV[1] = now (float seconds)
# ARGV[2] = window seconds (e.g. 60)
# ARGV[3] = rpm limit
# ARGV[4] = concurrent limit
# ARGV[5] = window TTL seconds (e.g. 120)
# ARGV[6] = unique member for this request's window entry (see ``window_member``)
#
# Returns:
#   {"ok",        new_count}                     -> admit
#   {"rate",      current_count, oldest_score}   -> 429 sliding-window
#   {"concurrent", current_concurrent}           -> 429 concurrent-cap
#
# ``oldest_score`` is the timestamp of the OLDEST live entry in the window,
# returned as a STRING (a Lua number would be truncated to an integer by the
# Lua->RESP conversion, losing the sub-second part). It is what lets the caller
# compute an honest ``Retry-After``: that entry ages out of the window at
# ``oldest_score + window``, which is the first instant a slot actually frees.
_ADMIT_SCRIPT: Final[str] = """
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local rpm_limit = tonumber(ARGV[3])
local concurrent_limit = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', KEYS[1], 0, now - window)
local count = redis.call('ZCARD', KEYS[1])
if count >= rpm_limit then
    local oldest = redis.call('ZRANGE', KEYS[1], 0, 0, 'WITHSCORES')
    -- ZRANGE ... WITHSCORES reaches Lua in one of TWO shapes, and which one
    -- depends on the RESP protocol the surrounding runtime speaks:
    --   RESP2 -> a FLAT array  {member, score}      -> score is oldest[2]
    --   RESP3 -> NESTED pairs  {{member, score}}    -> score is oldest[1][2]
    -- Handle both. Reading only oldest[2] silently yields nil under RESP3,
    -- which collapses oldest_score to '' and makes retry_after_for_window
    -- take its missing-score fallback (a blanket 60) instead of the honest
    -- time-to-free. Shape-tolerance here beats pinning the client protocol.
    local oldest_score = ''
    if type(oldest[1]) == 'table' then
        if oldest[1][2] then
            oldest_score = tostring(oldest[1][2])
        end
    elseif oldest[2] then
        oldest_score = tostring(oldest[2])
    end
    return {'rate', count, oldest_score}
end

-- Prune running-job members whose expiry deadline has passed (a hard-killed
-- worker that never ZREM'd its slot), then count the survivors. A live job's
-- deadline is in the future, so it is never pruned.
redis.call('ZREMRANGEBYSCORE', KEYS[2], '-inf', now)
local concurrent = redis.call('ZCARD', KEYS[2])
if concurrent >= concurrent_limit then
    return {'concurrent', concurrent}
end

redis.call('ZADD', KEYS[1], now, ARGV[6])
redis.call('EXPIRE', KEYS[1], ttl)
return {'ok', count + 1}
"""


# Sliding-window length. ``requests_per_minute`` is per THIS many seconds.
_WINDOW_SECONDS: Final[int] = 60
# TTL on the window key — 2x the window, so an idle tenant's key self-collects.
_WINDOW_TTL_SECONDS: Final[int] = 120


def retry_after_for_window(now: float, oldest_score: str | float | bytes | None) -> int:
    """Seconds until the sliding window frees a slot, as an RFC 9110 integer.

    The oldest entry in the window ages out at ``oldest_score + window``; that
    is the first instant ``ZREMRANGEBYSCORE`` drops it and admission can pass.
    Rounded UP (a client that retries a fraction of a second early just 429s
    again), floored at 1 (``Retry-After: 0`` invites a hot loop), and capped at
    the window length (nothing can ever require a longer wait than one full
    window). Falls back to the full window if the score is missing/garbage —
    the honest worst case, never 0.

    EVERY step that touches the raw Redis value lives inside the ``try``. The
    score is attacker-influenceable (anything can ``ZADD`` a member into the
    window key) and each stage has its own failure mode, none of them
    ``ValueError``: ``.decode()`` raises ``UnicodeDecodeError`` on invalid
    UTF-8, ``float()`` raises ``TypeError`` on a non-str/bytes value, and
    ``math.ceil()`` raises ``OverflowError`` on ``inf``/``1e400``/``-inf`` and
    ``ValueError`` on ``nan``. With the decode and the ceil outside a
    ``ValueError``-only handler, a single poisoned score turned a 429 into a
    500 — the exact failure this helper exists to prevent. The return is always
    a plain ``int`` in ``[1, _WINDOW_SECONDS]``.
    """
    if oldest_score is None or oldest_score == "":
        return _WINDOW_SECONDS
    try:
        if isinstance(oldest_score, bytes):
            oldest_score = oldest_score.decode()
        expires_at = float(oldest_score) + _WINDOW_SECONDS
        return max(1, min(_WINDOW_SECONDS, math.ceil(expires_at - now)))
    except ValueError, TypeError, OverflowError, UnicodeError:
        return _WINDOW_SECONDS


def retry_after_for_concurrency() -> int:
    """Seconds hint for the concurrent-job cap, as an RFC 9110 integer.

    A concurrency slot frees only when a running Playwright job FINISHES, which
    the limiter cannot predict: the only bound it holds is each slot's expiry
    deadline (``max_run_seconds`` + buffer ≈ 25 min), and handing a client a
    1500-second hint would be worse than useless. So this is a deliberately
    conservative FIXED hint, grounded in
    ``ExecutionSettings.heartbeat_interval_seconds`` (default 60s): that is the
    cadence at which a running job's state is refreshed in this service, i.e.
    the shortest interval over which a retry can observe genuinely new
    occupancy. It also equals the sliding-window length, so a client that
    honours it never lands its retry inside the same rate window.
    """
    return max(1, get_settings().execution.heartbeat_interval_seconds)


def concurrent_jobs_key(tenant_id: str) -> str:
    """Redis key for the per-tenant running-jobs sorted set (member=job_id,
    score=expiry deadline). A NEW key suffix (``:jobs``) — distinct from the
    retired integer counter ``test_runner:concurrent:{tenant_id}`` — so a rolling
    deploy never mixes a string and a zset on one key (WRONGTYPE); the old int
    key is abandoned and self-expires."""
    return f"test_runner:concurrent:{tenant_id}:jobs"


def window_member(now: float) -> str:
    """Unique member for this request's sliding-window entry.

    The retired implementation used ``str(now)`` alone: two requests served on
    the same clock tick produced the SAME ZSET member, the second ``ZADD``
    overwrote the first rather than adding to it, and the window undercounted —
    so a burst over-admitted (measured: 40 of 100 admitted against an rpm of
    10). A unique suffix makes every admitted request its own member; the SCORE
    stays ``now``, which is what ``ZREMRANGEBYSCORE`` ages out.
    """
    return f"{now!r}:{uuid.uuid7()}"


def window_key(tenant_id: str) -> str:
    """Redis key for the per-tenant sliding-window sorted set."""
    return f"test_runner:ratelimit:{tenant_id}:window"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-tenant sliding window rate limiter using Redis."""

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Enforce per-tenant rate limits on test execution requests."""
        settings = get_settings()

        # Only rate-limit POST requests to test-runner endpoints
        if request.method != "POST" or not request.url.path.startswith("/api/test-runner/"):
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        requests_per_minute = settings.rate_limit.requests_per_minute
        concurrent_limit = settings.rate_limit.concurrent_jobs_per_tenant

        now = time.time()
        try:
            redis_client = get_redis_client()

            verdict = await cast(
                Any,
                redis_client.eval_script(
                    _ADMIT_SCRIPT,
                    keys=[window_key(tenant_id), concurrent_jobs_key(tenant_id)],
                    args=[
                        str(now),
                        str(_WINDOW_SECONDS),
                        str(requests_per_minute),
                        str(concurrent_limit),
                        str(_WINDOW_TTL_SECONDS),
                        window_member(now),
                    ],
                ),
            )
        except OSError, RuntimeError, RedisError:
            # Fail OPEN on ANY redis error (connection/script/WRONGTYPE/OOM) — a rate-limit
            # backend blip must never 500 a legitimate request. `RedisError` (redis-py's base)
            # does NOT subclass OSError, so it was previously uncaught → the middleware 500'd.
            logger.warning("Rate limiter failed (Redis unavailable), allowing request")
            return await call_next(request)

        # verdict = [tag, value]; redis-py decodes bytes when decode_responses=True,
        # but we accept both for safety.
        tag = verdict[0]
        if isinstance(tag, bytes):
            tag = tag.decode()

        # Both 429 branches carry `Retry-After` (RFC 9110 integer seconds). Without
        # it the orchestrator's client retried 3x with 1s/2s linear backoff (~3s
        # total) against a 60s window / a slot that frees only when a minutes-long
        # recording job ends — exhausting its retries every time (22 lost runs,
        # 2026-07-20..24). The envelope BODY is deliberately unchanged.
        if tag == "rate":
            oldest_score = verdict[2] if len(verdict) > 2 else None
            return envelope_response(
                f"Rate limit exceeded: {requests_per_minute} requests per minute",
                429,
                request=request,
                headers={"Retry-After": str(retry_after_for_window(now, oldest_score))},
            )
        if tag == "concurrent":
            return envelope_response(
                f"Concurrent job limit exceeded: {concurrent_limit} jobs",
                429,
                request=request,
                headers={"Retry-After": str(retry_after_for_concurrency())},
            )

        return await call_next(request)
