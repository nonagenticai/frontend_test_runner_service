"""In-flight job registry + shutdown drain state.

Why this module exists
----------------------
Test runs execute as detached FastAPI ``BackgroundTasks`` (``_process_job``),
not as a queue a second pod can pick up. Nothing owns a job but the process
running it, so when that pod is deleted (HPA scale-down, rollout, eviction) the
run simply evaporates: the row stays ``processing``, stops heartbeating, and 30
minutes later the reaper rewrites it as
``"Job stuck (no heartbeat — producing pod likely crashed)"``. That message is
misleading — the pod did not crash, it was asked to stop and did not wait.

This registry is the missing piece: a process-wide count of the runs currently
in flight, plus a ``draining`` flag, so shutdown can (a) stop advertising
readiness, (b) WAIT for those runs, and (c) only then tear down Redis/DB.

Deliberately minimal: a set of job ids and an ``asyncio.Event`` that is SET
while the set is empty. No locks — every mutation happens on the single service
event loop.

Ownership: ONE ``JobDrain`` per ASGI app, created by ``_on_startup`` and stored
at ``app.state.job_drain``. It is deliberately NOT a module-level singleton —
that made the drain flag process-global state that leaked from any test which
ran the real lifespan into every test after it, and needed an autouse reset
fixture to paper over. Every consumer reaches it from the app object:
``src.api.health``/``src.api.routes`` via ``request.app``, ``src.asgi`` via the
``app`` argument its lifespan hooks already receive, and the SIGTERM handler via
a closure the startup hook binds it into.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


class JobDrain:
    """Tracks in-flight ``_process_job`` runs and the drain (SIGTERM) flag."""

    def __init__(self) -> None:
        self._in_flight: set[str] = set()
        self._draining = False
        # Set == nothing in flight. Starts set, so a drain with no work returns
        # immediately instead of burning the whole timeout.
        self._idle = asyncio.Event()
        self._idle.set()

    # -- drain flag ---------------------------------------------------------

    @property
    def draining(self) -> bool:
        """True once shutdown has begun; readiness must report NOT ready."""
        return self._draining

    def begin_drain(self) -> bool:
        """Mark the process as draining. Idempotent; returns True on the first call.

        Called from the SIGTERM handler (so readiness flips BEFORE uvicorn
        starts waiting on connections) and again from the lifespan shutdown
        hook, which is the only path in tests / non-signal shutdowns.
        """
        if self._draining:
            return False
        self._draining = True
        logger.warning(
            "Drain started — readiness will report 503; %d job(s) in flight: %s",
            len(self._in_flight),
            sorted(self._in_flight) or "none",
        )
        return True

    # -- in-flight registry -------------------------------------------------

    def register(self, job_id: str) -> None:
        """Record that ``job_id`` is executing in this process."""
        self._in_flight.add(job_id)
        self._idle.clear()

    def unregister(self, job_id: str) -> None:
        """Record that ``job_id`` has left this process (any outcome)."""
        self._in_flight.discard(job_id)
        if not self._in_flight:
            self._idle.set()

    def in_flight(self) -> list[str]:
        """Ids of the runs currently executing, sorted for stable logging."""
        return sorted(self._in_flight)

    # -- the wait -----------------------------------------------------------

    async def wait_for_idle(self, timeout: float) -> list[str]:
        """Block until nothing is in flight, or ``timeout`` seconds elapse.

        Returns the ABANDONED job ids: empty on a clean drain, otherwise the
        ids still running when the bound expired. The caller must treat a
        non-empty return as an error condition and say so loudly — silently
        proceeding to close the DB is exactly the bug this module fixes.
        """
        if not self._in_flight:
            return []
        try:
            await asyncio.wait_for(self._idle.wait(), timeout)
        except TimeoutError:
            return self.in_flight()
        return []


def get_job_drain(app: FastAPI) -> JobDrain | None:
    """Return the drain this app instance owns, or ``None`` if it has none.

    The drain is created by ``_on_startup`` and stored on ``app.state``, so a
    ``None`` here means the ASGI lifespan has not run (or failed before its
    first step). Callers must treat that as NOT READY rather than as "not
    draining": the alternative — assuming a missing registry means healthy —
    is exactly how a wiring regression would silently disable the whole gate.
    """
    return getattr(app.state, "job_drain", None)
