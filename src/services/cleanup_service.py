"""Cleanup service for stuck and old test execution jobs.

Multi-replica safety: every replica runs its own cleanup loop, but only ONE may
reap at a time — otherwise two pods could each evaluate the stale window and a
pod could fail a job another pod is actively running. :meth:`run_cleanup_pass`
gates the whole pass behind a Postgres transaction-level advisory lock
(``pg_try_advisory_xact_lock``): the first pod to acquire it does the work in a
single transaction (so the lock is held across both the stuck-reap and the
old-delete), and any other pod that cannot acquire it skips this tick.

The stale reap only fails ``processing`` jobs whose ``updated_at`` is older than
the cutoff. A running job heartbeats ``updated_at`` (see
``JobManager.touch_heartbeat``), so a genuinely live job — of any length — is
never reaped; only a crashed/stalled pod that stops heartbeating goes stale.

A SECOND reap covers ``pending``. ``create_job`` commits the row before the
background task that would call ``mark_processing`` ever runs, so a hard kill in
that window used to strand the row permanently — ``pending`` matched neither the
stuck reap nor the retention delete. Its cutoff is separate and far longer
(6h vs 30 min) and clocks on ``created_at``, because ``pending`` is unheartbeated
and is a legitimate state a live job occupies for a moment.

Artifact retention: deleting the ROW is only half a reap. Every run persists a
local scratch dir ``artifacts_root_dir/<job_id>`` and an object-store
``{job_id}/`` prefix (report tree + per-scenario clips), and nothing used to
remove either — so the disk and the bucket grew without bound, and dropping the
row ORPHANED the objects permanently because the id needed to find them was
gone. The old-job reap therefore purges a job's artifacts in the same pass,
BEFORE the row disappears. The purge is strictly best-effort: a storage error
is logged and the DB reap continues (an un-deleted object is a leak; a skipped
row reap is an unbounded table).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import get_settings

if TYPE_CHECKING:
    from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

# Stable, service-wide advisory-lock key so every FTR replica contends on the
# same lock (an arbitrary constant unique to this service's cleanup loop).
_CLEANUP_ADVISORY_LOCK_KEY = 0x46_54_52_43  # "FTRC"

# ---------------------------------------------------------------------------
# SECURITY DEFINER reaper DDL — single source for the RLS migration + tests.
# ---------------------------------------------------------------------------
#
# Why SQL functions and not plain ORM UPDATE/DELETE: in prod the runtime connects
# as the restricted, non-owner role ``w2v_ftr_app`` under
# ``TENANT_RLS_MODE=enforce``. The cleanup pass is CROSS-TENANT — it sweeps every
# tenant's stuck/old jobs — and has no tenant context, so a plain
# ``UPDATE test_execution_jobs ...`` / ``DELETE ...`` would be RLS-scoped to the
# (empty) ``app.current_tenant`` GUC and touch 0 rows. That is the "reaper-grant
# trap": CI never catches it because CI runs as the table owner (owner-bypass under
# ENABLE-not-FORCE). These ``SECURITY DEFINER`` functions are owned by the table
# owner (``w2v_service`` in prod, the owning CI/superuser role in tests) so their
# bodies bypass RLS and see every tenant's rows. EXECUTE is revoked from PUBLIC and
# granted to ``w2v_ftr_app`` in a ``pg_roles``-guarded ``DO`` block (CI-safe: the
# role only exists in prod). The DDL lives here — next to the loop that calls the
# functions — so the alembic migration and the test schema-bootstrap share one
# source of truth for the function names + bodies (à la content_service's
# reconciler.py).
REAP_STUCK_FN_NAME = "reap_stuck_test_execution_jobs"
DELETE_OLD_FN_NAME = "delete_old_test_execution_jobs"

_CREATE_REAP_STUCK_FN = """
CREATE OR REPLACE FUNCTION reap_stuck_test_execution_jobs(stuck_older interval)
  RETURNS integer
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
DECLARE
  v_count integer;
BEGIN
  UPDATE test_execution_jobs
     SET status = 'failed',
         error_message = 'Job stuck (no heartbeat — producing pod likely crashed)',
         completed_at = now(),
         updated_at = now()
   WHERE status = 'processing'
     AND updated_at < now() - stuck_older;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$fn$
"""

REAP_NEVER_STARTED_FN_NAME = "reap_never_started_test_execution_jobs"

# ``pending`` is a THIRD strand class, and until this function existed nothing
# reclaimed it. ``create_job`` COMMITS the row as ``pending`` and only then hands
# ``_process_job`` to ``BackgroundTasks``, so anything that kills the process
# between those two points — SIGKILL / grace-period expiry / kubelet eviction, a
# ``CancelledError`` (a ``BaseException``, so ``_process_job``'s ``except
# Exception`` never sees it) raised before ``mark_processing``, or the task being
# dropped before ``drain.register`` — leaves a ``pending`` row forever: the
# graceful drain never saw it, ``reap_stuck_test_execution_jobs`` matches only
# ``processing``, and the retention delete matches only ``completed``/``failed``.
#
# ⛔ The cutoff MUST stay separate from — and far longer than — the stuck cutoff.
# ``processing`` is heartbeated, so 30 min of silence there proves the producer is
# dead. ``pending`` is a LEGITIMATE state a live job occupies between the commit
# and ``mark_processing``, and it is NOT heartbeated, so collapsing the two
# cutoffs would fail out queued work that is about to run. Default 6h.
#
# It clocks on ``created_at``, deliberately, NOT on ``updated_at``: a pending row
# has no heartbeat, so ``updated_at`` measures nothing about whether the job was
# picked up. ``COALESCE`` only guards the row-inserted-out-of-band case where
# ``created_at`` is NULL — an unmeasurable age must not become an un-reclaimable
# row (that is the same silent strand this function exists to close).
_CREATE_REAP_NEVER_STARTED_FN = """
CREATE OR REPLACE FUNCTION reap_never_started_test_execution_jobs(never_started_older interval)
  RETURNS integer
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
DECLARE
  v_count integer;
BEGIN
  UPDATE test_execution_jobs
     SET status = 'failed',
         error_message = 'Job never started (accepted but the runner process died'
                         || ' before execution began — there is no retry queue;'
                         || ' re-submit with a new Idempotency-Key)',
         completed_at = now(),
         updated_at = now()
   WHERE status = 'pending'
     AND COALESCE(created_at, updated_at) < now() - never_started_older;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$fn$
"""

LIST_OLD_FN_NAME = "list_old_test_execution_jobs"

# The artifact purge needs the ids of the rows the reap is ABOUT to delete, and a
# plain SELECT here hits exactly the trap the delete function exists to avoid:
# under TENANT_RLS_MODE=enforce the restricted runtime role would see the empty
# ``app.current_tenant`` GUC and read back 0 ids, silently purging nothing while
# the rows vanished. So the lister is SECURITY DEFINER too, and its predicate is
# a VERBATIM copy of the delete's — if the two ever drift, the reap deletes rows
# whose artifacts were never listed (a permanent orphan).
_CREATE_LIST_OLD_FN = """
CREATE OR REPLACE FUNCTION list_old_test_execution_jobs(retention interval)
  RETURNS TABLE(job_id text)
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
BEGIN
  RETURN QUERY
  SELECT t.job_id::text
    FROM test_execution_jobs t
   WHERE t.status IN ('completed', 'failed')
     AND t.completed_at < now() - retention;
END;
$fn$
"""

_CREATE_DELETE_OLD_FN = """
CREATE OR REPLACE FUNCTION delete_old_test_execution_jobs(retention interval)
  RETURNS integer
  LANGUAGE plpgsql SECURITY DEFINER SET search_path FROM CURRENT AS $fn$
DECLARE
  v_count integer;
BEGIN
  DELETE FROM test_execution_jobs
   WHERE status IN ('completed', 'failed')
     AND completed_at < now() - retention;
  GET DIAGNOSTICS v_count = ROW_COUNT;
  RETURN v_count;
END;
$fn$
"""

_REVOKE_REAP_STUCK_FN = (
    "REVOKE EXECUTE ON FUNCTION reap_stuck_test_execution_jobs(interval) FROM PUBLIC"
)
_REVOKE_DELETE_OLD_FN = (
    "REVOKE EXECUTE ON FUNCTION delete_old_test_execution_jobs(interval) FROM PUBLIC"
)
_REVOKE_LIST_OLD_FN = (
    "REVOKE EXECUTE ON FUNCTION list_old_test_execution_jobs(interval) FROM PUBLIC"
)
_REVOKE_REAP_NEVER_STARTED_FN = (
    "REVOKE EXECUTE ON FUNCTION reap_never_started_test_execution_jobs(interval) FROM PUBLIC"
)

# Guarded so the migration is CI-safe: ``w2v_ftr_app`` only exists in prod (created
# out-of-band in the Stage-A role cutover), not in CI/tests.
_GRANT_REAP_STUCK_FN = """
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_ftr_app') THEN
    GRANT EXECUTE ON FUNCTION reap_stuck_test_execution_jobs(interval) TO w2v_ftr_app;
  END IF;
END
$do$
"""

_GRANT_DELETE_OLD_FN = """
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_ftr_app') THEN
    GRANT EXECUTE ON FUNCTION delete_old_test_execution_jobs(interval) TO w2v_ftr_app;
  END IF;
END
$do$
"""

_GRANT_LIST_OLD_FN = """
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_ftr_app') THEN
    GRANT EXECUTE ON FUNCTION list_old_test_execution_jobs(interval) TO w2v_ftr_app;
  END IF;
END
$do$
"""

_GRANT_REAP_NEVER_STARTED_FN = """
DO $do$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'w2v_ftr_app') THEN
    GRANT EXECUTE ON FUNCTION reap_never_started_test_execution_jobs(interval)
      TO w2v_ftr_app;
  END IF;
END
$do$
"""

_DROP_LIST_OLD_FN = "DROP FUNCTION IF EXISTS list_old_test_execution_jobs(interval)"
_DROP_REAP_STUCK_FN = "DROP FUNCTION IF EXISTS reap_stuck_test_execution_jobs(interval)"
_DROP_DELETE_OLD_FN = "DROP FUNCTION IF EXISTS delete_old_test_execution_jobs(interval)"
_DROP_REAP_NEVER_STARTED_FN = (
    "DROP FUNCTION IF EXISTS reap_never_started_test_execution_jobs(interval)"
)


def create_reap_function_statements() -> list[str]:
    """DDL to (idempotently) create the reaper functions only — used by the test bootstrap."""
    return [
        _CREATE_REAP_STUCK_FN,
        _CREATE_DELETE_OLD_FN,
        _CREATE_LIST_OLD_FN,
        _CREATE_REAP_NEVER_STARTED_FN,
    ]


def upgrade_statements() -> list[str]:
    """Full forward DDL for the alembic migration: create + lock down + guarded grant."""
    return [
        _CREATE_REAP_STUCK_FN,
        _CREATE_DELETE_OLD_FN,
        _CREATE_LIST_OLD_FN,
        _REVOKE_REAP_STUCK_FN,
        _REVOKE_DELETE_OLD_FN,
        _REVOKE_LIST_OLD_FN,
        _GRANT_REAP_STUCK_FN,
        _GRANT_DELETE_OLD_FN,
        _GRANT_LIST_OLD_FN,
    ]


def list_old_fn_upgrade_statements() -> list[str]:
    """Forward DDL for the ``list_old_test_execution_jobs`` function ALONE.

    Migration 003 already shipped the other two functions to prod, so the
    artifact-purge lister needs its own migration (005) rather than a re-run of
    :func:`upgrade_statements`.
    """
    return [_CREATE_LIST_OLD_FN, _REVOKE_LIST_OLD_FN, _GRANT_LIST_OLD_FN]


def list_old_fn_downgrade_statements() -> list[str]:
    """Reverse DDL for :func:`list_old_fn_upgrade_statements`."""
    return [_DROP_LIST_OLD_FN]


def never_started_fn_upgrade_statements() -> list[str]:
    """Forward DDL for ``reap_never_started_test_execution_jobs`` ALONE (migration 008).

    A NEW function rather than a widened ``reap_stuck_test_execution_jobs``: the two
    arms need INDEPENDENT cutoffs (30 min heartbeat-gated vs 6h wall-clock), and
    adding a second argument to the shipped function would need a
    ``DROP FUNCTION`` + re-``GRANT`` dance (``CREATE OR REPLACE`` cannot change
    arity, and the recreated function is born with an EMPTY ACL — the reaper would
    then hit "permission denied for function" in prod while CI, running as the
    owner, stayed green).
    """
    return [
        _CREATE_REAP_NEVER_STARTED_FN,
        _REVOKE_REAP_NEVER_STARTED_FN,
        _GRANT_REAP_NEVER_STARTED_FN,
    ]


def never_started_fn_downgrade_statements() -> list[str]:
    """Reverse DDL for :func:`never_started_fn_upgrade_statements`."""
    return [_DROP_REAP_NEVER_STARTED_FN]


def downgrade_statements() -> list[str]:
    """Reverse DDL for the alembic migration: drop the functions."""
    return [_DROP_REAP_STUCK_FN, _DROP_DELETE_OLD_FN, _DROP_LIST_OLD_FN]


class CleanupService:
    """Handles cleanup of stuck and expired jobs, and of their artifacts."""

    def __init__(
        self,
        storage: StorageService | None = None,
        artifacts_root: str | Path | None = None,
    ) -> None:
        """``storage``/``artifacts_root`` default to the configured ones.

        Both are injectable so a test can point the purge at a tmp dir and a
        local-backend store without touching global config.
        """
        self._storage = storage
        self._artifacts_root_override = artifacts_root

    # --- artifact purge ------------------------------------------------------

    def _artifacts_root(self) -> Path | None:
        """The configured artifacts root, or ``None`` if it is unusable.

        Fail CLOSED on a missing/blank/filesystem-root setting. An empty
        ``artifacts_root_dir`` would make ``Path("") / job_id`` a *relative*
        path resolved against the process CWD, and ``"/"`` would put the purge
        one bad job id away from deleting a system directory. Neither is a
        directory this service ever wrote to, so refusing costs nothing.
        """
        raw = self._artifacts_root_override
        if raw is None:
            raw = get_settings().execution.artifacts_root_dir
        text_root = str(raw).strip()
        if not text_root:
            logger.warning("artifacts_root_dir is empty — refusing to purge artifacts")
            return None
        root = Path(text_root)
        if not root.is_absolute() or root.resolve() == Path(root.anchor):
            logger.warning("Refusing to purge artifacts under unsafe root %r", text_root)
            return None
        return root

    def _job_artifact_dir(self, root: Path, job_id: str) -> Path | None:
        """Resolve ``root/<job_id>``, or ``None`` if it escapes ``root``.

        ``job_id`` comes from the DB, but it is caller-chosen at submit time, so
        it is treated as untrusted: an id of ``../..`` or an absolute path would
        otherwise aim ``rmtree`` outside the artifacts tree. The check is done on
        the RESOLVED paths (symlinks included) and requires a strict
        parent-of relationship — equality to ``root`` itself is rejected too,
        since that would delete every job's artifacts at once.
        """
        candidate = (root / job_id).resolve()
        resolved_root = root.resolve()
        if candidate == resolved_root or resolved_root not in candidate.parents:
            logger.error(
                "Refusing to purge artifacts for job_id %r — resolves outside %s",
                job_id,
                resolved_root,
            )
            return None
        return candidate

    def _purge_job_dir(self, job_id: str) -> bool:
        """Remove this job's local artifact dir. Never raises."""
        root = self._artifacts_root()
        if root is None:
            return False
        target = self._job_artifact_dir(root, job_id)
        if target is None:
            return False
        try:
            shutil.rmtree(target)
        except FileNotFoundError:
            # Expected on the common path: the dir is pod-local scratch, so a
            # job produced by another replica (or already purged) simply has none
            # here. Not an error.
            return False
        except OSError as exc:
            logger.warning("Failed to purge artifact dir %s: %r", target, exc)
            return False
        logger.info("Purged artifact dir %s", target)
        return True

    def _get_storage(self) -> StorageService | None:
        if self._storage is None:
            from src.services.storage_service import StorageService as _StorageService

            self._storage = _StorageService()
        return self._storage

    async def purge_job_artifacts(self, job_id: str) -> None:
        """Delete a job's local dir AND its ``{job_id}/`` object prefix.

        Best-effort by contract: EVERY failure is logged and swallowed. This runs
        immediately before the row reap, and letting a MinIO blip abort that reap
        would trade a bounded leak (objects) for an unbounded one (rows).
        """
        await asyncio.to_thread(self._purge_job_dir, job_id)
        try:
            storage = self._get_storage()
            if storage is not None:
                await storage.delete_job_objects(job_id)
        except Exception as exc:  # noqa: BLE001 - purge must never abort the reap
            logger.warning("Failed to purge object storage for job %s: %r", job_id, exc)

    @staticmethod
    async def _list_old_stmt(db: AsyncSession, retention_days: int) -> list[str]:
        """Ids of the rows :meth:`_delete_old_stmt` is about to delete.

        Must be called BEFORE the delete, in the same transaction: afterwards the
        ids are gone and the artifacts are unreachable forever.
        """
        result = await db.execute(
            text(f"SELECT {LIST_OLD_FN_NAME}(make_interval(days => :days))"),
            {"days": retention_days},
        )
        return [str(row[0]) for row in result.all() if row[0] is not None]

    async def _purge_artifacts_for_old_jobs(self, db: AsyncSession, retention_days: int) -> int:
        """Purge artifacts for every row the retention window is about to reap."""
        try:
            job_ids = await self._list_old_stmt(db, retention_days)
        except Exception as exc:  # noqa: BLE001 - purge must never abort the reap
            logger.warning("Could not list old jobs for artifact purge: %r", exc)
            return 0
        for job_id in job_ids:
            await self.purge_job_artifacts(job_id)
        return len(job_ids)

    @staticmethod
    async def _reap_stuck_stmt(db: AsyncSession, stuck_timeout_minutes: int) -> int:
        """Execute the stuck-job reap (no commit). Returns rows affected.

        Runs via the ``reap_stuck_test_execution_jobs`` SECURITY DEFINER function so
        the CROSS-TENANT sweep executes as the function owner and bypasses RLS. A
        plain ORM ``UPDATE`` here would, under ``TENANT_RLS_MODE=enforce`` + the
        restricted ``w2v_ftr_app`` role, be scoped to the empty tenant GUC and reap 0
        rows. Fails only ``processing`` jobs whose ``updated_at`` is older than the
        cutoff — a live job heartbeats ``updated_at`` and is therefore excluded.
        """
        result = await db.execute(
            text(f"SELECT {REAP_STUCK_FN_NAME}(make_interval(mins => :mins))"),
            {"mins": stuck_timeout_minutes},
        )
        return int(result.scalar_one())

    @staticmethod
    async def _reap_never_started_stmt(db: AsyncSession, never_started_timeout_hours: int) -> int:
        """Reclaim rows that were accepted but never picked up (no commit). Rows affected.

        The third strand class: ``create_job`` commits ``pending`` and only then
        schedules ``_process_job``, so a SIGKILL / eviction / pre-``mark_processing``
        cancellation leaves a row nothing else reclaims. SECURITY DEFINER for the same
        cross-tenant/RLS reason as :meth:`_reap_stuck_stmt`.

        ⛔ ``never_started_timeout_hours`` is DELIBERATELY a separate, much longer
        cutoff than ``stuck_timeout_minutes``. ``pending`` is unheartbeated and is a
        legitimate state a live job occupies; reusing the 30-minute processing cutoff
        would fail out queued work that is about to run.
        """
        result = await db.execute(
            text(f"SELECT {REAP_NEVER_STARTED_FN_NAME}(make_interval(hours => :hours))"),
            {"hours": never_started_timeout_hours},
        )
        return int(result.scalar_one())

    @staticmethod
    async def _delete_old_stmt(db: AsyncSession, retention_days: int) -> int:
        """Execute the old-job delete (no commit). Returns rows affected.

        Runs via the ``delete_old_test_execution_jobs`` SECURITY DEFINER function for
        the same cross-tenant / RLS-bypass reason as :meth:`_reap_stuck_stmt`.
        """
        result = await db.execute(
            text(f"SELECT {DELETE_OLD_FN_NAME}(make_interval(days => :days))"),
            {"days": retention_days},
        )
        return int(result.scalar_one())

    async def cleanup_stuck_jobs(
        self,
        db: AsyncSession,
        stuck_timeout_minutes: int = 30,
    ) -> int:
        """Mark processing jobs older than timeout as failed (commits)."""
        n = await self._reap_stuck_stmt(db, stuck_timeout_minutes)
        await db.commit()
        db.expire_all()  # reaper mutates out-of-band (raw SQL); refresh the ORM view
        return n

    async def cleanup_never_started_jobs(
        self,
        db: AsyncSession,
        never_started_timeout_hours: int = 6,
    ) -> int:
        """Mark never-started ``pending`` jobs older than the cutoff as failed (commits)."""
        n = await self._reap_never_started_stmt(db, never_started_timeout_hours)
        await db.commit()
        db.expire_all()  # reaper mutates out-of-band (raw SQL); refresh the ORM view
        return n

    async def cleanup_old_jobs(
        self,
        db: AsyncSession,
        retention_days: int = 14,
    ) -> int:
        """Delete jobs older than retention period, and their artifacts (commits)."""
        await self._purge_artifacts_for_old_jobs(db, retention_days)
        n = await self._delete_old_stmt(db, retention_days)
        await db.commit()
        db.expire_all()  # deleter mutates out-of-band (raw SQL); refresh the ORM view
        return n

    async def run_cleanup_pass(
        self,
        db: AsyncSession,
        *,
        stuck_timeout_minutes: int = 30,
        never_started_timeout_hours: int = 6,
        retention_days: int = 14,
    ) -> tuple[int, int, int] | None:
        """Run one cleanup pass under a transaction-level advisory lock.

        Only one replica reaps at a time. Returns
        ``(stuck_failed, never_started_failed, old_deleted)`` if this pod won the
        lock and did the work, or ``None`` if another pod holds the lock this tick
        (skip, try again next tick). All three arms run in the SAME transaction so
        the advisory lock is held across them, then a single commit releases it.
        """
        acquired = cast(
            bool,
            (
                await db.execute(
                    text("SELECT pg_try_advisory_xact_lock(:k)"),
                    {"k": _CLEANUP_ADVISORY_LOCK_KEY},
                )
            ).scalar(),
        )
        if not acquired:
            # Release the just-started transaction so the lock probe does not
            # linger; another pod is reaping this tick.
            await db.rollback()
            return None

        stuck = await self._reap_stuck_stmt(db, stuck_timeout_minutes)
        never_started = await self._reap_never_started_stmt(db, never_started_timeout_hours)
        # Artifacts FIRST: after the delete the ids are gone and the objects are
        # orphaned with nothing left to find them by.
        await self._purge_artifacts_for_old_jobs(db, retention_days)
        old = await self._delete_old_stmt(db, retention_days)
        await db.commit()  # releases the xact advisory lock
        # The SECURITY DEFINER reaper mutates rows out-of-band (raw SQL, invisible to
        # the ORM identity map). Expire so any subsequent read on this session reflects
        # DB truth rather than a stale in-session copy — the previous ORM ``update()``
        # synchronized the session implicitly; the function call does not.
        db.expire_all()
        return stuck, never_started, old
