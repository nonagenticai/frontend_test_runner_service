"""Tests for cleanup service."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from sqlalchemy import text

from src.database.models import TestExecutionJob
from src.services.cleanup_service import _CLEANUP_ADVISORY_LOCK_KEY, CleanupService
from src.services.job_manager import JobManager
from tests.conftest import make_local_storage


class TestCleanupService:
    async def test_cleanup_stuck_jobs(self, db_session):
        """Jobs stuck in processing for too long should be marked failed."""
        stuck_time = datetime.now(UTC) - timedelta(minutes=60)
        job = TestExecutionJob(
            job_id="stuck-001",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            updated_at=stuck_time,
        )
        db_session.add(job)
        await db_session.commit()

        service = CleanupService()
        cleaned = await service.cleanup_stuck_jobs(db_session, stuck_timeout_minutes=30)
        assert cleaned == 1

        await db_session.refresh(job)
        assert job.status == "failed"
        assert "stuck" in (job.error_message or "").lower()

    async def test_no_stuck_jobs(self, db_session):
        """Recent processing jobs should not be cleaned up."""
        job = TestExecutionJob(
            job_id="recent-001",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            updated_at=datetime.now(UTC),
        )
        db_session.add(job)
        await db_session.commit()

        service = CleanupService()
        cleaned = await service.cleanup_stuck_jobs(db_session, stuck_timeout_minutes=30)
        assert cleaned == 0

    async def test_cleanup_old_jobs(self, db_session):
        """Completed/failed jobs past retention should be deleted."""
        old_time = datetime.now(UTC) - timedelta(days=30)
        job = TestExecutionJob(
            job_id="old-001",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="completed",
            target_url="https://example.com",
            completed_at=old_time,
        )
        db_session.add(job)
        await db_session.commit()

        service = CleanupService()
        deleted = await service.cleanup_old_jobs(db_session, retention_days=14)
        assert deleted == 1

    async def test_no_old_jobs(self, db_session):
        """Recent completed jobs should not be deleted."""
        job = TestExecutionJob(
            job_id="recent-002",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="completed",
            target_url="https://example.com",
            completed_at=datetime.now(UTC),
        )
        db_session.add(job)
        await db_session.commit()

        service = CleanupService()
        deleted = await service.cleanup_old_jobs(db_session, retention_days=14)
        assert deleted == 0


class TestRunCleanupPass:
    """run_cleanup_pass — advisory-lock-gated, lease-aware single-pass reap
    (multi-replica safety)."""

    async def test_stuck_job_reaped_exactly_once(self, db_session):
        """A genuinely stuck job (no heartbeat) is reaped, and a second pass is a
        no-op — not double-processed."""
        job = TestExecutionJob(
            job_id="reap-1",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            updated_at=datetime.now(UTC) - timedelta(minutes=60),
        )
        db_session.add(job)
        await db_session.commit()

        svc = CleanupService()
        first = await svc.run_cleanup_pass(db_session, stuck_timeout_minutes=30, retention_days=14)
        assert first == (1, 0, 0)
        second = await svc.run_cleanup_pass(db_session, stuck_timeout_minutes=30, retention_days=14)
        assert second == (0, 0, 0)

        fresh = await JobManager.get_job(db_session, "reap-1")
        assert fresh is not None
        assert fresh.status == "failed"

    async def test_live_heartbeating_job_not_reaped(self, db_session):
        """A job created long ago but whose updated_at was refreshed by a recent
        heartbeat is NEVER reaped, even past the 30-min cutoff."""
        job = TestExecutionJob(
            job_id="live-1",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            created_at=datetime.now(UTC) - timedelta(minutes=90),
            updated_at=datetime.now(UTC),  # heartbeated just now
        )
        db_session.add(job)
        await db_session.commit()

        outcome = await CleanupService().run_cleanup_pass(
            db_session, stuck_timeout_minutes=30, retention_days=14
        )
        assert outcome == (0, 0, 0)

        fresh = await JobManager.get_job(db_session, "live-1")
        assert fresh is not None
        assert fresh.status == "processing"

    async def test_advisory_lock_prevents_concurrent_reap(self, db_engine, db_session):
        """When another connection holds the advisory lock, run_cleanup_pass skips
        (returns None) and reaps nothing — only one replica reaps per tick."""
        stuck = TestExecutionJob(
            job_id="lock-1",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            updated_at=datetime.now(UTC) - timedelta(minutes=60),
        )
        db_session.add(stuck)
        await db_session.commit()

        holder = await db_engine.connect()
        try:
            # Hold the lock in an uncommitted transaction on a separate connection.
            await holder.execute(
                text("SELECT pg_advisory_xact_lock(:k)"),
                {"k": _CLEANUP_ADVISORY_LOCK_KEY},
            )

            outcome = await CleanupService().run_cleanup_pass(
                db_session, stuck_timeout_minutes=30, retention_days=14
            )
            assert outcome is None
        finally:
            await holder.rollback()
            await holder.close()

        # The stuck job survived — the losing pod did not reap it.
        fresh = await JobManager.get_job(db_session, "lock-1")
        assert fresh is not None
        assert fresh.status == "processing"


class TestArtifactRetention:
    """Reaping the ROW without the artifacts leaks the disk AND the bucket, and
    orphans the objects permanently — the job_id needed to find them dies with
    the row. The purge therefore runs in the same pass, before the delete."""

    @staticmethod
    def _old_job(job_id: str) -> TestExecutionJob:
        return TestExecutionJob(
            job_id=job_id,
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="completed",
            target_url="https://example.com",
            completed_at=datetime.now(UTC) - timedelta(days=30),
        )

    @staticmethod
    def _seed_artifacts(root: Path, job_id: str) -> Path:
        job_dir = root / job_id
        (job_dir / "playwright-report" / "data").mkdir(parents=True)
        (job_dir / "playwright-report" / "index.html").write_text("<html></html>")
        (job_dir / "clip-1.mp4").write_bytes(b"\x00" * 16)
        return job_dir

    async def test_reap_removes_the_jobs_artifact_dir(self, db_session, tmp_path):
        """The whole ``artifacts_root_dir/<job_id>`` tree goes with the row."""
        root = tmp_path / "jobs"
        root.mkdir()
        job_dir = self._seed_artifacts(root, "artifacts-1")
        keep_dir = self._seed_artifacts(root, "artifacts-keep")
        db_session.add(self._old_job("artifacts-1"))
        await db_session.commit()

        svc = CleanupService(storage=make_local_storage(tmp_path / "objects"), artifacts_root=root)
        assert await svc.cleanup_old_jobs(db_session, retention_days=14) == 1

        assert not job_dir.exists()
        assert keep_dir.exists()  # another job's artifacts are untouched

    async def test_reap_deletes_the_object_storage_prefix(self, db_session, tmp_path):
        """The ``{job_id}/`` object prefix goes too — and ONLY that job's."""
        objects = tmp_path / "objects"
        store = make_local_storage(objects)
        await store.store_bytes("artifacts-2", "playwright-report/index.html", b"<html>")
        await store.store_bytes("artifacts-2-sibling", "playwright-report/index.html", b"<html>")
        db_session.add(self._old_job("artifacts-2"))
        await db_session.commit()

        svc = CleanupService(storage=store, artifacts_root=tmp_path / "jobs")
        assert await svc.cleanup_old_jobs(db_session, retention_days=14) == 1

        assert await store.retrieve("artifacts-2", "playwright-report/index.html") is None
        # A prefix that merely STARTS WITH this job's id is a different job.
        assert (
            await store.retrieve("artifacts-2-sibling", "playwright-report/index.html") is not None
        )

    async def test_reap_tolerates_a_missing_artifact_dir(self, db_session, tmp_path):
        """The dir is pod-local scratch; a job produced by another replica has
        none here. That is the COMMON case, not an error — the row still goes."""
        root = tmp_path / "jobs"
        root.mkdir()
        db_session.add(self._old_job("artifacts-3"))
        await db_session.commit()

        svc = CleanupService(storage=make_local_storage(tmp_path / "objects"), artifacts_root=root)
        assert await svc.cleanup_old_jobs(db_session, retention_days=14) == 1
        assert await JobManager.get_job(db_session, "artifacts-3") is None

    def test_purge_refuses_a_job_id_that_escapes_the_root(self, tmp_path, caplog):
        """job_id is caller-influenced at submit time; a traversal must never
        aim rmtree outside the artifacts root."""
        root = tmp_path / "jobs"
        (root / "keep").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (outside / "precious.txt").write_text("do not delete")

        svc = CleanupService(artifacts_root=root)
        with caplog.at_level(logging.ERROR, logger="src.services.cleanup_service"):
            assert svc._purge_job_dir("../outside") is False
        assert (outside / "precious.txt").exists()
        assert "resolves outside" in caplog.text
        # The root ITSELF is refused too — that would delete every job at once.
        assert svc._purge_job_dir(".") is False
        assert (root / "keep").exists()

    def test_purge_refuses_an_empty_or_filesystem_root(self, tmp_path):
        """An empty root resolves against CWD; ``/`` is one bad id from disaster."""
        assert CleanupService(artifacts_root="")._purge_job_dir("job-1") is False
        assert CleanupService(artifacts_root="   ")._purge_job_dir("job-1") is False
        assert CleanupService(artifacts_root="/")._purge_job_dir("job-1") is False
        assert CleanupService(artifacts_root="relative/jobs")._purge_job_dir("job-1") is False

    async def test_storage_failure_does_not_abort_the_row_reap(self, db_session, tmp_path):
        """An un-deleted object is a bounded leak; a skipped row reap is an
        unbounded table. So a storage error is logged and the DELETE runs."""

        class _ExplodingStore:
            async def delete_job_objects(self, job_id: str) -> bool:
                raise RuntimeError("MinIO unreachable")

        root = tmp_path / "jobs"
        root.mkdir()
        job_dir = self._seed_artifacts(root, "artifacts-4")
        db_session.add(self._old_job("artifacts-4"))
        await db_session.commit()

        svc = CleanupService(storage=cast(Any, _ExplodingStore()), artifacts_root=root)
        assert await svc.cleanup_old_jobs(db_session, retention_days=14) == 1
        assert await JobManager.get_job(db_session, "artifacts-4") is None
        # The filesystem half still happened — the two halves are independent.
        assert not job_dir.exists()

    async def test_run_cleanup_pass_purges_before_deleting_the_row(self, db_session, tmp_path):
        """The full advisory-lock-gated pass, not just the standalone helper."""
        root = tmp_path / "jobs"
        root.mkdir()
        job_dir = self._seed_artifacts(root, "artifacts-5")
        db_session.add(self._old_job("artifacts-5"))
        await db_session.commit()

        svc = CleanupService(storage=make_local_storage(tmp_path / "objects"), artifacts_root=root)
        assert await svc.run_cleanup_pass(
            db_session, stuck_timeout_minutes=30, retention_days=14
        ) == (0, 0, 1)
        assert not job_dir.exists()

    async def test_a_job_inside_retention_keeps_its_artifacts(self, db_session, tmp_path):
        """The purge predicate must match the delete predicate exactly."""
        root = tmp_path / "jobs"
        root.mkdir()
        job_dir = self._seed_artifacts(root, "artifacts-6")
        job = self._old_job("artifacts-6")
        job.completed_at = datetime.now(UTC)
        db_session.add(job)
        await db_session.commit()

        svc = CleanupService(storage=make_local_storage(tmp_path / "objects"), artifacts_root=root)
        assert await svc.cleanup_old_jobs(db_session, retention_days=14) == 0
        assert job_dir.exists()


class TestNeverStartedReap:
    """The ``pending`` strand: rows accepted but never picked up by a runner.

    ``create_job`` COMMITS the row as ``pending`` and only then hands
    ``_process_job`` to ``BackgroundTasks``. A SIGKILL / eviction / pre-
    ``mark_processing`` ``CancelledError`` in that window used to leave the row at
    ``pending`` forever — the stuck reap matches only ``processing`` and the
    retention delete only ``completed``/``failed``.
    """

    @staticmethod
    def _pending(job_id: str, *, created_ago: timedelta, updated_ago: timedelta | None = None):
        now = datetime.now(UTC)
        return TestExecutionJob(
            job_id=job_id,
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="pending",
            target_url="https://example.com",
            created_at=now - created_ago,
            updated_at=now - (updated_ago if updated_ago is not None else created_ago),
        )

    async def test_pending_job_older_than_the_cutoff_is_reclaimed(self, db_session):
        """(a) A never-started row past the pending cutoff is failed out."""
        db_session.add(self._pending("never-started-1", created_ago=timedelta(hours=9)))
        await db_session.commit()

        n = await CleanupService().cleanup_never_started_jobs(
            db_session, never_started_timeout_hours=6
        )
        assert n == 1, (
            "a 9h-old 'pending' row was NOT reclaimed — nothing else in the service "
            "ever touches status='pending', so this row is stranded forever"
        )

        fresh = await JobManager.get_job(db_session, "never-started-1")
        assert fresh is not None
        assert fresh.status == "failed"
        assert "never started" in (fresh.error_message or "").lower()
        assert fresh.completed_at is not None

    async def test_pending_job_inside_the_cutoff_is_spared(self, db_session):
        """A freshly-queued row is left alone — ``pending`` is a legitimate state."""
        db_session.add(self._pending("never-started-2", created_ago=timedelta(minutes=5)))
        await db_session.commit()

        n = await CleanupService().cleanup_never_started_jobs(
            db_session, never_started_timeout_hours=6
        )
        assert n == 0

        fresh = await JobManager.get_job(db_session, "never-started-2")
        assert fresh is not None
        assert fresh.status == "pending"

    async def test_the_two_cutoffs_must_not_collapse(self, db_session):
        """(b) REGRESSION GUARD for the collapse.

        A row OLD by the 30-minute *processing* cutoff but YOUNG by the 6-hour
        *pending* cutoff must be SPARED. Widening the stuck reap to
        ``status IN ('processing','pending')`` — i.e. reusing the heartbeat-gated
        30-min cutoff for an unheartbeated state — fails out live queued work.
        """
        db_session.add(self._pending("never-started-3", created_ago=timedelta(minutes=90)))
        await db_session.commit()

        svc = CleanupService()
        outcome = await svc.run_cleanup_pass(
            db_session,
            stuck_timeout_minutes=30,
            never_started_timeout_hours=6,
            retention_days=14,
        )
        assert outcome == (0, 0, 0), (
            "a 90-minute-old 'pending' row was reclaimed: the pending cutoff has "
            "collapsed onto the 30-minute processing cutoff. 'pending' is NOT "
            "heartbeated, so this fails out work that is about to run."
        )

        fresh = await JobManager.get_job(db_session, "never-started-3")
        assert fresh is not None
        assert fresh.status == "pending"

    async def test_pending_arm_clocks_on_created_at_not_the_heartbeat_column(self, db_session):
        """(c) ``updated_at`` measures nothing for an unheartbeated row.

        Old by ``created_at`` (9h) but freshly-stamped ``updated_at``: it must
        still be reclaimed. Clocking the pending arm on ``updated_at`` would let
        any incidental row touch reset a strand's age indefinitely.
        """
        db_session.add(
            self._pending(
                "never-started-4",
                created_ago=timedelta(hours=9),
                updated_ago=timedelta(seconds=0),
            )
        )
        await db_session.commit()

        n = await CleanupService().cleanup_never_started_jobs(
            db_session, never_started_timeout_hours=6
        )
        assert n == 1, (
            "the pending arm is clocking on updated_at (the heartbeat column), not "
            "created_at — a pending row never heartbeats, so updated_at is not its age"
        )

        fresh = await JobManager.get_job(db_session, "never-started-4")
        assert fresh is not None
        assert fresh.status == "failed"

    async def test_a_processing_job_is_not_touched_by_the_pending_arm(self, db_session):
        """The arms stay disjoint: the pending reap must not double-fail a stuck job."""
        job = TestExecutionJob(
            job_id="never-started-5",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="processing",
            target_url="https://example.com",
            created_at=datetime.now(UTC) - timedelta(hours=9),
            updated_at=datetime.now(UTC),
        )
        db_session.add(job)
        await db_session.commit()

        n = await CleanupService().cleanup_never_started_jobs(
            db_session, never_started_timeout_hours=6
        )
        assert n == 0

        fresh = await JobManager.get_job(db_session, "never-started-5")
        assert fresh is not None
        assert fresh.status == "processing"

    async def test_run_cleanup_pass_reports_the_pending_arm(self, db_session):
        """The scheduled pass (not just the standalone helper) reclaims pending rows."""
        db_session.add(self._pending("never-started-6", created_ago=timedelta(hours=9)))
        await db_session.commit()

        outcome = await CleanupService().run_cleanup_pass(
            db_session,
            stuck_timeout_minutes=30,
            never_started_timeout_hours=6,
            retention_days=14,
        )
        assert outcome == (0, 1, 0)
