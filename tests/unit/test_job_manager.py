"""Tests for JobManager."""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.services.job_manager import JobManager, JobStateConflictError


class TestJobManager:
    async def test_create_job(self, db_session):
        job, created = await JobManager.create_job(
            db_session,
            job_id="test-job-001",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
            bdd_job_id="bdd-123",
            config={"browsers": ["chromium"]},
        )
        assert created is True
        assert job.job_id == "test-job-001"
        assert job.tenant_id == "01950000-0000-7000-8000-000000000001"
        assert job.target_url == "https://example.com"
        assert job.bdd_job_id == "bdd-123"
        assert job.status == "pending"

    async def test_create_job_dedupes_on_idempotency_key(self, db_session):
        """A re-submit with the same (tenant, idempotency_key) returns the ORIGINAL
        job with created=False — so demo_video_record retrying after a resumable
        timeout does NOT spawn a second paid recording run."""
        _, c1 = await JobManager.create_job(
            db_session,
            job_id="idem-job-A",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
            idempotency_key="run-abc:demo_video_record",
        )
        assert c1 is True
        second, c2 = await JobManager.create_job(
            db_session,
            job_id="idem-job-B",  # a different fresh job_id (as the route would mint)
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
            idempotency_key="run-abc:demo_video_record",
        )
        assert c2 is False
        assert second.job_id == "idem-job-A"  # the ORIGINAL, not idem-job-B

    async def test_create_job_idempotency_key_is_tenant_scoped(self, db_session):
        """Same key under a DIFFERENT tenant is a distinct job (no cross-tenant dedup)."""
        _, c1 = await JobManager.create_job(
            db_session,
            job_id="idem-t1",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
            idempotency_key="shared-key",
        )
        _, c2 = await JobManager.create_job(
            db_session,
            job_id="idem-t2",
            tenant_id="01950000-0000-7000-8000-000000000002",
            target_url="https://example.com",
            idempotency_key="shared-key",
        )
        assert c1 is True
        assert c2 is True  # different tenant → not deduped

    async def test_get_job(self, db_session):
        await JobManager.create_job(
            db_session,
            job_id="test-job-002",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        job = await JobManager.get_job(db_session, "test-job-002")
        assert job is not None
        assert job.job_id == "test-job-002"

    async def test_get_nonexistent_job(self, db_session):
        job = await JobManager.get_job(db_session, "nonexistent")
        assert job is None

    async def test_mark_processing(self, db_session):
        await JobManager.create_job(
            db_session,
            job_id="test-job-003",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        await JobManager.mark_processing(db_session, "test-job-003")

        job = await JobManager.get_job(db_session, "test-job-003")
        assert job is not None
        assert job.status == "processing"

    async def test_mark_completed(self, db_session):
        await JobManager.create_job(
            db_session,
            job_id="test-job-004",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        # Must go through processing first (status guard)
        await JobManager.mark_processing(db_session, "test-job-004")
        await JobManager.mark_completed(
            db_session,
            "test-job-004",
            summary={"total": 5, "passed": 4, "failed": 1},
            test_results=[
                {"name": "test1", "status": "passed"},
                {"name": "test2", "status": "failed"},
            ],
            report_url="/reports/test-job-004",
            execution_time_seconds=12.5,
        )

        job = await JobManager.get_job(db_session, "test-job-004")
        assert job is not None
        assert job.status == "completed"
        assert job.summary is not None
        assert job.summary["total"] == 5
        assert job.test_results is not None
        assert len(job.test_results) == 2
        assert job.report_url == "/reports/test-job-004"
        assert job.execution_time_seconds == 12.5
        assert job.completed_at is not None

    async def test_mark_failed(self, db_session):
        await JobManager.create_job(
            db_session,
            job_id="test-job-005",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        # Must go through processing first (status guard)
        await JobManager.mark_processing(db_session, "test-job-005")
        await JobManager.mark_failed(
            db_session,
            "test-job-005",
            error_message="npm ci failed",
            execution_time_seconds=3.2,
        )

        job = await JobManager.get_job(db_session, "test-job-005")
        assert job is not None
        assert job.status == "failed"
        assert job.error_message == "npm ci failed"
        assert job.execution_time_seconds == 3.2
        assert job.completed_at is not None

    async def test_mark_processing_is_idempotent_fail_on_second_call(self, db_session):
        """Transitioning a processing/terminal job back to processing must raise."""
        await JobManager.create_job(
            db_session,
            job_id="test-job-006",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        await JobManager.mark_processing(db_session, "test-job-006")
        with pytest.raises(JobStateConflictError):
            await JobManager.mark_processing(db_session, "test-job-006")

    async def test_mark_processing_race_exactly_one_winner(self, db_engine):
        """Two concurrent sessions racing on mark_processing: one wins, one conflicts."""
        session_factory = async_sessionmaker(db_engine, expire_on_commit=False)
        async with session_factory() as seed_session:
            await JobManager.create_job(
                seed_session,
                job_id="test-job-race",
                tenant_id="01950000-0000-7000-8000-000000000001",
                target_url="https://example.com",
            )

        async def _worker() -> str:
            async with session_factory() as session:
                try:
                    await JobManager.mark_processing(session, "test-job-race")
                    return "won"
                except JobStateConflictError:
                    return "lost"

        results = await asyncio.gather(_worker(), _worker(), _worker())
        assert sorted(results) == ["lost", "lost", "won"]

        async with session_factory() as check:
            final = await JobManager.get_job(check, "test-job-race")
            assert final is not None
            assert final.status == "processing"


class TestHeartbeat:
    """touch_heartbeat — the running job's liveness lease against the reaper."""

    async def test_touch_bumps_updated_at_for_processing(self, db_session):
        from datetime import UTC, datetime, timedelta

        await JobManager.create_job(
            db_session,
            job_id="hb-1",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        await JobManager.mark_processing(db_session, "hb-1")
        # Age updated_at into the past, then heartbeat.
        job = await JobManager.get_job(db_session, "hb-1")
        assert job is not None
        job.updated_at = datetime.now(UTC) - timedelta(minutes=45)
        await db_session.commit()

        touched = await JobManager.touch_heartbeat(db_session, "hb-1")
        assert touched is True

        refreshed = await JobManager.get_job(db_session, "hb-1")
        assert refreshed is not None
        assert refreshed.updated_at is not None
        # Now fresh again (within the last minute).
        assert refreshed.updated_at > datetime.now(UTC) - timedelta(minutes=1)

    async def test_touch_noop_for_terminal_job(self, db_session):
        """A completed/failed job must not be resurrected by a stray heartbeat."""
        await JobManager.create_job(
            db_session,
            job_id="hb-2",
            tenant_id="01950000-0000-7000-8000-000000000001",
            target_url="https://example.com",
        )
        await JobManager.mark_processing(db_session, "hb-2")
        await JobManager.mark_completed(
            db_session,
            "hb-2",
            summary={"total": 1, "passed": 1, "failed": 0},
            test_results=[],
        )
        touched = await JobManager.touch_heartbeat(db_session, "hb-2")
        assert touched is False

        job = await JobManager.get_job(db_session, "hb-2")
        assert job is not None
        assert job.status == "completed"
