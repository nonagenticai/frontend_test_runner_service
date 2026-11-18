"""Job lifecycle management for test execution jobs."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any, cast

from sqlalchemy import select, text, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import CursorResult
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import TestExecutionJob

logger = logging.getLogger(__name__)


class JobStateConflictError(RuntimeError):
    """Raised when an atomic job-state transition loses a race.

    A caller should treat this as "another worker already transitioned
    this job" — log and bail cleanly, do not retry.
    """


class JobManager:
    """Static CRUD helpers for TestExecutionJob records."""

    @staticmethod
    async def create_job(
        db: AsyncSession,
        *,
        job_id: str,
        tenant_id: str,
        target_url: str,
        bdd_job_id: str | None = None,
        package_url: str | None = None,
        config: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> tuple[TestExecutionJob, bool]:
        """Create a test execution job record.

        Returns ``(job, created)``. When ``idempotency_key`` is set, a re-submit
        with the SAME ``(tenant_id, idempotency_key)`` returns the EXISTING job
        with ``created=False`` (the partial unique index from migration 004
        dedupes) so the caller does NOT spawn a second recording run — the fix for
        demo_video_record re-submitting a paid FTR job after a resumable timeout.
        With no key, behaviour is unchanged (always a fresh job).
        """
        if idempotency_key is not None:
            stmt = (
                pg_insert(TestExecutionJob)
                .values(
                    job_id=job_id,
                    tenant_id=tenant_id,
                    status="pending",
                    target_url=target_url,
                    bdd_job_id=bdd_job_id,
                    package_url=package_url,
                    config=config or {},
                    idempotency_key=idempotency_key,
                )
                .on_conflict_do_nothing(
                    index_elements=["tenant_id", "idempotency_key"],
                    index_where=text("idempotency_key IS NOT NULL"),
                )
                .returning(TestExecutionJob.job_id)
            )
            inserted = (await db.execute(stmt)).first()
            await db.commit()
            if inserted is not None:
                created = await JobManager.get_job(db, job_id)
                return cast(TestExecutionJob, created), True
            # Conflict: an execution with this key already exists — return it.
            existing = (
                await db.execute(
                    select(TestExecutionJob).where(
                        TestExecutionJob.tenant_id == tenant_id,
                        TestExecutionJob.idempotency_key == idempotency_key,
                    )
                )
            ).scalar_one()
            return existing, False

        job = TestExecutionJob(
            job_id=job_id,
            tenant_id=tenant_id,
            status="pending",
            target_url=target_url,
            bdd_job_id=bdd_job_id,
            package_url=package_url,
            config=config or {},
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)
        return job, True

    @staticmethod
    async def get_job(db: AsyncSession, job_id: str) -> TestExecutionJob | None:
        """Retrieve a job by its job_id."""
        result = await db.execute(select(TestExecutionJob).where(TestExecutionJob.job_id == job_id))
        return result.scalar_one_or_none()

    @staticmethod
    async def mark_processing(db: AsyncSession, job_id: str) -> None:
        """Atomically transition a pending job to processing.

        Raises :class:`JobStateConflictError` if the job is not currently pending
        (another worker already claimed it, or it is already terminal).
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(TestExecutionJob)
                .where(
                    TestExecutionJob.job_id == job_id,
                    TestExecutionJob.status == "pending",
                )
                .values(status="processing", updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        if result.rowcount == 0:
            raise JobStateConflictError(
                f"job {job_id!r} is not in 'pending' state — cannot transition to 'processing'"
            )

    @staticmethod
    async def touch_heartbeat(db: AsyncSession, job_id: str) -> bool:
        """Bump ``updated_at`` for a running job (its liveness lease).

        Only touches a job still in ``processing`` (a completed/failed job must
        not be resurrected). Returns ``True`` if a row was updated. The reaper
        fails only ``processing`` jobs whose ``updated_at`` is older than the
        stale window, so a job that keeps heartbeating is never reaped.
        """
        result = cast(
            CursorResult[Any],
            await db.execute(
                update(TestExecutionJob)
                .where(
                    TestExecutionJob.job_id == job_id,
                    TestExecutionJob.status == "processing",
                )
                .values(updated_at=datetime.now(UTC))
            ),
        )
        await db.commit()
        return result.rowcount > 0

    @staticmethod
    async def mark_completed(
        db: AsyncSession,
        job_id: str,
        *,
        summary: dict[str, Any],
        test_results: list[dict[str, Any]],
        report_url: str = "",
        artifact_paths: dict[str, Any] | None = None,
        scenario_clips: list[dict[str, Any]] | None = None,
        capture_warnings: list[str] | None = None,
        execution_time_seconds: float = 0.0,
    ) -> None:
        """Mark a job as completed with results."""
        now = datetime.now(UTC)
        await db.execute(
            update(TestExecutionJob)
            .where(
                TestExecutionJob.job_id == job_id,
                TestExecutionJob.status == "processing",
            )
            .values(
                status="completed",
                summary=summary,
                test_results=test_results,
                scenario_clips=scenario_clips or [],
                capture_warnings=capture_warnings or [],
                report_url=report_url or None,
                artifact_paths=artifact_paths or {},
                execution_time_seconds=execution_time_seconds,
                updated_at=now,
                completed_at=now,
            )
        )
        await db.commit()

    @staticmethod
    async def mark_failed(
        db: AsyncSession,
        job_id: str,
        *,
        error_message: str,
        execution_time_seconds: float = 0.0,
    ) -> None:
        """Mark a job as failed with error details."""
        now = datetime.now(UTC)
        await db.execute(
            update(TestExecutionJob)
            .where(
                TestExecutionJob.job_id == job_id,
                TestExecutionJob.status.in_(["pending", "processing"]),
            )
            .values(
                status="failed",
                error_message=error_message,
                execution_time_seconds=execution_time_seconds,
                updated_at=now,
                completed_at=now,
            )
        )
        await db.commit()
