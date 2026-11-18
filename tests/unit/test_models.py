"""Tests for database models."""

from sqlalchemy import select

from src.database.models import TestExecutionJob, TestRunnerUsage


class TestTestExecutionJobModel:
    async def test_create_job(self, db_session):
        job = TestExecutionJob(
            job_id="model-test-001",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="pending",
            target_url="https://example.com",
        )
        db_session.add(job)
        await db_session.commit()

        result = await db_session.execute(
            select(TestExecutionJob).where(TestExecutionJob.job_id == "model-test-001")
        )
        loaded = result.scalar_one()
        assert loaded.job_id == "model-test-001"
        assert loaded.tenant_id == "01950000-0000-7000-8000-000000000001"
        assert loaded.status == "pending"
        assert loaded.target_url == "https://example.com"
        assert loaded.created_at is not None

    async def test_job_with_results(self, db_session):
        job = TestExecutionJob(
            job_id="model-test-002",
            tenant_id="01950000-0000-7000-8000-000000000001",
            status="completed",
            target_url="https://example.com",
            summary={"total": 3, "passed": 2, "failed": 1},
            test_results=[
                {"name": "test1", "status": "passed"},
                {"name": "test2", "status": "passed"},
                {"name": "test3", "status": "failed"},
            ],
            execution_time_seconds=15.7,
        )
        db_session.add(job)
        await db_session.commit()

        result = await db_session.execute(
            select(TestExecutionJob).where(TestExecutionJob.job_id == "model-test-002")
        )
        loaded = result.scalar_one()
        assert loaded.summary["total"] == 3
        assert len(loaded.test_results) == 3
        assert loaded.execution_time_seconds == 15.7


class TestTestRunnerUsageModel:
    async def test_create_usage(self, db_session):
        usage = TestRunnerUsage(
            tenant_id="01950000-0000-7000-8000-000000000001",
            date="2026-03-30",
            jobs_started=5,
            jobs_completed=4,
            jobs_failed=1,
            total_tests_run=50,
            total_tests_passed=45,
        )
        db_session.add(usage)
        await db_session.commit()

        result = await db_session.execute(
            select(TestRunnerUsage).where(
                TestRunnerUsage.tenant_id == "01950000-0000-7000-8000-000000000001"
            )
        )
        loaded = result.scalar_one()
        assert loaded.date == "2026-03-30"
        assert loaded.total_tests_run == 50
        assert loaded.total_tests_passed == 45
