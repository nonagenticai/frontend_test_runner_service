"""SQLAlchemy ORM models for frontend_test_runner_service."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Index, String, Text, UniqueConstraint, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from w2v_common.database.rls import register_rls_ddl

# Use JSONB on PostgreSQL, plain JSON on SQLite/others
JSONType = JSON().with_variant(JSONB, "postgresql")

# ⛔ ``as_uuid=False`` is load-bearing, not a style choice. With ``as_uuid=True``
# SQLAlchemy hands back ``uuid.UUID`` objects, which silently breaks every
# ``tenant_id: str`` annotation, every ``==`` against a string tenant, every
# ``str``-keyed cache key and every JSON serialisation in this service — the shape
# that has frozen outboxes elsewhere in the fleet. ``as_uuid=False`` keeps the
# Python value a ``str`` while the COLUMN is a real Postgres ``uuid``, so the
# database gains the type check, the index gains 16-byte keys, and the RLS
# predicate can cast the GUC once (``…::uuid``) instead of casting the column on
# every row.


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""


class TestExecutionJob(Base):
    """Tracks a test execution job."""

    __test__ = False  # Not a pytest test class

    __tablename__ = "test_execution_jobs"
    __table_args__ = (
        # Partial unique index backing the Idempotency-Key dedup (also created by
        # migration 004 for existing DBs). Declared here so create_all builds it in
        # tests/CI. NULL keys are never deduped (legacy/unkeyed jobs stay distinct).
        Index(
            "uq_test_execution_jobs_tenant_idempotency_key",
            "tenant_id",
            "idempotency_key",
            unique=True,
            postgresql_where=text("idempotency_key IS NOT NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False), index=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")

    # Input
    target_url: Mapped[str] = mapped_column(Text)
    bdd_job_id: Mapped[str | None] = mapped_column(String(64), default=None)
    package_url: Mapped[str | None] = mapped_column(Text, default=None)

    # Caller-supplied dedup key (Idempotency-Key header). When set, a partial
    # unique index on (tenant_id, idempotency_key) makes a re-submit of the SAME
    # execution (e.g. the orchestrator's demo_video_record retrying after a
    # resumable timeout) return the ORIGINAL job instead of spawning a second
    # full bddgen+Playwright recording (duplicate paid compute). See migration 004.
    idempotency_key: Mapped[str | None] = mapped_column(String(128), default=None)

    # Configuration
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONType, default=dict)

    # Results
    summary: Mapped[dict[str, Any] | None] = mapped_column(JSONType, default=dict)
    test_results: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONType, default=list)
    scenario_clips: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONType, default=list)
    report_url: Mapped[str | None] = mapped_column(Text, default=None)
    error_message: Mapped[str | None] = mapped_column(Text, default=None)
    # Why a capture produced no directed-footage track (no defineBddConfig to
    # wire, an unchainable importTestFrom, missing recorder assets, or a run
    # that recorded zero interaction). A track-less capture renders as a static
    # film, so the reason must survive on the JOB — a pod log line does not.
    capture_warnings: Mapped[list[str] | None] = mapped_column(JSONType, default=list)

    # Artifact paths
    artifact_paths: Mapped[dict[str, Any] | None] = mapped_column(JSONType, default=dict)

    # Timing
    execution_time_seconds: Mapped[float | None] = mapped_column(default=None)
    created_at: Mapped[datetime | None] = mapped_column(default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime | None] = mapped_column(
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )
    completed_at: Mapped[datetime | None] = mapped_column(default=None)


class TestRunnerUsage(Base):
    """Daily usage tracking per tenant."""

    __test__ = False  # Not a pytest test class

    __tablename__ = "test_runner_usage"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    tenant_id: Mapped[str] = mapped_column(PG_UUID(as_uuid=False))
    date: Mapped[str] = mapped_column(String(10))
    jobs_started: Mapped[int | None] = mapped_column(default=0)
    jobs_completed: Mapped[int | None] = mapped_column(default=0)
    jobs_failed: Mapped[int | None] = mapped_column(default=0)
    total_tests_run: Mapped[int | None] = mapped_column(default=0)
    total_tests_passed: Mapped[int | None] = mapped_column(default=0)

    __table_args__ = (
        UniqueConstraint("tenant_id", "date", name="uq_test_runner_usage_tenant_date"),
    )


# Phase D RLS: enable + tenant_isolation policy on every tenant_id table when a
# fresh schema is built via create_all (tests/dev). Prod gets the SAME DDL from the
# alembic migration (single source: w2v_common.database.rls). ``tenant_id`` is
# NOT NULL on both tables → strict-equality policy. Inert until the runtime connects
# as the restricted ``w2v_ftr_app`` role with ``TENANT_RLS_MODE=enforce``.
register_rls_ddl(Base.metadata)
