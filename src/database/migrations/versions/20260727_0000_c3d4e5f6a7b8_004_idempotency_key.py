"""004_idempotency_key

Add the ``idempotency_key`` column to ``test_execution_jobs`` plus a PARTIAL
unique index on ``(tenant_id, idempotency_key) WHERE idempotency_key IS NOT NULL``.
A caller (the orchestrator's ``demo_video_record`` stage) sends a stable
``Idempotency-Key`` header so that re-submitting the SAME execution after a
resumable timeout returns the original job instead of spawning a second full
bddgen + Playwright recording run (duplicate paid compute). Mirrors
``video_compiler_service``'s idempotency index. Partial so legacy/unkeyed jobs
(NULL key) are never deduped.

Revision ID: c3d4e5f6a7b8
Revises: b2c3d4e5f6a7
Create Date: 2026-07-27 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | None = "b2c3d4e5f6a7"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_INDEX_NAME = "uq_test_execution_jobs_tenant_idempotency_key"


def upgrade() -> None:
    op.add_column(
        "test_execution_jobs",
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
    )
    op.create_index(
        _INDEX_NAME,
        "test_execution_jobs",
        ["tenant_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("idempotency_key IS NOT NULL"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX_NAME, table_name="test_execution_jobs")
    op.drop_column("test_execution_jobs", "idempotency_key")
