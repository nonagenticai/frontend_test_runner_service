"""006_capture_warnings

Add the ``capture_warnings`` JSONB column to ``test_execution_jobs``.

A capture that ships with no directed-footage track (``focus_track`` /
``cursor_track``) renders as a STATIC film: the orchestrator's
``footage_direction`` forces ``zoom="off"`` and ``footage_plans`` pops both
tracks, so the only code path that adds camera movement to a screen recording
is dead by data starvation. The reasons that happens — no ``defineBddConfig()``
to wire, an ``importTestFrom`` we cannot chain onto, missing recorder assets in
the image, or a run whose scenarios recorded zero interaction — used to be a
``logger.warning`` inside a silent ``return``, i.e. invisible to every consumer
of the job. They now ride out on the job row so the defect is diagnosable from
the job output.

Nullable + defaulted: existing rows read back as ``[]`` through
``job.capture_warnings or []``, so the column is additive and backfill-free.

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-08-19 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e5f6a7b8c9d0"
down_revision: str | None = "d4e5f6a7b8c9"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "test_execution_jobs",
        sa.Column(
            "capture_warnings",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("test_execution_jobs", "capture_warnings")
