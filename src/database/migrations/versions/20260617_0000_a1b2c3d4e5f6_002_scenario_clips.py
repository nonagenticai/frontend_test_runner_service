"""002_scenario_clips

Add the ``scenario_clips`` JSONB column to ``test_execution_jobs`` — per-scenario
recorded video clips harvested from a Playwright run, keyed by their ``clip-<n>``
join id and consumed by the orchestrator demo_video pipeline.

Revision ID: a1b2c3d4e5f6
Revises: 39dfe255dd9f
Create Date: 2026-06-17 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "a1b2c3d4e5f6"
down_revision: str | None = "39dfe255dd9f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "test_execution_jobs",
        sa.Column(
            "scenario_clips",
            sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql"),
            nullable=True,
        ),
    )


def downgrade() -> None:
    op.drop_column("test_execution_jobs", "scenario_clips")
