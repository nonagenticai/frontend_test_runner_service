"""005_list_old_jobs_fn

Add the ``list_old_test_execution_jobs`` SECURITY DEFINER function.

The retention reap used to delete only the ROW. Nothing removed the job's
``artifacts_root_dir/<job_id>`` scratch dir or its ``{job_id}/`` object-store
prefix, so both grew without bound — and dropping the row ORPHANED the objects
permanently, because the id needed to find them went with it. The cleanup pass
now purges a job's artifacts immediately before the row disappears, which needs
the ids of the rows the delete is about to remove.

Reading them requires SECURITY DEFINER for exactly the reason the delete
function does: the sweep is CROSS-TENANT and the prod runtime role
(``w2v_ftr_app``, ``TENANT_RLS_MODE=enforce``) would otherwise be RLS-scoped to
the empty ``app.current_tenant`` GUC and read back 0 ids — silently purging
nothing while the rows vanished. EXECUTE is revoked from PUBLIC and granted to
``w2v_ftr_app`` in a ``pg_roles``-guarded ``DO`` block (CI-safe: the role only
exists in prod). Idempotent (``CREATE OR REPLACE``) and fully reversible.

The DDL is single-sourced in ``src/services/cleanup_service.py`` so the
migration, the test bootstrap, and the calling code cannot drift.

Revision ID: d4e5f6a7b8c9
Revises: c3d4e5f6a7b8
Create Date: 2026-08-18 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

from src.services.cleanup_service import (
    list_old_fn_downgrade_statements,
    list_old_fn_upgrade_statements,
)

revision: str = "d4e5f6a7b8c9"
down_revision: str | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for stmt in list_old_fn_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in list_old_fn_downgrade_statements():
        op.execute(stmt)
