"""008_reap_never_started_fn

Add the ``reap_never_started_test_execution_jobs`` SECURITY DEFINER function.

⛔ **The strand this closes.** ``POST /api/test-runner/execute`` COMMITS the job
row as ``pending`` and only THEN hands ``_process_job`` to FastAPI's
``BackgroundTasks``, which runs after the response. Anything that kills the
process in that window — SIGKILL, grace-period expiry, a kubelet eviction, a
``CancelledError`` raised before ``mark_processing`` (``CancelledError`` is a
``BaseException``, so ``_process_job``'s ``except Exception`` never sees it and
its ``finally`` only unregisters from the drain), or the task being dropped
before ``drain.register`` — leaves the row at ``pending`` FOREVER:

* the graceful drain never saw the job, so ``_hand_back_abandoned_jobs`` cannot
  fail it;
* ``reap_stuck_test_execution_jobs`` matches ``status = 'processing'`` only;
* ``delete_old_test_execution_jobs`` matches ``status IN ('completed','failed')``
  only.

The caller polls a job that will never move, and the row is never even deleted by
retention. This migration adds the missing arm.

⛔ **Why a NEW function instead of widening the shipped one to**
``status IN ('processing','pending')``: the two arms need INDEPENDENT cutoffs. A
``processing`` job heartbeats ``updated_at``, so 30 minutes of silence proves the
producer is dead. A ``pending`` job does NOT heartbeat and is a legitimate state a
live job occupies between the commit and ``mark_processing`` — reusing the 30-min
cutoff would fail out queued work that is about to run, which is a regression, not
a fix. The pending arm therefore takes its own ``interval`` (default 6h) and
clocks on ``created_at``, not on the heartbeat column. Adding a second argument to
``reap_stuck_test_execution_jobs`` was rejected for a second reason: ``CREATE OR
REPLACE`` cannot change a function's arity, so it would need ``DROP FUNCTION`` +
re-``CREATE``, and the recreated function is born with an EMPTY ACL — the prod
reaper (running as ``w2v_ftr_app``) would hit "permission denied for function"
while CI, running as the table owner, stayed green.

SECURITY DEFINER for the same reason as its siblings: the sweep is CROSS-TENANT
and the prod runtime role under ``TENANT_RLS_MODE=enforce`` would otherwise be
scoped to the empty ``app.current_tenant`` GUC and reap 0 rows. EXECUTE is
revoked from PUBLIC and granted to ``w2v_ftr_app`` in a ``pg_roles``-guarded
``DO`` block (CI-safe: the role only exists in prod). Idempotent
(``CREATE OR REPLACE``) and fully reversible.

The DDL is single-sourced in ``src/services/cleanup_service.py`` so the
migration, the test bootstrap, and the calling code cannot drift.

Revision ID: a7b8c9d0e1f2
Revises: f6a7b8c9d0e1
Create Date: 2026-09-05 00:00:00.000000
"""

from collections.abc import Sequence

from alembic import op

from src.services.cleanup_service import (
    never_started_fn_downgrade_statements,
    never_started_fn_upgrade_statements,
)

revision: str = "a7b8c9d0e1f2"
down_revision: str | None = "f6a7b8c9d0e1"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    for stmt in never_started_fn_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in never_started_fn_downgrade_statements():
        op.execute(stmt)
