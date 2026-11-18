"""Phase D RLS — enable row-level security + the cross-tenant cleanup reaper functions.

Applies the SAME policies the ``create_all`` ``register_rls_ddl`` event applies in tests,
rendered from the single source ``w2v_common.database.rls`` (all tenant_id tables:
``test_execution_jobs`` + ``test_runner_usage``; both NOT NULL → strict-equality policy,
varchar → text predicate). ENABLE-not-FORCE → inert until the runtime connects as the
restricted non-owner role ``w2v_ftr_app`` with ``TENANT_RLS_MODE=enforce``. The
``run-migrations`` initContainer connects as the privileged owner, so owner DDL lands
normally.

Also creates the ``reap_stuck_test_execution_jobs`` / ``delete_old_test_execution_jobs``
SECURITY DEFINER functions (single-sourced in ``src/services/cleanup_service.py``): the
periodic cleanup pass is CROSS-TENANT, so it must run as the function owner to bypass RLS —
a plain UPDATE/DELETE as the restricted role would fail-close to 0 rows. EXECUTE is revoked
from PUBLIC and granted to ``w2v_ftr_app`` in a ``pg_roles``-guarded ``DO`` block (CI-safe:
the role only exists in prod). Fully reversible.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
"""

from alembic import op

from src.services.cleanup_service import (
    downgrade_statements as reaper_downgrade_statements,
)
from src.services.cleanup_service import (
    upgrade_statements as reaper_upgrade_statements,
)

revision: str = "b2c3d4e5f6a7"
down_revision: str | None = "a1b2c3d4e5f6"
branch_labels: str | None = None
depends_on: str | None = None


# ⛔ FROZEN 2026-09-02 — these statements used to be rendered at run time from the
# LIVE ``Base.metadata`` via ``w2v_common.database.rls.{up,down}grade_statements``.
# That makes a historical migration describe TODAY's ORM instead of the schema that
# existed when it ran: ``_predicate()`` renders ``tenant_id = <guc>`` for a
# ``String`` column and ``tenant_id = <guc>::uuid`` for anything else, so the moment
# the ORM column became ``uuid`` this migration started emitting a ``::uuid``
# predicate against a column that is still ``VARCHAR(128)`` at this point in the
# chain — ``operator does not exist: character varying = uuid`` — and
# ``alembic upgrade head`` from ZERO broke. The statements below are byte-for-byte
# what the helpers rendered at the time (varchar tenant_id, all NOT NULL → strict
# equality), pinned as literals. A migration is a historical record; it must not
# move when the models move. The type change itself lives in ``f6a7b8c9d0e1``.
_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
_PREDICATE = f"tenant_id = {_GUC}"  # varchar column → text-to-text comparison
_TENANT_TABLES: tuple[str, ...] = (
    "test_execution_jobs",
    "test_runner_usage",
)
_ALL_POLICY_NAMES = (
    "tenant_isolation",
    "tenant_isolation_read",
    "tenant_isolation_ins",
    "tenant_isolation_upd",
    "tenant_isolation_del",
)


def _rls_upgrade_statements() -> list[str]:
    out: list[str] = []
    for table in _TENANT_TABLES:
        out += [
            f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
            f"DROP POLICY IF EXISTS tenant_isolation ON {table}",
            f"CREATE POLICY tenant_isolation ON {table} "
            f"USING ({_PREDICATE}) WITH CHECK ({_PREDICATE})",
        ]
    return out


def _rls_downgrade_statements() -> list[str]:
    out: list[str] = []
    for table in _TENANT_TABLES:
        out += [f"DROP POLICY IF EXISTS {policy} ON {table}" for policy in _ALL_POLICY_NAMES]
        out.append(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
    return out


def upgrade() -> None:
    for stmt in _rls_upgrade_statements():
        op.execute(stmt)
    for stmt in reaper_upgrade_statements():
        op.execute(stmt)


def downgrade() -> None:
    for stmt in reaper_downgrade_statements():
        op.execute(stmt)
    for stmt in _rls_downgrade_statements():
        op.execute(stmt)
