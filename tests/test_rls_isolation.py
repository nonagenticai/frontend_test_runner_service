"""Proves the Phase D RLS policies ISOLATE tenants (not just that they exist).

Connects AS a throwaway non-owner ``NOBYPASSRLS`` role — the posture the runtime
``w2v_ftr_app`` role has under enforce — and asserts the shipped policies (applied by
``register_rls_ddl`` on ``create_all``) actually scope reads/writes on
``test_execution_jobs``. Needs a superuser test DB (``CREATE ROLE`` + ``SET ROLE``);
skips loudly when the DB role cannot create roles (e.g. a restricted CI role).
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.exc import ProgrammingError

from tests.conftest import TENANT_A_ID as TENANT_A
from tests.conftest import TENANT_B_ID as TENANT_B

_ROLE = "w2v_ftr_app_rls_test"
_INS = (
    "INSERT INTO test_execution_jobs (job_id, tenant_id, status, target_url) "
    "VALUES (:job_id, :tenant, 'pending', 'https://example.com')"
)


@pytest.mark.asyncio
async def test_rls_isolates_tenants_under_restricted_role(db_engine) -> None:
    async with db_engine.begin() as conn:
        # skip-gate FIRST (before the RLS-affected seed): a non-superuser CI role
        # can't CREATE ROLE and its seed would hit the policy with no GUC.
        try:
            await conn.execute(text(f'CREATE ROLE "{_ROLE}" NOSUPERUSER NOBYPASSRLS NOINHERIT'))
        except ProgrammingError as exc:
            pytest.skip(f"cannot CREATE ROLE (needs superuser/CREATEROLE): {exc.orig}")
        await conn.execute(
            text(f'GRANT SELECT, INSERT, UPDATE, DELETE ON test_execution_jobs TO "{_ROLE}"')
        )
        # The restricted role needs the id sequence to satisfy the autoincrement PK
        # default on its own (allowed) INSERTs.
        await conn.execute(text(f'GRANT USAGE ON SEQUENCE test_execution_jobs_id_seq TO "{_ROLE}"'))

        await conn.execute(text(_INS), {"job_id": "job-a", "tenant": TENANT_A})
        await conn.execute(text(_INS), {"job_id": "job-b", "tenant": TENANT_B})

        try:
            await conn.execute(text(f'SET LOCAL ROLE "{_ROLE}"'))

            n = (await conn.execute(text("SELECT count(*) FROM test_execution_jobs"))).scalar()
            assert n == 0, f"leaked {n} rows with no GUC"

            await conn.execute(
                text("SELECT set_config('app.current_tenant', :t, true)"), {"t": TENANT_A}
            )
            # ``tenant_id`` is a real ``uuid`` column now, and a RAW ``text()`` SELECT is
            # NOT the ORM path — psycopg adapts it to ``uuid.UUID`` regardless of the
            # model's ``as_uuid=False``. Compare as text.
            rows = (
                (await conn.execute(text("SELECT tenant_id FROM test_execution_jobs")))
                .scalars()
                .all()
            )
            assert [str(r) for r in rows] == [TENANT_A], f"not isolated under GUC=A: {rows}"

            await conn.execute(text("SAVEPOINT s"))
            with pytest.raises(ProgrammingError):
                await conn.execute(text(_INS), {"job_id": "job-x", "tenant": TENANT_B})
            await conn.execute(text("ROLLBACK TO SAVEPOINT s"))
        finally:
            await conn.execute(text("RESET ROLE"))
            await conn.execute(text(f'DROP OWNED BY "{_ROLE}"'))
            await conn.execute(text(f'DROP ROLE IF EXISTS "{_ROLE}"'))


@pytest.mark.asyncio
async def test_tenant_guc_rebinds_after_commit(db_session_factory, monkeypatch) -> None:
    """The after_begin listener keeps ``app.current_tenant`` set across COMMITs.

    Defense-in-depth regression: FTR's ``_process_job`` reuses NO single session
    across commits (each phase opens its own ``get_db_session()``), but this proves
    the shared ``install_rls_tenant_rebind`` listener re-binds the GUC on every
    transaction so a future refactor that DID reuse a session across commits could
    not silently reintroduce the fail-closed-write bug under enforce.
    """
    from w2v_common.auth import clear_identity_context, set_identity_context
    from w2v_common.database.rls import install_rls_tenant_rebind

    monkeypatch.setenv("TENANT_RLS_MODE", "enforce")
    install_rls_tenant_rebind()
    set_identity_context(TENANT_A, None, None)
    try:
        async with db_session_factory() as session:
            # First transaction: a statement that commits (ends the bound transaction).
            await session.execute(text("SELECT 1"))
            await session.commit()
            # Second, fresh transaction: the GUC must still be the tenant — proving
            # the re-bind fired on after_begin, not just on the initial bind.
            guc = (
                await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
            ).scalar()
            assert guc == TENANT_A, f"GUC not re-bound after commit: {guc!r}"
    finally:
        clear_identity_context()

    # Control: with no tenant in context the listener is a no-op (request/no-tenant paths).
    async with db_session_factory() as session:
        guc = (
            await session.execute(text("SELECT current_setting('app.current_tenant', true)"))
        ).scalar()
        assert not guc, f"GUC leaked with empty identity context: {guc!r}"
