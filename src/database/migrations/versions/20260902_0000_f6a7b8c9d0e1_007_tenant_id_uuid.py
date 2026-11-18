"""``tenant_id`` becomes a real Postgres ``uuid`` (RLS-safe, single transaction).

Fleet standard: ``tenant_id`` is a uuid everywhere. These tables stored it as
``VARCHAR(128)``.

⛔ **Why the policy dance.** ``ALTER TABLE t ALTER COLUMN tenant_id TYPE uuid``
fails outright while an RLS policy references the column::

    ERROR:  cannot alter type of a column used in a policy definition
    DETAIL:  policy tenant_isolation on table t depends on column "tenant_id"

so each table needs ``DROP POLICY`` -> ``ALTER TYPE`` -> ``CREATE POLICY`` and all
of it must be atomic. Alembic wraps ``upgrade()`` in one transaction, and this
migration adds nothing that would break out of it (no CONCURRENTLY, no explicit
COMMIT), so a failure anywhere rolls the policies back with the type.

⛔ **The trap that follows.** If the recreate step is ever skipped or renders
empty, the migration DROPs the policy, ALTERs, and *succeeds* — leaving RLS
ENABLED with NO policy, which denies every row to the restricted app role while
the owner (and every migration-time check) still sees everything. Nothing in the
happy path would say so. :func:`_assert_policies` therefore re-reads
``pg_policies`` after the DDL and raises unless every table carries a
``tenant_isolation`` policy whose ``qual`` contains the expected cast.

The recreated predicate casts the GUC — ``NULLIF(current_setting(...), '')::uuid``
— because a policy recreated with the old uncast predicate fails at query time
with ``operator does not exist: uuid = text``.

⛔ Table names are LITERAL here on purpose. Rendering them from a live
``Base.metadata`` makes the migration describe today's ORM instead of the schema
that existed when it ran; that is exactly the defect that made the historical
``20260710_0000_b2c3d4e5f6a7_003_rls`` RLS migration unrunnable from zero once the ORM column type
changed (frozen to literal SQL in the same PR).

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "f6a7b8c9d0e1"
down_revision: str | None = "e5f6a7b8c9d0"
branch_labels: str | None = None
depends_on: str | None = None

#: ⛔ LITERAL — never derived from ``Base.metadata``. See the module docstring.
TENANT_TABLES: tuple[str, ...] = (
    "test_execution_jobs",
    "test_runner_usage",
)

#: Legacy non-uuid labels mapped to their registered tenant uuid before the census
#: runs. Evidence for every entry lives in the PR body; an unidentifiable label is
#: NOT guessed — the census below raises and the migration refuses to run.
TENANT_LABEL_REMAP: dict[str, str] = {}

POLICY_NAME = "tenant_isolation"

#: Postgres POSIX class, not Python's ``re``. Anchored, case-insensitive.
_UUID_RE = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"

_GUC = "NULLIF(current_setting('app.current_tenant', true), '')"
#: Pre-change predicate — varchar column, text-to-text comparison.
_TEXT_PREDICATE = f"tenant_id = {_GUC}"
#: Post-change predicate — uuid column, GUC cast to uuid.
_UUID_PREDICATE = f"tenant_id = {_GUC}::uuid"


# S608: ``table`` is one of the module-level literal TENANT_TABLES constants above,
# never user input. Values ARE bound; only the identifier is interpolated, which no
# DBAPI can parameterise.
def _non_uuid_sql(table: str) -> str:
    return f"SELECT DISTINCT tenant_id::text FROM {table} WHERE tenant_id::text !~ '{_UUID_RE}'"  # noqa: S608


def _case_collision_sql(table: str) -> str:
    # ⛔ NOT ``count(*) <> count(DISTINCT lower(...))`` — that counts ROWS, so any
    # healthy table with more than one row per tenant trips it. The question is
    # "does one case-folded key cover more than one distinct spelling", which is a
    # count of DISTINCT spellings per folded key.
    return (
        f"SELECT lower(tenant_id::text) AS folded, count(DISTINCT tenant_id::text) AS spellings "  # noqa: S608
        f"FROM {table} GROUP BY 1 HAVING count(DISTINCT tenant_id::text) > 1"
    )


def _remap_labels(conn: sa.Connection) -> None:
    """Rewrite known legacy labels to their registered uuid, before the census."""
    for label, tenant_uuid in TENANT_LABEL_REMAP.items():
        for table in TENANT_TABLES:
            conn.execute(
                sa.text(
                    f"UPDATE {table} SET tenant_id = :new WHERE tenant_id::text = :old"  # noqa: S608
                ),
                {"new": tenant_uuid, "old": label},
            )


def _run_census(conn: sa.Connection, table: str) -> None:
    bad = [r[0] for r in conn.execute(sa.text(_non_uuid_sql(table)))]
    if bad:
        raise RuntimeError(
            f"{table}.tenant_id holds {len(bad)} value(s) that are not uuids: {sorted(bad)!r}. "
            f"Refusing to ALTER — identify each one and add it to TENANT_LABEL_REMAP with "
            f"evidence, or clean the rows. A guessed tenant id is worse than a visible "
            f"placeholder."
        )
    collisions = conn.execute(sa.text(_case_collision_sql(table))).fetchall()
    if collisions:
        raise RuntimeError(
            f"{table}.tenant_id has case-only collisions that ``::uuid`` would silently "
            f"merge (uuid is case-insensitive, varchar is not): {collisions!r}. Refusing "
            f"to ALTER — a UNIQUE index over tenant_id would collapse those rows."
        )


def _run_controls(conn: sa.Connection) -> None:
    """Prove the census is neither vacuous nor over-eager, in-band, every run.

    * NEGATIVE control — a *healthy* table with many rows per tenant (4 rows for
      one tenant, 1 for another) must pass BOTH checks. This is the case the
      naive ``count(*) <> count(DISTINCT lower(...))`` formulation refuses.
    * POSITIVE control — two spellings of one uuid differing only in case must be
      caught. Without this a census that always returns nothing looks identical
      to a clean table.
    """
    conn.execute(
        sa.text(
            "CREATE TEMP TABLE _tenant_census_ok (tenant_id varchar(128) NOT NULL) ON COMMIT DROP"
        )
    )
    conn.execute(
        sa.text(
            "INSERT INTO _tenant_census_ok (tenant_id) VALUES "
            "('0195f48b-d000-7000-8000-0000000000aa'), "
            "('0195f48b-d000-7000-8000-0000000000aa'), "
            "('0195f48b-d000-7000-8000-0000000000aa'), "
            "('0195f48b-d000-7000-8000-0000000000aa'), "
            "('0195f48b-d000-7000-8000-0000000000bb')"
        )
    )
    healthy_bad = conn.execute(sa.text(_non_uuid_sql("_tenant_census_ok"))).fetchall()
    healthy_collisions = conn.execute(sa.text(_case_collision_sql("_tenant_census_ok"))).fetchall()
    if healthy_bad or healthy_collisions:
        raise RuntimeError(
            "CONTROL FAILED: the census refused a healthy many-rows-per-tenant table "
            f"(non-uuid={healthy_bad!r}, collisions={healthy_collisions!r}). The check is "
            "wrong, not the data."
        )

    conn.execute(
        sa.text(
            "CREATE TEMP TABLE _tenant_census_bad (tenant_id varchar(128) NOT NULL) ON COMMIT DROP"
        )
    )
    conn.execute(
        sa.text(
            "INSERT INTO _tenant_census_bad (tenant_id) VALUES "
            "('0195f48b-d000-7000-8000-0000000000aa'), "
            "('0195F48B-D000-7000-8000-0000000000AA')"
        )
    )
    caught = conn.execute(sa.text(_case_collision_sql("_tenant_census_bad"))).fetchall()
    if len(caught) != 1 or caught[0][1] != 2:
        raise RuntimeError(
            "CONTROL FAILED: the case-collision census did not catch a planted "
            f"case-only duplicate (got {caught!r}). A census that cannot fail proves nothing."
        )
    conn.execute(sa.text("DROP TABLE _tenant_census_ok"))
    conn.execute(sa.text("DROP TABLE _tenant_census_bad"))


def _assert_policies(conn: sa.Connection, *, expect_cast: bool) -> None:
    """Fail loudly if any table lost its policy or kept the wrong predicate.

    RLS ENABLED with zero policies denies every row to the non-owner app role
    while the owner still sees everything — invisible to any check that runs as
    the migration user. This is the only guard against that outcome.
    """
    for table in TENANT_TABLES:
        row = conn.execute(
            sa.text(
                "SELECT qual FROM pg_policies "
                "WHERE schemaname = 'public' AND tablename = :t AND policyname = :p"
            ),
            {"t": table, "p": POLICY_NAME},
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"RLS policy {POLICY_NAME} is MISSING on {table} after the type change. "
                f"The table still has RLS ENABLED, so the restricted app role now reads "
                f"ZERO rows while the owner sees everything. Refusing to commit."
            )
        qual = row[0] or ""
        has_cast = "::uuid" in qual
        if has_cast is not expect_cast:
            raise RuntimeError(
                f"RLS policy {POLICY_NAME} on {table} has the wrong predicate: {qual!r} "
                f"(expected a ``::uuid`` cast: {expect_cast}). An uncast predicate against a "
                f"uuid column raises ``operator does not exist: uuid = text`` at query time."
            )


def upgrade() -> None:
    conn = op.get_bind()
    _run_controls(conn)
    _remap_labels(conn)
    for table in TENANT_TABLES:
        _run_census(conn, table)
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}")
        op.execute(f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE uuid USING tenant_id::uuid")
        op.execute(
            f"CREATE POLICY {POLICY_NAME} ON {table} "
            f"USING ({_UUID_PREDICATE}) WITH CHECK ({_UUID_PREDICATE})"
        )
    _assert_policies(conn, expect_cast=True)


def downgrade() -> None:
    """Restore VARCHAR(128) **and** the original uncast text predicate.

    NOTE: the :data:`TENANT_LABEL_REMAP` rewrites are deliberately NOT reversed —
    re-introducing a non-uuid tenant label would only re-create the defect this
    migration exists to remove, and the remapped rows are valid under both column
    types. The type and the policy, which are what an operator rolls back for,
    round-trip exactly.
    """
    conn = op.get_bind()
    for table in TENANT_TABLES:
        op.execute(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}")
        op.execute(
            f"ALTER TABLE {table} ALTER COLUMN tenant_id TYPE VARCHAR(128) USING tenant_id::text"
        )
        op.execute(
            f"CREATE POLICY {POLICY_NAME} ON {table} "
            f"USING ({_TEXT_PREDICATE}) WITH CHECK ({_TEXT_PREDICATE})"
        )
    _assert_policies(conn, expect_cast=False)
