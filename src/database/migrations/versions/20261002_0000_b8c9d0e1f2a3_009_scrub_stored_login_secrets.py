"""009_scrub_stored_login_secrets

Remove login secrets already persisted in ``test_execution_jobs.config``.

``POST /execute`` stored ``request.model_dump()`` — ``auth_config.password``
included — on the job row, and ``GET /{job_id}/result`` returned that ``config``
verbatim. Measured on the primary 2026-10-02 (db ``frontend_test_runner_service``):
2 rows, both from the merqi showcase spike (jobs ``01a0fc54-4ebf`` and
``01a0fc55-1ed1``). From this release the route stores the config WITHOUT secrets
(``without_auth_secrets``); this is the one-shot scrub of what was written before.

Every secret field of every login (dict or list form) is removed; nothing else in
the row changes. Re-runnable: a scrubbed row no longer matches. RLS is ENABLEd, not
FORCEd, and alembic runs as the table owner, so every tenant's rows are visible.
The field set is a FROZEN copy of ``src.api.schemas.AUTH_SECRET_FIELDS``.

Revision ID: b8c9d0e1f2a3
Revises: a7b8c9d0e1f2
Create Date: 2026-10-02
"""

from __future__ import annotations

import json
import logging
from typing import Any

from alembic import op
from sqlalchemy import Connection, text

revision: str = "b8c9d0e1f2a3"  # pragma: allowlist secret
down_revision: str | None = "a7b8c9d0e1f2"  # pragma: allowlist secret
branch_labels: str | None = None
depends_on: str | None = None

log = logging.getLogger("alembic.runtime.migration")

_SECRET_FIELDS = frozenset({"password", "token", "client_secret", "api_key", "secret"})


def _scrub(config: dict[str, Any]) -> bool:
    auth = config.get("auth_config")
    entries = auth if isinstance(auth, list) else [auth]
    changed = False
    for entry in entries:
        if isinstance(entry, dict):
            for field in _SECRET_FIELDS & entry.keys():
                del entry[field]
                changed = True
    return changed


def scrub_rows(bind: Connection) -> int:
    """Remove stored login secrets; return the number of rows changed."""
    rows = bind.execute(
        text("SELECT id, config FROM test_execution_jobs WHERE config::text LIKE '%auth_config%'")
    ).all()
    changed = 0
    for row_id, config in rows:
        if isinstance(config, dict) and _scrub(config):
            bind.execute(
                text("UPDATE test_execution_jobs SET config = CAST(:c AS jsonb) WHERE id = :id"),
                {"c": json.dumps(config), "id": row_id},
            )
            changed += 1
    return changed


def upgrade() -> None:
    log.info(
        "009: scrubbed login secrets from %d test_execution_jobs row(s)", scrub_rows(op.get_bind())
    )


def downgrade() -> None:
    # Forward-only: a removed secret is not recoverable, by design.
    pass
