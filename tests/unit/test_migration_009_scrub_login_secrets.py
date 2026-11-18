"""Migration 009 removes login secrets already persisted on job rows (2 in prod)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sqlalchemy import select

from src.database.models import TestExecutionJob
from src.services.job_manager import JobManager

_PATH = next(
    (Path(__file__).resolve().parents[2] / "src/database/migrations/versions").glob(
        "*_009_scrub_stored_login_secrets.py"
    )
)
_SECRET = "pw-stored-before-the-fix"  # pragma: allowlist secret


def _migration():
    spec = importlib.util.spec_from_file_location("m_009", _PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def test_scrub_rows_removes_stored_secrets_and_is_rerunnable(db_session) -> None:
    tenant = "01950000-0000-7000-8000-000000000001"
    configs = {
        "dict": {"auth_config": {"login_url": "https://a.example", "password": _SECRET}},
        "list": {
            "auth_config": [
                {"login_url": "https://a.example", "password": _SECRET},
                {"login_url": "https://b.example", "token": _SECRET},
            ]
        },
        "clean": {"browsers": ["chromium"]},
    }
    for name, config in configs.items():
        await JobManager.create_job(
            db_session,
            job_id=f"job-{name}",
            tenant_id=tenant,
            target_url="https://a.example",
            bdd_job_id="b",
            config=config,
        )
    await db_session.commit()

    migration = _migration()
    conn = await db_session.connection()
    first = await conn.run_sync(migration.scrub_rows)
    second = await conn.run_sync(migration.scrub_rows)
    await db_session.commit()

    assert (first, second) == (2, 0)
    db_session.expire_all()
    rows = {
        j.job_id: j.config for j in (await db_session.execute(select(TestExecutionJob))).scalars()
    }
    assert _SECRET not in str(rows)
    assert rows["job-list"]["auth_config"][1] == {"login_url": "https://b.example"}
    assert rows["job-clean"] == {"browsers": ["chromium"]}
