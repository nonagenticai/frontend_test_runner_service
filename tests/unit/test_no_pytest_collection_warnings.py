"""Regression: importing every src.* submodule must not emit PytestCollectionWarning.

SQLAlchemy/pydantic models prefixed `Test*` (TestExecutionJob, TestSummary,
TestCaseResult, etc.) trigger PytestCollectionWarning at import time unless
explicitly opted out with `__test__ = False`. This test walks the full src
package, imports every submodule, and turns PytestCollectionWarning into an
error so any newly-added Test*-prefixed model that forgets the opt-out fails
fast in CI.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from types import ModuleType

import pytest

import src


def _all_submodules(pkg: ModuleType) -> list[str]:
    # Skip alembic migration scripts — they touch `alembic.context.config` at
    # import time, which only resolves while alembic is actually running, not
    # under standalone `importlib.import_module`. Migrations cannot contain
    # SQLAlchemy/pydantic models anyway, so they cannot be a source of
    # `PytestCollectionWarning` — the thing this test is regressing against.
    return [
        m.name
        for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")
        if ".migrations" not in m.name
    ]


def test_no_pytest_collection_warnings() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error", pytest.PytestCollectionWarning)
        for name in _all_submodules(src):
            importlib.import_module(name)
