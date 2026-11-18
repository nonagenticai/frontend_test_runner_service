"""Tests for API schemas."""

from typing import Any

import pytest
from pydantic import ValidationError

from src.api.schemas import (
    AsyncJobResponse,
    JobStatusResponse,
    TestCaseResult,
    TestExecutionRequest,
    TestExecutionResult,
    TestSummary,
)


class TestTestExecutionRequest:
    def test_minimal_request_with_bdd_job_id(self):
        req = TestExecutionRequest(
            bdd_job_id="job-123",
            target_url="https://example.com",
        )
        assert req.bdd_job_id == "job-123"
        assert req.target_url == "https://example.com"
        assert req.browsers == ["chromium"]
        assert req.timeout_per_test_ms == 30_000
        assert req.retries == 0
        assert req.workers == 1
        assert req.screenshot_on_failure is True
        assert req.trace_on_failure is True

    def test_minimal_request_with_package_url(self):
        req = TestExecutionRequest(
            package_url="https://storage.example.com/tests.zip",
            target_url="https://example.com",
        )
        assert req.package_url == "https://storage.example.com/tests.zip"
        assert req.bdd_job_id is None

    def test_full_request(self):
        req = TestExecutionRequest(
            bdd_job_id="job-456",
            target_url="https://example.com",
            browsers=["chromium", "firefox"],
            timeout_per_test_ms=60_000,
            retries=2,
            workers=4,
            grep_pattern="@smoke",
            max_run_seconds=90,
            screenshot_on_failure=False,
            trace_on_failure=False,
        )
        assert req.browsers == ["chromium", "firefox"]
        assert req.timeout_per_test_ms == 60_000
        assert req.retries == 2
        assert req.workers == 4
        assert req.grep_pattern == "@smoke"
        assert req.max_run_seconds == 90

    def test_max_run_seconds_defaults_to_none(self) -> None:
        """Absent override means 'use the service ceiling' at run time."""
        req = TestExecutionRequest(bdd_job_id="j", target_url="https://example.com")
        assert req.max_run_seconds is None

    @pytest.mark.parametrize(
        "field,value",
        [
            ("timeout_per_test_ms", 0),
            ("timeout_per_test_ms", 999),
            ("timeout_per_test_ms", 600_001),
            ("timeout_per_test_ms", 2_000_000_000),
            ("retries", -1),
            ("retries", 11),
            ("retries", 1_000_000),
            ("workers", 0),
            ("workers", 17),
            ("workers", 10_000),
            ("max_run_seconds", 9),
            ("max_run_seconds", 1_801),
            ("max_run_seconds", 2_000_000_000),
        ],
    )
    def test_rejects_out_of_bounds_int_fields(self, field: str, value: int) -> None:
        """Bounds prevent DoS via unbounded ints flowing into subprocess timeouts."""
        kwargs: dict[str, Any] = {
            "bdd_job_id": "j1",
            "target_url": "https://example.com",
            field: value,
        }
        with pytest.raises(ValidationError) as exc_info:
            TestExecutionRequest(**kwargs)
        assert field in str(exc_info.value)

    @pytest.mark.parametrize(
        "field,value",
        [
            ("timeout_per_test_ms", 1_000),
            ("timeout_per_test_ms", 600_000),
            ("retries", 0),
            ("retries", 10),
            ("workers", 1),
            ("workers", 16),
        ],
    )
    def test_accepts_boundary_values(self, field: str, value: int) -> None:
        """Boundary values at the exact edges of the allowed range succeed."""
        kwargs: dict[str, Any] = {
            "bdd_job_id": "j1",
            "target_url": "https://example.com",
            field: value,
        }
        req = TestExecutionRequest(**kwargs)
        assert getattr(req, field) == value


class TestBrowsersBound:
    """`browsers` is bounded + de-duplicated (argv self-DoS guard).

    ``execution_engine`` expands each entry into a ``--project <name>`` argv
    pair; unbounded, ``["chromium"] * 100_000`` builds ~1.6 MB of argv and
    ``create_subprocess_exec`` dies with "Argument list too long".
    """

    def test_defaults_to_chromium(self) -> None:
        req = TestExecutionRequest(bdd_job_id="j1", target_url="https://example.com")
        assert req.browsers == ["chromium"]

    @pytest.mark.parametrize(
        "browsers",
        [
            ["chromium", "firefox", "webkit", "chromium"],
            ["chromium"] * 4,
            ["chromium"] * 100_000,
        ],
    )
    def test_rejects_more_than_three_entries(self, browsers: list[str]) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TestExecutionRequest(
                bdd_job_id="j1",
                target_url="https://example.com",
                browsers=browsers,
            )
        assert "browsers" in str(exc_info.value)

    def test_accepts_full_valid_set(self) -> None:
        req = TestExecutionRequest(
            bdd_job_id="j1",
            target_url="https://example.com",
            browsers=["chromium", "firefox", "webkit"],
        )
        assert req.browsers == ["chromium", "firefox", "webkit"]

    def test_dedupes_preserving_order(self) -> None:
        req = TestExecutionRequest(
            bdd_job_id="j1",
            target_url="https://example.com",
            browsers=["chromium", "chromium", "firefox"],
        )
        assert req.browsers == ["chromium", "firefox"]

    def test_unknown_browser_still_rejected_with_existing_message(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            TestExecutionRequest(
                bdd_job_id="j1",
                target_url="https://example.com",
                browsers=["chromium", "netscape"],
            )
        message = str(exc_info.value)
        assert "Unknown browsers: ['netscape']" in message
        assert "Valid: ['chromium', 'firefox', 'webkit']" in message


class TestTestSummary:
    def test_defaults(self):
        s = TestSummary()
        assert s.total == 0
        assert s.passed == 0
        assert s.failed == 0
        assert s.skipped == 0
        assert s.flaky == 0
        assert s.pass_rate == 0.0

    def test_with_values(self):
        s = TestSummary(total=10, passed=8, failed=1, skipped=1, pass_rate=80.0)
        assert s.total == 10
        assert s.passed == 8
        assert s.pass_rate == 80.0


class TestTestCaseResult:
    def test_defaults(self):
        tc = TestCaseResult()
        assert tc.name == ""
        assert tc.status == ""
        assert tc.duration_ms == 0.0

    def test_with_values(self):
        tc = TestCaseResult(
            name="Login test",
            feature="Authentication",
            status="passed",
            duration_ms=1234.5,
        )
        assert tc.name == "Login test"
        assert tc.feature == "Authentication"
        assert tc.status == "passed"


class TestAsyncJobResponse:
    def test_response(self):
        r = AsyncJobResponse(job_id="abc", poll_url="/api/test-runner/abc")
        assert r.job_id == "abc"
        assert r.status == "pending"
        assert r.poll_url == "/api/test-runner/abc"


class TestJobStatusResponse:
    def test_response(self):
        r = JobStatusResponse(job_id="abc", status="processing")
        assert r.job_id == "abc"
        assert r.status == "processing"
        assert r.error is None


class TestTestExecutionResult:
    def test_defaults(self):
        r = TestExecutionResult(job_id="abc", status="completed")
        assert r.summary.total == 0
        assert r.test_results == []
        assert r.execution_time_seconds == 0.0
