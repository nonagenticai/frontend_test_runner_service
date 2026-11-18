"""Tests for configuration."""

import pytest

from src.config import Settings, clear_settings_cache, get_settings

# Every ambient env var that would shadow a declared default. `Settings` reads
# the real process environment, so a test asserting a DEFAULT has to clear these
# first or it is really asserting whatever the runner happens to export.
_DEFAULT_SHADOWING_ENV = ("ENVIRONMENT",)


class TestConfig:
    def setup_method(self):
        clear_settings_cache()

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch):
        # CI exports ENVIRONMENT=test at the job level (see the archetype's env
        # block), so without this the assertion below read 'test' and the whole
        # `Lint, types, tests` job went red on main — which silently SKIPS
        # build/promote/deploy. The default is what this test is about, so the
        # ambient value must be removed rather than accommodated.
        for name in _DEFAULT_SHADOWING_ENV:
            monkeypatch.delenv(name, raising=False)
        clear_settings_cache()

        s = Settings()
        assert s.app_name == "Frontend Test Runner Service"
        assert s.api.port == 6316
        assert s.environment == "development"

    def test_environment_is_read_from_the_ambient_env(self, monkeypatch: pytest.MonkeyPatch):
        """The flip side of the above: the default must still be overridable."""
        monkeypatch.setenv("ENVIRONMENT", "production")
        clear_settings_cache()

        assert Settings().environment == "production"

    def test_execution_defaults(self):
        s = Settings()
        assert s.execution.default_timeout_per_test_ms == 30_000
        assert s.execution.max_workers == 4
        assert s.execution.artifacts_root_dir == "/app/data/jobs"
        assert s.execution.max_concurrent_executions == 5

    def test_cleanup_defaults(self):
        s = Settings()
        assert s.cleanup.retention_days == 14
        assert s.cleanup.stuck_job_timeout_minutes == 30

    def test_rate_limit_defaults(self):
        s = Settings()
        assert s.rate_limit.requests_per_minute == 20
        assert s.rate_limit.concurrent_jobs_per_tenant == 3

    def test_no_downstream_settings_attribute(self):
        """Peer URLs come from w2v_common.discovery, not a Settings.downstream block."""
        s = Settings()
        assert not hasattr(s, "downstream"), (
            "Settings must NOT carry a downstream sub-config; use w2v_common.discovery.service_url."
        )

    def test_downstream_settings_class_removed(self):
        import src.config as cfg

        assert not hasattr(cfg, "DownstreamSettings"), (
            "DownstreamSettings was removed; use w2v_common.discovery.service_url."
        )

    def test_bdd_playwright_url_from_discovery(self, monkeypatch):
        monkeypatch.setenv("W2V_BDD_PLAYWRIGHT_URL", "http://bdd.test:9000")
        from w2v_common.discovery import service_url

        assert service_url("bdd_playwright") == "http://bdd.test:9000"

    def test_bdd_playwright_default_uses_k8s_dns(self, monkeypatch):
        monkeypatch.delenv("W2V_LOCAL", raising=False)
        monkeypatch.delenv("W2V_BDD_PLAYWRIGHT_URL", raising=False)
        from w2v_common.discovery import service_url

        assert service_url("bdd_playwright") == "http://bdd-playwright-service.w2v:6313"

    def test_get_settings_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_clear_cache(self):
        s1 = get_settings()
        clear_settings_cache()
        s2 = get_settings()
        assert s1 is not s2

    def test_is_development(self):
        s = Settings(environment="development")
        assert s.is_development() is True
        assert s.is_production() is False

    def test_is_production(self):
        s = Settings(environment="production")
        assert s.is_production() is True
        assert s.is_development() is False
