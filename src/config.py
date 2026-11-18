"""Configuration management for frontend_test_runner_service."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from w2v_common.config import (
    APISettings as BaseAPISettings,
)
from w2v_common.config import (
    AuthSettings,
    LoggingSettings,
    StorageSettings,
)
from w2v_common.config import (
    DatabaseSettings as BaseDatabaseSettings,
)
from w2v_common.config import (
    RedisSettings as BaseRedisSettings,
)

_DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5450/frontend_test_runner"  # pragma: allowlist secret


class DatabaseSettings(BaseDatabaseSettings):
    """Database settings with frontend_test_runner defaults."""

    url: str = Field(default=_DEFAULT_DATABASE_URL, description="PostgreSQL database URL")


class RedisSettings(BaseRedisSettings):
    """Redis settings with frontend_test_runner defaults."""

    url: str = Field(default="redis://localhost:6390/0", description="Redis connection URL")


class APISettings(BaseAPISettings):
    """API settings with frontend_test_runner default port."""

    port: int = Field(default=6316, description="API server port")


class ExecutionSettings(BaseSettings):
    """Test execution configuration."""

    default_timeout_per_test_ms: int = Field(
        default=30_000, description="Default per-test timeout in milliseconds"
    )
    max_workers: int = Field(default=4, description="Max parallel Playwright workers")
    node_modules_cache_dir: str = Field(
        default="/app/data/node_modules_cache",
        description="Cached node_modules for faster npm ci",
    )
    artifacts_root_dir: str = Field(
        default="/app/data/jobs",
        description="Directory where persisted reports and job artifacts are stored",
    )
    max_concurrent_executions: int = Field(
        default=5, description="Max concurrent test execution jobs"
    )
    convert_clips_to_mp4: bool = Field(
        default=True,
        description="Transcode recorded per-scenario WebM clips to H.264 MP4 on persist",
    )
    max_run_seconds: int = Field(
        default=1_500,
        description=(
            "Wall-clock ceiling for a single `npx playwright test` run. A backstop "
            "only — per-test time is bounded by --timeout, so this just caps a "
            "pathological suite. Kept under the orchestrator's 3x600s re-poll total "
            "(~1800s) so the record stage keeps polling the idempotent job."
        ),
    )
    heartbeat_interval_seconds: int = Field(
        default=60,
        description=(
            "How often a running job touches its updated_at so the lease-gated "
            "reaper never fails a live job. Must stay well under the cleanup "
            "stuck-job cutoff (default 30 min) — with a 60s heartbeat a live run "
            "of any length keeps updated_at fresh, so only a crashed pod (which "
            "stops heartbeating) ever goes stale enough to be reaped."
        ),
    )
    video_drain_timeout_seconds: float = Field(
        default=30.0,
        description=(
            "After results.json looks stable, how long to keep polling any "
            "in-flight .webm recordings for a stable size before killing the "
            "process group. results.json is written from the JSON reporter's "
            "end hook, which fires when the TESTS finish — but Playwright "
            "flushes each clip's video during context/browser TEARDOWN, which "
            "can still be running at that moment. Killing too early can catch "
            "the ffmpeg muxer mid-flush and leave the last clip short or "
            "header-less. 30s is generous for a muxer flush (sub-second in "
            "practice) while staying small next to max_run_seconds (1500s), so "
            "a stuck muxer can only add 30s, never re-introduce the "
            "minutes-long teardown hang the early-exit was built to avoid."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="EXECUTION_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class RateLimitSettings(BaseSettings):
    """Rate limiting configuration."""

    requests_per_minute: int = Field(default=20, description="Max requests per minute per tenant")
    concurrent_jobs_per_tenant: int = Field(default=3, description="Max concurrent jobs per tenant")

    model_config = SettingsConfigDict(
        env_prefix="RATE_LIMIT_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class CleanupSettings(BaseSettings):
    """Job cleanup configuration."""

    retention_days: int = Field(default=14, description="Days to retain completed/failed jobs")
    stuck_job_timeout_minutes: int = Field(
        default=30, description="Minutes after which a processing job is considered stuck"
    )
    # ⛔ Separate from — and far longer than — stuck_job_timeout_minutes ON PURPOSE.
    # A ``processing`` job heartbeats ``updated_at``, so 30 min of silence proves the
    # producer died. A ``pending`` job does NOT heartbeat and is legitimately queued
    # between the create_job commit and mark_processing; collapsing the two cutoffs
    # would fail out live work.
    never_started_job_timeout_hours: int = Field(
        default=6,
        description=(
            "Hours after creation before a job still 'pending' (accepted but never "
            "picked up by a runner process) is failed out"
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="CLEANUP_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class ShutdownSettings(BaseSettings):
    """Graceful-shutdown drain configuration."""

    drain_timeout_seconds: int = Field(
        default=1_800,
        description=(
            "How long shutdown waits for in-flight test runs before abandoning "
            "them. Must exceed ExecutionSettings.max_run_seconds (1500s) or a "
            "run that started just before SIGTERM is guaranteed to be dropped; "
            "the default is that ceiling plus the 300s slot buffer. Raising it "
            "is pointless unless the Deployment's terminationGracePeriodSeconds "
            "is raised to match — kubelet SIGKILLs at that grace period no "
            "matter what this says."
        ),
    )

    model_config = SettingsConfigDict(
        env_prefix="SHUTDOWN_", env_file=".env", case_sensitive=False, extra="ignore"
    )


class Settings(BaseSettings):
    """Main application settings."""

    app_name: str = Field(default="Frontend Test Runner Service")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default="development")

    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    api: APISettings = Field(default_factory=APISettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cleanup: CleanupSettings = Field(default_factory=CleanupSettings)
    shutdown: ShutdownSettings = Field(default_factory=ShutdownSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    def get_database_url(self) -> str:
        """Return the configured database connection URL."""
        return self.database.url

    def get_redis_url(self) -> str:
        """Return the configured Redis connection URL."""
        return self.redis.url

    def is_production(self) -> bool:
        """Return True if the environment is production."""
        return self.environment.lower() == "production"

    def is_development(self) -> bool:
        """Return True if the environment is development."""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def clear_settings_cache() -> None:
    """Clear the cached settings instance."""
    get_settings.cache_clear()
