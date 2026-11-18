"""Pydantic request/response schemas for the test runner API."""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# ── Login helpers ───────────────────────────────────────────

#: One login of ``auth_config`` (type, login_url, username, password, sso_button, …).
AuthLogin = dict[str, Any]
#: Fields of a login that are secrets: never persisted on the job row, never returned.
AUTH_SECRET_FIELDS = frozenset({"password", "token", "client_secret", "api_key", "secret"})
MAX_AUTH_LOGINS = 10
#: A login's label (``requester``, ``gm``): a slug, mirrors the orchestrator's auth_secrets.
_LABEL_RE = re.compile(r"[a-z0-9](?:[a-z0-9_-]{0,38}[a-z0-9])?")


def login_label(entry: AuthLogin) -> str | None:
    """The login's label, ``None`` when it has none."""
    raw = entry.get("label")
    return raw if isinstance(raw, str) and raw else None


def _validate_login_keys(keys: list[tuple[str, AuthLogin]]) -> None:
    """Each login of a list is keyed ``(origin, label)``; no key twice, and an origin
    holding several logins needs every one of them labelled."""
    seen: set[tuple[str, str | None]] = set()
    labels_by_origin: dict[str, list[str | None]] = {}
    for origin, entry in keys:
        label = login_label(entry)
        if (origin, label) in seen:
            if label is None:
                raise ValueError(
                    f"auth_config lists origin {origin} twice — give each login a label"
                )
            raise ValueError(f"auth_config lists origin {origin} with label {label!r} twice")
        seen.add((origin, label))
        labels_by_origin.setdefault(origin, []).append(label)
    for origin, labels in labels_by_origin.items():
        if len(labels) > 1 and None in labels:
            raise ValueError(
                f"auth_config lists origin {origin} more than once: every login needs a label"
            )


def logins_file_key(origin: str, label: str | None, origin_count: int) -> str:
    """Key of a login in ``.auth/logins.json``: the bare origin when it is the origin's
    only login (unchanged), else ``<origin>#<label>`` (every login on a shared origin
    is labelled, so the key is unique)."""
    return f"{origin}#{label}" if origin_count > 1 and label else origin


def auth_logins(auth_config: object) -> list[AuthLogin]:
    """The login dicts of ``auth_config``: a dict is one login, a list is several."""
    if isinstance(auth_config, dict):
        return [auth_config]
    if isinstance(auth_config, list):
        return [e for e in auth_config if isinstance(e, dict)]
    return []


def reject_url_parser_differentials(url: str, field: str) -> None:
    """Refuse a URL on which Python's ``urlparse`` and a browser's WHATWG parser disagree.

    ``https://attacker.example\\@twenty.example/`` is host ``twenty.example`` to
    ``urlparse`` (so the SSRF check and the origin match approve it) and host
    ``attacker.example`` to Chromium (WHATWG reads ``\\`` as ``/``). Refused, never
    interpreted: any backslash, whitespace or control character anywhere, and ``@``
    (userinfo) or ``%`` (``%5C`` / ``%40``) in the authority.
    """
    if any(c == "\\" or c.isspace() or ord(c) < 0x20 or ord(c) == 0x7F for c in url):
        raise ValueError(f"{field} must not contain backslashes, whitespace or control characters")
    rest = url.split("://", 1)[1] if "://" in url else url
    for stop in "/?#":
        rest = rest.split(stop, 1)[0]
    if "@" in rest or "%" in rest:
        raise ValueError(f"{field} must not carry userinfo or percent-encoding in its host")


def strict_login_url(raw: object, field: str) -> str:
    """A URL a credential may be typed into: no parser differential, https only."""
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be a non-empty https URL")
    reject_url_parser_differentials(raw, field)
    if raw[:8].lower() != "https://":
        raise ValueError(f"{field} must use https")
    parsed = urlparse(raw)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port") from exc
    if not parsed.hostname or port == 0:
        raise ValueError(f"{field} must name a host")
    return raw


def canonical_origin(raw: object) -> str | None:
    """THE origin function of a strict login URL, else None.

    Serialised exactly like the browser's ``new URL(u).origin`` (default port 443
    omitted), so a package keying ``AUTH_LOGINS_FILE`` by ``page.url`` origin matches."""
    try:
        url = strict_login_url(raw, "url")
    except ValueError:
        return None
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    return f"https://{host}" if parsed.port in (None, 443) else f"https://{host}:{parsed.port}"


def login_origin(entry: AuthLogin) -> str | None:
    """Origin of the page a login is typed into: ``login_url``, else ``origin``."""
    for key in ("login_url", "origin"):
        raw = entry.get(key)
        if isinstance(raw, str) and raw:
            return canonical_origin(raw)
    return None


def login_secrets(auth_config: object) -> list[str]:
    """Every non-empty secret value of every login — the strings no artifact may carry."""
    return [
        str(entry[f])
        for entry in auth_logins(auth_config)
        for f in AUTH_SECRET_FIELDS
        if isinstance(entry.get(f), str) and entry[f]
    ]


def without_auth_secrets(config: dict[str, object]) -> dict[str, object]:
    """A copy of a request dump with every login secret removed — what the job row stores.

    The live request (secrets included) only ever travels in memory to the background
    job; the persisted ``config`` is returned by ``GET /{job_id}/result``.
    """
    out = dict(config)
    auth = out.get("auth_config")
    if isinstance(auth, (dict, list)):
        scrubbed = [
            {k: v for k, v in entry.items() if k not in AUTH_SECRET_FIELDS}
            for entry in auth_logins(auth)
        ]
        out["auth_config"] = scrubbed[0] if isinstance(auth, dict) else scrubbed
    return out


# ── Request schemas ─────────────────────────────────────────


class TestExecutionRequest(BaseModel):
    """Request to execute a test suite against a target URL."""

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(extra="forbid")

    # Source (at least one required)
    package_url: str | None = Field(default=None, description="URL to download test ZIP package")
    bdd_job_id: str | None = Field(
        default=None, description="BDD Playwright Service job ID to fetch package from"
    )

    # Target
    target_url: str = Field(description="Live site URL to test against")

    # Authentication for login-gated sites (injected as env vars for auth.setup.ts)
    auth_config: AuthLogin | list[AuthLogin] | None = Field(
        default=None,
        description=(
            "Auth credentials + SSO hints forwarded to auth.setup.ts: "
            "{type, login_url, username, password, sso_button, success_url}. "
            "sso_button is the IdP button label to click when the login form "
            "is not inline (e.g. Keycloak); success_url is the post-login URL "
            "to wait for. Both optional. A LIST is one login per origin (each "
            "entry carries `origin` or a `login_url` it is derived from) for a "
            "recording that signs into several apps; the package reads them from "
            "the JSON file named by AUTH_LOGINS_FILE, keyed by origin (or origin#label when several share it)."
        ),
    )

    # Options
    # Bounded like every sibling knob (workers/retries/timeout_per_test_ms).
    # There is no body-size middleware in this service, and execution_engine
    # expands each entry into a `--project <name>` argv pair, so an unbounded
    # list (e.g. ["chromium"] * 100_000) builds ~1.6 MB of argv and makes
    # create_subprocess_exec fail with "Argument list too long" — a self-DoS.
    # Three is the whole valid set, so the cap costs no legitimate caller.
    browsers: list[str] = Field(default_factory=lambda: ["chromium"], max_length=3)

    _VALID_BROWSERS: frozenset[str] = frozenset({"chromium", "firefox", "webkit"})

    @field_validator("target_url")
    @classmethod
    def validate_target_url(cls, v: str) -> str:
        """Reject non-HTTP URLs at the schema level."""
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"target_url must use http or https scheme: {v}")
        if not parsed.hostname:
            raise ValueError(f"target_url must have a valid hostname: {v}")
        return v

    @field_validator("auth_config")
    @classmethod
    def validate_auth_config(
        cls, v: AuthLogin | list[AuthLogin] | None
    ) -> AuthLogin | list[AuthLogin] | None:
        """Every login URL is strict (https, no parser differential); a login's
        ``origin`` and ``login_url`` agree; a list names each (origin, label)
        once, and an origin with several logins labels them all."""
        if v is None:
            return v
        if isinstance(v, list) and (not v or len(v) > MAX_AUTH_LOGINS):
            raise ValueError(f"auth_config list must hold 1..{MAX_AUTH_LOGINS} logins")
        keys: list[tuple[str, AuthLogin]] = []
        for entry in auth_logins(v):
            if entry.get("label") is not None and not (
                isinstance(entry["label"], str) and _LABEL_RE.fullmatch(entry["label"])
            ):
                raise ValueError(
                    "a login's label must be 1-40 lowercase letters, digits, '-' or '_'"
                )
            origins = set()
            for key in ("login_url", "origin"):
                if key in entry and entry[key] is not None:
                    strict_login_url(entry[key], key)
                    origins.add(canonical_origin(entry[key]))
            if len(origins) > 1:
                raise ValueError("a login's origin and login_url must name the same origin")
            if isinstance(v, list):
                origin = next(iter(origins), None)
                if origin is None:
                    raise ValueError(
                        "every auth_config list entry needs an https origin or login_url"
                    )
                keys.append((origin, entry))
        _validate_login_keys(keys)
        return v

    @model_validator(mode="after")
    def _login_bound_target_is_strict(self) -> TestExecutionRequest:
        # target_url is where a login without its own login_url is typed — the same
        # parser-differential exfiltration applies, so it is strict whenever a login is.
        if self.auth_config:
            strict_login_url(self.target_url, "target_url")
        return self

    @field_validator("browsers")
    @classmethod
    def validate_browsers(cls, v: list[str]) -> list[str]:
        """Reject unknown browser project names and de-duplicate, preserving order.

        Playwright's ``--project`` is a filter, not a repeat-count, so a repeated
        name selects the same project rather than re-running it: the duplicates
        are pure argv weight. Dedupe keeps first-seen order so the caller's
        browser precedence is unchanged.
        """
        valid = {"chromium", "firefox", "webkit"}
        invalid = [b for b in v if b not in valid]
        if invalid:
            raise ValueError(f"Unknown browsers: {invalid}. Valid: {sorted(valid)}")
        return list(dict.fromkeys(v))

    timeout_per_test_ms: int = Field(
        default=30_000,
        ge=1_000,
        le=600_000,
        description="Per-test timeout in ms (1s - 10min)",
    )
    retries: int = Field(
        default=0, ge=0, le=10, description="Number of retries per failing test (0 - 10)"
    )
    workers: int = Field(
        default=1, ge=1, le=16, description="Parallel Playwright worker processes (1 - 16)"
    )
    grep_pattern: str | None = Field(
        default=None, max_length=500, description="Filter which tests to run"
    )
    max_run_seconds: int | None = Field(
        default=None,
        ge=10,
        le=1_800,
        description=(
            "Optional wall-clock override for this run. Defaults to the service "
            "ceiling (EXECUTION_MAX_RUN_SECONDS). Pair with a single-`@clip` "
            "grep_pattern for a short-budget per-clip dry-run instead of a full "
            "harvest. Bounded under the orchestrator's ~1800s re-poll total so a "
            "record stage keeps polling the idempotent job."
        ),
    )
    screenshot_on_failure: bool = Field(
        default=True,
        description=(
            "NOT IMPLEMENTED — accepted for backwards compatibility and IGNORED. "
            "`playwright test` exposes no `--screenshot` flag, and the config "
            "carrying `use.screenshot` is generated by bdd_playwright_service, "
            "not by this service, so there is no lever here to honour it. "
            "Setting it to false adds an explicit notice to the job's "
            "`capture_warnings` rather than silently doing nothing; change the "
            "generator if you need failure screenshots off."
        ),
    )
    trace_on_failure: bool = Field(
        default=True,
        description=(
            "Record a Playwright trace on the first retry of a failing test. "
            "Wired through `playwright test --trace on-first-retry|off`, which "
            "overrides the generated config's hardcoded `on-first-retry`. Set "
            "false to keep trace zips (which the HTML report embeds and this "
            "service uploads to the shared object store) out of the artifacts."
        ),
    )
    expect_recording: bool = Field(
        default=True,
        description=(
            "Whether this package is expected to produce one video clip per "
            "scenario. When true (the default) a run that persists ZERO usable "
            "clips is FAILED rather than reported as a successful empty run — "
            "the orchestrator's demo_video_record stage would otherwise compile "
            "and bill for a footage-free video. Set false ONLY for a package "
            "that deliberately does not record (Playwright `use.video: 'off'`, "
            "or 'retain-on-failure' where all tests pass): there, N passing "
            "specs and zero clips is a healthy result and the zero-footage gate "
            "would be a false positive that the Idempotency-Key semantics make "
            "permanent for that key."
        ),
    )


# ── Response schemas ────────────────────────────────────────


class AsyncJobResponse(BaseModel):
    """Returned immediately when a job is created."""

    job_id: str
    status: str = "pending"
    poll_url: str


class TestSummary(BaseModel):
    """Aggregate pass/fail counts."""

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(extra="forbid")
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    flaky: int = 0
    pass_rate: float = 0.0


class TestCaseResult(BaseModel):
    """Individual test case outcome."""

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(extra="allow")
    name: str = ""
    feature: str = ""
    status: str = ""
    duration_ms: float = 0.0
    error: str = ""
    screenshot_url: str = ""
    trace_url: str = ""


class ScenarioClip(BaseModel):
    """A per-scenario recorded video clip harvested from a Playwright run.

    ``clip_key`` is the pipeline-wide join spine (the ``clip-<n>`` tag stamped by
    bdd_playwright_service) binding this clip to its narration part and compile
    shot. ``video_url`` is the tenant-scoped stream-through route on this service.
    """

    __test__ = False  # Not a pytest test class

    model_config = ConfigDict(extra="allow")
    clip_key: str = ""
    feature: str = ""
    scenario: str = ""
    status: str = ""
    video_url: str = ""
    duration_ms: float = 0.0
    size_bytes: int = 0
    has_audio: bool = False
    width: int = 0
    height: int = 0
    # Directed-footage sidecar (VP-2 schema), harvested from an injected
    # page-init track recorder and normalized into RECORDED-VIDEO pixel space.
    # All three are None when no ``w2v-track`` attachment was captured — an
    # ADDITIVE, backward-compatible extension: a track-less clip's model_dump()
    # gains three ``null`` keys, and the orchestrator reads them via ``.get()``.
    #   focus_track: [{"t": float, "bbox": {"x","y","w","h"}, "action": str}]
    #   cursor_track: [{"t": float, "x": float, "y": float, "click": bool}]
    #   viewport:    {"w": int, "h": int}  (recorded video WxH)
    focus_track: list[dict[str, Any]] | None = None
    cursor_track: list[dict[str, Any]] | None = None
    viewport: dict[str, int] | None = None


class JobStatusResponse(BaseModel):
    """Response for job status polling."""

    job_id: str
    status: str
    progress: float = 0.0
    error: str | None = None


class TestExecutionResult(BaseModel):
    """Full result of a test execution job."""

    __test__ = False  # Not a pytest test class

    job_id: str
    status: str
    target_url: str = ""
    summary: TestSummary = Field(default_factory=TestSummary)
    test_results: list[TestCaseResult] = Field(default_factory=list)
    report_url: str | None = None
    trace_urls: list[str] = Field(default_factory=list)
    scenario_clips: list[ScenarioClip] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    config: dict[str, Any] = Field(default_factory=dict)
    # Non-fatal capture defects that make the footage undirected (see
    # ExecutionEngine._ensure_track_recorder / _track_coverage_warnings).
    # Empty == the directed-footage track recorder was wired AND recorded.
    capture_warnings: list[str] = Field(default_factory=list)
