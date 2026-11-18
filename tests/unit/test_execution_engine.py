"""Tests for the execution engine."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ipaddress
import json
import logging
import re
import socket
import zipfile
from pathlib import Path
from typing import Any, ClassVar, cast
from urllib.parse import quote, quote_plus

import pytest

from src.api.schemas import TestCaseResult, TestExecutionRequest
from src.services.execution_engine import (
    ExecutionEngine,
    _clamp_error_text,
    _clip_key_for_spec,
    _convert_to_mp4,
    _count_specs,
    _ffprobe_meta,
    _load_and_normalize_track,
    _normalize_track,
    _probe_video_fps,
    _ReportWaitOutcome,
    _resolve_lead_trim_seconds,
    _safe_extract,
    _salvage_sort_key,
    _subprocess_error_text,
    _validate_url,
)


def _fake_getaddrinfo(mapping: dict[str, str]):
    """Build a ``socket.getaddrinfo`` stub driven by a hostname -> IP mapping.

    IP literals resolve to themselves (that is what the real resolver does);
    an unmapped hostname raises ``gaierror`` so the guard's fail-closed DNS
    path is exercised rather than hitting the network from a unit test.
    """

    def _resolve(host, *args, **kwargs):
        try:
            addr = str(ipaddress.ip_address(host))
        except ValueError:
            if host not in mapping:
                raise socket.gaierror(f"unmapped host in test: {host}") from None
            addr = mapping[host]
        is_v6 = isinstance(ipaddress.ip_address(addr), ipaddress.IPv6Address)
        family = socket.AF_INET6 if is_v6 else socket.AF_INET
        return [(family, socket.SOCK_STREAM, 6, "", (addr, 0))]

    return _resolve


_TEST_DNS = {
    "example.com": "93.184.216.34",
    "auth.example.com": "93.184.216.34",
    # The bypass proof: an innocuous-looking public hostname that maps into the
    # Tailscale tailnet (the live postgres primary). The pre-fix _validate_url
    # returned early for ANY hostname and allowed this.
    "internal-thing.example.com": "100.64.0.5",
    "metadata-ish.example.com": "169.254.169.254",
}


@pytest.fixture(autouse=True)
def _stub_dns(monkeypatch):
    """No unit test may depend on live DNS.

    ``_validate_url`` now delegates to ``w2v_common.net.validate_public_url``,
    which does a real ``socket.getaddrinfo`` and fails CLOSED. Every test that
    touches a URL path (``_configure_target``, ``_resolve_package``) therefore
    needs a deterministic resolver.
    """
    monkeypatch.setattr(socket, "getaddrinfo", _fake_getaddrinfo(_TEST_DNS))


class TestValidateUrl:
    """SSRF protection — _validate_url (delegates to w2v_common.net)."""

    # ── still accepted ────────────────────────────────────────────────────
    def test_accepts_public_https(self):
        _validate_url("https://example.com/pkg.zip")

    def test_accepts_public_http(self):
        _validate_url("http://example.com/foo.zip")

    def test_accepts_public_ip_literal(self):
        _validate_url("http://8.8.8.8/x")

    def test_accepts_addresses_just_outside_the_cgnat_block(self):
        """The deny is the /10, not a ``startswith("100.")`` prefix match."""
        _validate_url("http://100.63.255.255/x")  # one below 100.64.0.0
        _validate_url("http://100.128.0.0/x")  # one above 100.127.255.255

    # ── scheme / hostname contract (unchanged messages) ───────────────────
    def test_rejects_other_schemes(self):
        for bad in ("file:///etc/passwd", "ftp://example.com/a", "gopher://x"):
            with pytest.raises(ValueError, match="http or https"):
                _validate_url(bad)

    def test_rejects_missing_hostname(self):
        with pytest.raises(ValueError, match="valid hostname"):
            _validate_url("https://")

    # ── the bypass proof ──────────────────────────────────────────────────
    def test_rejects_hostname_resolving_into_the_tailnet(self):
        """THE regression this fix exists for.

        Before delegating to ``validate_public_url`` this returned early for
        every non-IP-literal hostname with no DNS lookup at all, so a tenant
        could name any host that resolved to an internal address.
        """
        with pytest.raises(ValueError, match=re.escape("blocked address 100.64.0.5")):
            _validate_url("http://internal-thing.example.com/pkg.zip")

    def test_rejects_hostname_resolving_to_cloud_metadata(self):
        with pytest.raises(ValueError, match=re.escape("blocked address 169.254.169.254")):
            _validate_url("http://metadata-ish.example.com/latest/meta-data/")

    def test_unresolvable_hostname_fails_closed(self):
        with pytest.raises(ValueError, match="DNS resolution failed"):
            _validate_url("http://internal.example/foo")

    # ── literals rejected ─────────────────────────────────────────────────
    def test_rejects_cgnat_literals(self):
        """RFC 6598 100.64.0.0/10 — ``ipaddress.is_private`` is False for these."""
        for ip in ("100.64.0.1", "100.64.0.5", "100.64.0.6", "100.64.0.7"):
            assert not ipaddress.ip_address(ip).is_private  # why the old check missed it
            with pytest.raises(ValueError, match="blocked address"):
                _validate_url(f"http://{ip}/x")

    def test_rejects_ipv4_mapped_ipv6_cgnat_literal(self):
        with pytest.raises(ValueError, match="blocked address"):
            _validate_url("http://[::ffff:100.64.0.5]/x")

    def test_rejects_private_ipv4(self):
        for ip in ("10.0.0.1", "192.168.1.1", "172.16.0.1"):
            with pytest.raises(ValueError, match="blocked address"):
                _validate_url(f"http://{ip}/foo")

    def test_rejects_loopback(self):
        with pytest.raises(ValueError, match="blocked address"):
            _validate_url("http://127.0.0.1/foo")

    def test_rejects_link_local(self):
        with pytest.raises(ValueError, match="blocked address"):
            _validate_url("http://169.254.169.254/latest/meta-data/")

    def test_rejects_multicast_and_reserved(self):
        for ip in ("224.0.0.1", "240.0.0.1", "0.0.0.0"):
            with pytest.raises(ValueError, match="blocked address"):
                _validate_url(f"http://{ip}/x")

    # ── escape hatch preserved ────────────────────────────────────────────
    def test_allow_private_bypass(self):
        """allow_private=True is used for trusted internal URLs (bdd S2S)."""
        _validate_url("http://10.0.0.1/foo", allow_private=True)
        _validate_url("http://127.0.0.1/foo", allow_private=True)
        _validate_url("http://100.64.0.5/foo", allow_private=True)
        _validate_url("http://bdd-playwright.w2v.svc.cluster.local:6315/x", allow_private=True)

    def test_allow_private_still_enforces_scheme(self):
        with pytest.raises(ValueError, match="http or https"):
            _validate_url("file:///etc/passwd", allow_private=True)


class TestSafeExtract:
    """_safe_extract — path traversal, symlink, and size checks."""

    def _make_zip(self, tmp_path: Path, entries: list[tuple[str, bytes]]) -> Path:
        zip_path = tmp_path / "archive.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for name, data in entries:
                zf.writestr(name, data)
        return zip_path

    def test_extracts_normal_zip(self, tmp_path: Path):
        zip_path = self._make_zip(tmp_path, [("a.txt", b"hello"), ("sub/b.txt", b"world")])
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            _safe_extract(zf, dest)
        assert (dest / "a.txt").read_bytes() == b"hello"
        assert (dest / "sub" / "b.txt").read_bytes() == b"world"

    def test_rejects_path_traversal(self, tmp_path: Path):
        zip_path = self._make_zip(tmp_path, [("../escaped.txt", b"bad")])
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(ValueError, match="path traversal"):
                _safe_extract(zf, dest)

    def test_rejects_oversized_archive(self, tmp_path: Path, monkeypatch):
        # Forge a small ceiling so we can trigger the limit without writing 500MB
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod, "_MAX_UNCOMPRESSED_BYTES", 128, raising=True)
        zip_path = self._make_zip(tmp_path, [("big.bin", b"x" * 256)])
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(ValueError, match="uncompressed size"):
                _safe_extract(zf, dest)

    def _make_forged_zip(self, tmp_path: Path, name: str, payload: bytes, declared: int) -> Path:
        """A real DEFLATE zip whose header ``file_size`` fields are patched DOWN to
        ``declared`` while the compressed data + CRC stay honest.

        This is the decompression-bomb shape: an attacker understates the
        uncompressed size so a pre-check that sums ``ZipInfo.file_size`` passes,
        even though the deflate stream actually expands to ``len(payload)``.
        """
        import struct

        honest = tmp_path / f"{name}.honest.zip"
        with zipfile.ZipFile(honest, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(name, payload)
        raw = bytearray(honest.read_bytes())
        # Patch the uncompressed-size field in the local file header (offset +22
        # from PK\x03\x04) and central directory record (offset +24 from PK\x01\x02).
        for sig, off in ((b"PK\x03\x04", 22), (b"PK\x01\x02", 24)):
            i = 0
            while (j := raw.find(sig, i)) >= 0:
                if struct.unpack_from("<I", raw, j + off)[0] > 0:
                    struct.pack_into("<I", raw, j + off, declared)
                i = j + 4
        forged = tmp_path / f"{name}.forged.zip"
        forged.write_bytes(bytes(raw))
        return forged

    def test_rejects_decompression_bomb_with_understated_size(self, tmp_path: Path, monkeypatch):
        """The cap must bind on bytes ACTUALLY decompressed, not the declared
        ``file_size`` header — which an attacker sets to 0 to slip a bomb past a
        pre-check that trusts it."""
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod, "_MAX_UNCOMPRESSED_BYTES", 256 * 1024, raising=True)
        payload = b"A" * (2 * 1024 * 1024)  # 2 MB actual, >> 256 KB cap
        forged = self._make_forged_zip(tmp_path, "bomb.bin", payload, declared=0)
        dest = tmp_path / "out"
        dest.mkdir()
        with zipfile.ZipFile(forged) as zf:
            with pytest.raises(ValueError, match="uncompressed size"):
                _safe_extract(zf, dest)
        # Aborted cleanly: no partial bomb payload left behind.
        assert not (dest / "bomb.bin").exists() or (dest / "bomb.bin").stat().st_size <= 256 * 1024

    def test_rejects_symlink_escape(self, tmp_path: Path):
        """A symlink inside the sandbox pointing outside of it is rejected.

        Python's zipfile.extractall does not materialize symlinks from ZIP
        entries (it writes the target string as a regular file), so the
        post-extraction rglob check in _safe_extract is defense-in-depth.
        We simulate the bad state by pre-placing a symlink in dest and
        extracting an empty ZIP — the post-check then runs against the
        pre-placed symlink.
        """
        outside = tmp_path / "outside.txt"
        outside.write_text("secret")

        zip_path = self._make_zip(tmp_path, [])
        dest = tmp_path / "out"
        dest.mkdir()
        (dest / "link").symlink_to(outside)

        with zipfile.ZipFile(zip_path) as zf:
            with pytest.raises(ValueError, match="symlink escaping"):
                _safe_extract(zf, dest)


class TestSubprocessEnv:
    """_subprocess_env — strips secrets, keeps only safe keys."""

    def test_strips_secret_env_vars(self, monkeypatch):
        monkeypatch.setenv("DATABASE_URL", "postgresql://secret")
        monkeypatch.setenv("API_KEY", "topsecret")
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = ExecutionEngine._subprocess_env()
        assert "DATABASE_URL" not in env
        assert "API_KEY" not in env
        assert env["PATH"] == "/usr/bin:/bin"

    def test_keeps_allow_listed_keys(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/tester")
        monkeypatch.setenv("LANG", "en_US.UTF-8")
        env = ExecutionEngine._subprocess_env()
        assert env["HOME"] == "/home/tester"
        assert env["LANG"] == "en_US.UTF-8"

    def test_keeps_playwright_browsers_path(self, monkeypatch):
        """Without this, `npx playwright test` can't find the baked browsers and
        the run produces zero footage — the record-stage 'still running' wall."""
        monkeypatch.setenv("PLAYWRIGHT_BROWSERS_PATH", "/ms-playwright")
        env = ExecutionEngine._subprocess_env()
        assert env["PLAYWRIGHT_BROWSERS_PATH"] == "/ms-playwright"

    def test_merges_extra_values(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        env = ExecutionEngine._subprocess_env({"BASE_URL": "https://x"})
        assert env["BASE_URL"] == "https://x"
        assert env["PATH"] == "/usr/bin"


class TestConfigureTarget:
    """_configure_target — writes BASE_URL into .env and validates."""

    def _request(self, target_url: str) -> TestExecutionRequest:
        return TestExecutionRequest(bdd_job_id="bdd-1", target_url=target_url)

    def test_writes_env_file(self, tmp_path: Path):
        engine = ExecutionEngine()
        engine._configure_target(tmp_path, self._request("https://example.com"))
        assert (tmp_path / ".env").read_text() == "BASE_URL=https://example.com\n"

    def test_forwards_sso_button_and_success_url(self, tmp_path: Path):
        """auth_config.sso_button / success_url reach the package .env so the
        scaffold auth.setup.ts SSO branch (Keycloak et al.) is not starved of
        input — regression guard for the dead-path bug.
        """
        engine = ExecutionEngine()
        req = TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={
                "login_url": "https://auth.example.com/login",
                "username": "user@example.com",
                "password": "hunter2",  # pragma: allowlist secret
                "sso_button": "Sign in with Keycloak",
                "success_url": "https://example.com/dashboard",
            },
        )
        engine._configure_target(tmp_path, req)
        env = (tmp_path / ".env").read_text()
        assert "AUTH_SSO_BUTTON=Sign in with Keycloak" in env
        assert "AUTH_SUCCESS_URL=https://example.com/dashboard" in env

    def test_omits_sso_env_when_not_provided(self, tmp_path: Path):
        """A plain inline-form login must not write blank SSO overrides that
        would shadow the scaffold's "not /login" fallback.
        """
        engine = ExecutionEngine()
        req = TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={
                "login_url": "https://example.com/login",
                "username": "user@example.com",
                "password": "hunter2",  # pragma: allowlist secret
            },
        )
        engine._configure_target(tmp_path, req)
        env = (tmp_path / ".env").read_text()
        assert "AUTH_SSO_BUTTON" not in env
        assert "AUTH_SUCCESS_URL" not in env

    def test_rejects_newline_injection(self, tmp_path: Path):
        """target_url with a newline is caught by _configure_target's guard.

        The schema-layer validator only enforces scheme/hostname, so a newline
        can slip past pydantic — the belt-and-suspenders check in
        _configure_target is what actually blocks .env injection.
        """
        engine = ExecutionEngine()
        req = self._request("https://example.com\nX-Evil: 1")
        with pytest.raises(ValueError, match="newline"):
            engine._configure_target(tmp_path, req)

    def test_rejects_private_target(self, tmp_path: Path):
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match="blocked address"):
            engine._configure_target(tmp_path, self._request("http://10.0.0.1"))

    def test_rejects_target_hostname_resolving_into_the_tailnet(self, tmp_path: Path):
        """target_url is tenant-supplied — a hostname pointing at 100.64/10
        (this estate's Tailnet: postgres, registry, the k3s control plane) was
        accepted before the guard resolved hostnames."""
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match=re.escape("blocked address 100.64.0.5")):
            engine._configure_target(tmp_path, self._request("https://internal-thing.example.com"))


class TestConfigureTargetLoginUrlGuard:
    """``auth_config.login_url`` gets the SAME SSRF guard as ``target_url``.

    ``target_url`` was validated at ``_configure_target`` line 776 while
    ``login_url``, two lines below, was only newline-checked before being
    written to ``AUTH_LOGIN_URL``. The scaffold's ``auth.setup.ts`` navigates
    to that URL and **types the tenant's real credentials into it**, so an
    attacker-chosen ``login_url`` is a credential-harvesting sink, not merely
    an SSRF probe. A guarded call site sitting two lines above an unguarded
    one is the defect class this fleet tracks.
    """

    def _request(self, login_url: str) -> TestExecutionRequest:
        return TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={
                "login_url": login_url,
                "username": "user@example.com",
                "password": "hunter2",  # pragma: allowlist secret
            },
        )

    def test_rejects_private_login_url(self, tmp_path: Path):
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match="blocked address"):
            engine._configure_target(tmp_path, self._request("https://10.0.0.1/login"))

    def test_rejects_login_url_resolving_into_the_tailnet(self, tmp_path: Path):
        """The bypass proof: an innocuous public hostname mapping to 100.64/10."""
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match=re.escape("blocked address 100.64.0.5")):
            engine._configure_target(
                tmp_path, self._request("https://internal-thing.example.com/login")
            )

    def test_rejects_login_url_pointing_at_cloud_metadata(self, tmp_path: Path):
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match="blocked address"):
            engine._configure_target(
                tmp_path, self._request("https://metadata-ish.example.com/login")
            )

    def test_rejects_non_http_login_url_scheme(self, tmp_path: Path):
        """``file://`` / ``chrome://`` are reachable navigation targets for a
        browser runner; only https may reach AUTH_LOGIN_URL (a password never
        travels in cleartext)."""
        engine = ExecutionEngine()
        with pytest.raises(ValueError, match="must use https"):
            engine._configure_target(tmp_path, self._request("file:///etc/passwd"))

    def test_accepts_public_login_url_on_a_different_host(self, tmp_path: Path):
        """A legitimate external IdP (login on auth.example.com while the app
        under test is example.com) must still pass — the guard is a public/
        private check, not a same-origin check."""
        engine = ExecutionEngine()
        engine._configure_target(tmp_path, self._request("https://auth.example.com/login"))
        env = (tmp_path / ".env").read_text()
        assert "AUTH_LOGIN_URL=https://auth.example.com/login" in env

    def test_defaulted_login_url_still_written(self, tmp_path: Path):
        """When auth_config omits login_url it defaults to target_url, which is
        already validated — the default path must not regress."""
        engine = ExecutionEngine()
        req = TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={"username": "u", "password": "p"},  # pragma: allowlist secret
        )
        engine._configure_target(tmp_path, req)
        assert "AUTH_LOGIN_URL=https://example.com" in (tmp_path / ".env").read_text()


class TestAuthCredentialsNotInSubprocessEnv:
    """H3: AUTH_PASSWORD must not appear in the subprocess environment.

    /proc/<pid>/environ is readable by neighbour pods sharing the node UID
    on default K3s, so secrets are delivered via the 0600 .env file inside
    the 0700 sandbox tempdir, not via env vars.
    """

    def _request_with_auth(self) -> TestExecutionRequest:
        return TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={
                "login_url": "https://example.com/login",
                "username": "user@example.com",
                "password": "hunter2-supersecret",
            },
        )

    async def test_subprocess_env_omits_auth_password(self, tmp_path, monkeypatch):
        """Snapshot the env passed to asyncio.create_subprocess_exec — no AUTH_*."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text(
            '{"name":"x","version":"1.0.0","devDependencies":{}}'
        )
        (package_dir / "playwright.config.ts").write_text(
            "import { defineConfig } from '@playwright/test';\nexport default defineConfig({});\n"
        )

        request = self._request_with_auth()
        engine._configure_target(package_dir, request)

        captured: dict[str, Any] = {}

        async def _stub_create_subprocess_exec(*_args: Any, **kwargs: Any) -> Any:
            _ = _args
            captured["env"] = kwargs.get("env", {})
            # _run_tests now reads the report from disk (stdout goes to a file),
            # so emulate Playwright writing results.json + exiting cleanly.
            cwd = kwargs.get("cwd")
            if cwd:
                (Path(cwd) / "results.json").write_text('{"suites":[]}')

            class _Proc:
                pid = 4242
                returncode = 0

                async def wait(self):
                    return 0

            return _Proc()

        monkeypatch.setattr(
            "asyncio.create_subprocess_exec",
            _stub_create_subprocess_exec,
        )
        # Neutralise the process-group cleanup for the fake pid (treat its pgid as
        # FTR's own group so _terminate_process_group refuses to signal it).
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)

        await engine._run_tests(package_dir, request)

        env = captured["env"]
        assert "AUTH_PASSWORD" not in env
        assert "AUTH_USERNAME" not in env
        assert "AUTH_LOGIN_URL" not in env
        # No env value should equal the password — guards against future drift.
        assert "hunter2-supersecret" not in env.values()
        # The harmless ones should still be there.
        assert env["BASE_URL"] == "https://example.com"

    def test_env_file_is_mode_0600(self, tmp_path):
        """The .env file inside the sandbox dir is mode 0600 (owner-only r/w)."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text('{"name":"x","version":"1.0.0"}')
        (package_dir / "playwright.config.ts").write_text("export default {};")
        engine._configure_target(package_dir, self._request_with_auth())

        env_file = package_dir / ".env"
        mode = env_file.stat().st_mode & 0o777
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_dotenv_loader_injected_when_auth_present(self, tmp_path):
        """playwright.config.ts gains ``import 'dotenv/config'`` so creds load
        in-process and never appear in any /proc/<pid>/environ snapshot.
        """
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text(
            '{"name":"x","version":"1.0.0","devDependencies":{}}'
        )
        (package_dir / "playwright.config.ts").write_text(
            "import { defineConfig } from '@playwright/test';\nexport default defineConfig({});\n"
        )
        engine._configure_target(package_dir, self._request_with_auth())

        config = (package_dir / "playwright.config.ts").read_text()
        assert "import 'dotenv/config';" in config

        import json as _json

        pkg = _json.loads((package_dir / "package.json").read_text())
        # dotenv must be added to dependencies (not devDependencies) so npm ci
        # / npm install picks it up regardless of --omit settings.
        assert "dotenv" in pkg.get("dependencies", {})

    def test_dotenv_loader_idempotent(self, tmp_path):
        """Re-running _configure_target must not double-inject dotenv."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text('{"name":"x","version":"1.0.0"}')
        (package_dir / "playwright.config.ts").write_text("export default {};")
        engine._configure_target(package_dir, self._request_with_auth())
        engine._configure_target(package_dir, self._request_with_auth())
        config = (package_dir / "playwright.config.ts").read_text()
        assert config.count("import 'dotenv/config';") == 1

    def test_env_file_contains_credentials(self, tmp_path):
        """Sanity: .env carries the AUTH_* values that the subprocess env no
        longer carries — playwright loads them via dotenv at config-load time.
        """
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        engine._configure_target(package_dir, self._request_with_auth())
        body = (package_dir / ".env").read_text()
        assert "AUTH_USERNAME=user@example.com" in body
        assert "AUTH_PASSWORD=hunter2-supersecret" in body
        assert "AUTH_LOGIN_URL=https://example.com/login" in body

    def test_rejects_newline_in_auth_password(self, tmp_path):
        """Newline-in-credential blocks .env injection (defense in depth)."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        req = TestExecutionRequest(
            bdd_job_id="bdd-1",
            target_url="https://example.com",
            auth_config={
                "login_url": "https://example.com/login",
                "username": "user",
                "password": "pw\nAUTH_USERNAME=attacker",
            },
        )
        with pytest.raises(ValueError, match="newline"):
            engine._configure_target(package_dir, req)


class TestWallBudgetOverride:
    """max_run_seconds override reaches the run's wall-clock backstop."""

    @staticmethod
    def _stub_run(engine: ExecutionEngine, monkeypatch, tmp_path: Path) -> dict[str, Any]:
        """Stub the subprocess + report-wait so _run_tests reaches the wall-budget
        computation, capturing the timeout it hands to _wait_for_report_or_exit."""
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text(
            '{"name":"x","version":"1.0.0","devDependencies":{}}'
        )
        (package_dir / "playwright.config.ts").write_text(
            "import { defineConfig } from '@playwright/test';\nexport default defineConfig({});\n"
        )
        captured: dict[str, Any] = {}

        async def _stub_create_subprocess_exec(*_args: Any, **kwargs: Any) -> Any:
            cwd = kwargs.get("cwd")
            if cwd:
                (Path(cwd) / "results.json").write_text('{"suites":[]}')

            class _Proc:
                pid = 4242
                returncode = 0

                async def wait(self):
                    return 0

            return _Proc()

        async def _stub_wait(
            _proc: Any, _report: Path, timeout_seconds: float, **_kwargs: Any
        ) -> _ReportWaitOutcome:
            captured["timeout_seconds"] = timeout_seconds
            return _ReportWaitOutcome()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _stub_create_subprocess_exec)
        monkeypatch.setattr(engine, "_wait_for_report_or_exit", _stub_wait)
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)
        return {"package_dir": package_dir, "captured": captured}

    async def test_override_honored(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        ctx = self._stub_run(engine, monkeypatch, tmp_path)
        request = TestExecutionRequest(
            bdd_job_id="j", target_url="https://example.com", max_run_seconds=45
        )
        await engine._run_tests(ctx["package_dir"], request)
        assert ctx["captured"]["timeout_seconds"] == 45

    async def test_falls_back_to_settings_ceiling(self, tmp_path, monkeypatch):
        from src.config import get_settings

        engine = ExecutionEngine()
        ctx = self._stub_run(engine, monkeypatch, tmp_path)
        request = TestExecutionRequest(bdd_job_id="j", target_url="https://example.com")
        await engine._run_tests(ctx["package_dir"], request)
        assert ctx["captured"]["timeout_seconds"] == get_settings().execution.max_run_seconds


class TestZeroClipSalvageFailsTheRun:
    """A run that produces no usable footage must FAIL, so _process_job lands the
    job at `failed`. Landing `completed` with `scenario_clips: []` would let the
    orchestrator's demo_video_record stage compile (and bill for) a footage-free
    video.

    The failure decision does NOT live in the timeout branch — it lives in
    `_guard_zero_footage`, downstream of persistence (see TestZeroFootageGate).
    `_run_tests` only salvages + logs; a PARTIAL salvage must still return, which
    is why the salvage path exists at all.
    """

    @staticmethod
    def _stub_timeout_run(engine: ExecutionEngine, monkeypatch, tmp_path: Path) -> Path:
        """Spawn stub + a report-wait that always blows the wall budget, so
        _run_tests takes the TimeoutError salvage branch."""
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()

        async def _stub_create_subprocess_exec(*_args: Any, **_kwargs: Any) -> Any:
            class _Proc:
                pid = 4242
                returncode = 0

                async def wait(self) -> int:
                    return 0

            return _Proc()

        async def _stub_wait(
            _proc: Any, _report: Path, _timeout_seconds: float, **_kwargs: Any
        ) -> None:
            raise TimeoutError

        monkeypatch.setattr("asyncio.create_subprocess_exec", _stub_create_subprocess_exec)
        monkeypatch.setattr(engine, "_wait_for_report_or_exit", _stub_wait)
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)
        return package_dir

    async def test_timeout_with_no_footage_returns_empty_salvage_and_logs(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """The timeout branch no longer raises — it salvages, logs the actionable
        cause at ERROR, and returns. `_guard_zero_footage` fails the run."""
        engine = ExecutionEngine()
        package_dir = self._stub_timeout_run(engine, monkeypatch, tmp_path)
        # No test-results/**/*.webm at all → salvage is empty.
        request = TestExecutionRequest(
            bdd_job_id="j", target_url="https://example.com", max_run_seconds=45
        )
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            raw, timed_out = await engine._run_tests(package_dir, request)
        assert timed_out is True
        assert ExecutionEngine()._collect_clip_specs(raw) == []
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and "never launched" in errors[0].getMessage()

    async def test_run_tests_reports_timed_out_false_on_a_clean_run(
        self, tmp_path: Path, monkeypatch
    ):
        """The timeout flag is the gate's only reliable timeout signal, so a
        NORMAL run must report False — otherwise every healthy non-recording run
        would trip the gate."""
        engine = ExecutionEngine()
        ctx = TestWallBudgetOverride._stub_run(engine, monkeypatch, tmp_path)
        request = TestExecutionRequest(
            bdd_job_id="j", target_url="https://example.com", max_run_seconds=45
        )
        _raw, timed_out = await engine._run_tests(ctx["package_dir"], request)
        assert timed_out is False


def _nested_report_with_specs(
    n: int, *, video_path: Path | None, status: str = "passed"
) -> dict[str, Any]:
    """Build a Playwright JSON report shaped the way bddgen actually emits one:
    specs live inside a child describe-suite, NOT at `raw["suites"][*]["specs"]`.
    """
    specs: list[dict[str, Any]] = []
    for i in range(1, n + 1):
        attachments: list[dict[str, Any]] = []
        if video_path is not None:
            attachments.append({"name": "video", "path": str(video_path)})
        specs.append(
            {
                "title": f"Scenario {i}",
                "tags": [f"@clip-{i}"],
                "tests": [
                    {"results": [{"status": status, "duration": 900, "attachments": attachments}]}
                ],
            }
        )
    return {
        "suites": [
            {
                "title": "Login.feature",
                "specs": [],
                "suites": [{"title": "Login", "specs": specs, "suites": []}],
            }
        ]
    }


class TestZeroFootageGate:
    """`_guard_zero_footage` — the ONE authoritative zero-footage failure, sitting
    downstream of `_persist_artifacts` so it composes with every route to an empty
    harvest: timeout salvage, normal run, and all-clips-degenerate."""

    @staticmethod
    def _stub_pipeline(engine: ExecutionEngine, monkeypatch, package_dir: Path) -> None:
        """Neuter every execute() step before _run_tests."""

        async def _resolve(_request: Any, _tmp: Path, _token: str | None) -> Path:
            return package_dir

        async def _noop_async(*_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr(engine, "_resolve_package", _resolve)
        monkeypatch.setattr(engine, "_install_dependencies", _noop_async)
        monkeypatch.setattr(engine, "_configure_target", lambda *_a, **_k: None)
        monkeypatch.setattr(engine, "_ensure_track_recorder", lambda *_a, **_k: [])
        monkeypatch.setattr(engine, "_run_bddgen", _noop_async)

    @staticmethod
    def _degenerate_probe(monkeypatch) -> None:
        async def _zero(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 0.0, 0, 0, False, True

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _zero)

    async def _execute(
        self,
        engine: ExecutionEngine,
        tmp_path: Path,
        package_dir: Path,
        *,
        expect_recording: bool = True,
        grep_pattern: str | None = None,
    ) -> dict[str, Any]:
        from tests.conftest import make_local_storage

        engine._storage = make_local_storage(tmp_path / "store")
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            max_run_seconds=45,
            expect_recording=expect_recording,
            grep_pattern=grep_pattern,
        )
        return await engine.execute(
            request,
            persist_dir=tmp_path / "job-gate",
            public_artifact_base="/api/test-runner/job-gate/artifacts",
            work_dir=package_dir,
        )

    async def test_timeout_with_all_degenerate_clips_raises(self, tmp_path: Path, monkeypatch):
        """D2: N clips salvaged but every one degenerate ⇒ dropped ⇒ zero footage.
        The old in-branch `n_clips > 0` test passed this straight to `completed`."""
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = TestZeroClipSalvageFailsTheRun._stub_timeout_run(
            engine, monkeypatch, tmp_path
        )
        self._stub_pipeline(engine, monkeypatch, package_dir)
        self._degenerate_probe(monkeypatch)
        for n in (1, 2):
            clip_dir = package_dir / "test-results" / f"login-clip-{n}-chromium"
            clip_dir.mkdir(parents=True)
            (clip_dir / "video.webm").write_bytes(b"truncated")

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            with pytest.raises(RuntimeError, match="no usable footage"):
                await self._execute(engine, tmp_path, package_dir)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_non_timeout_run_with_all_degenerate_clips_raises(
        self, tmp_path: Path, monkeypatch
    ):
        """D3: a clean (non-timeout) run whose every clip is degenerate never
        entered the timeout branch at all, so the old guard could not see it."""
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        video = package_dir / "rec.webm"
        video.write_bytes(b"truncated")
        report = {
            "suites": [
                {
                    "title": "Login.feature",
                    "specs": [],
                    "suites": [
                        {
                            "title": "Login",
                            "specs": [
                                {
                                    "title": "Successful login",
                                    "tags": ["@clip-1"],
                                    "tests": [
                                        {
                                            "results": [
                                                {
                                                    "status": "passed",
                                                    "duration": 900,
                                                    "attachments": [
                                                        {"name": "video", "path": str(video)}
                                                    ],
                                                }
                                            ]
                                        }
                                    ],
                                }
                            ],
                            "suites": [],
                        }
                    ],
                }
            ]
        }

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        self._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        self._degenerate_probe(monkeypatch)

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            with pytest.raises(RuntimeError, match="no usable footage"):
                await self._execute(engine, tmp_path, package_dir)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_zero_specs_declared_completes_normally(self, tmp_path: Path, monkeypatch):
        """D-guard scope: an empty suite declared nothing to record. It must still
        complete — the gate must not start 500-ing on a legitimately empty run."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return {"suites": []}, False

        self._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)

        result = await self._execute(engine, tmp_path, package_dir)
        assert result["scenario_clips"] == []
        assert result["summary"]["total"] == 0

    @staticmethod
    def _healthy_probe(monkeypatch) -> None:
        async def _ok(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 1200.0, 1280, 720, False, True

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _ok)

    async def test_timeout_with_empty_synthetic_salvage_raises(self, tmp_path: Path, monkeypatch):
        """DEFECT A (headline). The zero-footage TIMEOUT path — no results.json,
        no .webm — makes `_salvage_raw_from_disk` return the SYNTHETIC empty
        report, which parses to total=0. A gate that reads only the report calls
        that "declared nothing to record" and completes the job with
        `scenario_clips: []`, letting the orchestrator bill a footage-free video.
        The out-of-band `timed_out` flag is what makes it fire.

        This drives the REAL `_run_tests` timeout branch end-to-end through
        `execute()` — it is not a hand-built gate argument.
        """
        engine = ExecutionEngine()
        package_dir = TestZeroClipSalvageFailsTheRun._stub_timeout_run(
            engine, monkeypatch, tmp_path
        )
        self._stub_pipeline(engine, monkeypatch, package_dir)
        # Deliberately NO test-results/**/*.webm and NO results.json.
        assert not (package_dir / "results.json").exists()

        with pytest.raises(RuntimeError, match="no usable footage"):
            await self._execute(engine, tmp_path, package_dir)

    async def test_timeout_with_partial_footage_returns_normally(self, tmp_path: Path, monkeypatch):
        """DEFECT C rewrite. The old test of this name asserted on
        `_collect_clip_specs(raw)` and never invoked `execute()` or the gate, so
        it could not detect the gate OVER-firing on a partial salvage. This one
        drives the whole path: a timed-out run that salvaged ONE healthy clip
        must complete, with that clip persisted.
        """
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = TestZeroClipSalvageFailsTheRun._stub_timeout_run(
            engine, monkeypatch, tmp_path
        )
        self._stub_pipeline(engine, monkeypatch, package_dir)
        self._healthy_probe(monkeypatch)
        clip_dir = package_dir / "test-results" / "login-clip-1-chromium"
        clip_dir.mkdir(parents=True)
        (clip_dir / "video.webm").write_bytes(b"a real recording")

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            result = await self._execute(engine, tmp_path, package_dir)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True
        assert [c["clip_key"] for c in result["scenario_clips"]] == ["clip-1"]

    async def test_expect_recording_false_with_specs_and_zero_clips_completes(
        self, tmp_path: Path, monkeypatch
    ):
        """DEFECT B. A package with Playwright `use.video: 'off'` legitimately
        yields N passing specs and zero clips. Hard-failing it would be a false
        positive — and because `job_manager` returns the existing row on an
        Idempotency-Key conflict regardless of status, that `failed` row would be
        handed back to EVERY resubmit under the same key, permanently.
        """
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        report = _nested_report_with_specs(5, video_path=None)

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        self._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)

        result = await self._execute(engine, tmp_path, package_dir, expect_recording=False)
        assert result["scenario_clips"] == []
        assert result["summary"]["total"] == 5

    async def test_expect_recording_true_is_the_default_and_still_fails(
        self, tmp_path: Path, monkeypatch
    ):
        """The opt-out must be OPT-IN: the orchestrator always records, so the
        same run without `expect_recording=False` must still fail."""
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        report = _nested_report_with_specs(5, video_path=None)

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        self._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)

        with pytest.raises(RuntimeError, match="no usable footage"):
            await self._execute(engine, tmp_path, package_dir)

    def test_gate_fires_on_timeout_even_when_the_report_declares_nothing(self):
        """DEFECT A at the gate's own boundary: total=0 + zero clips is a PASS
        without the timeout flag and a FAIL with it. This is the exact pair of
        arguments the synthetic empty salvage produces."""
        empty = {"summary": {"total": 0, "skipped": 0}, "scenario_clips": [], "raw": {"suites": []}}
        ExecutionEngine._guard_zero_footage(empty, timed_out=False)
        with pytest.raises(RuntimeError, match="no usable footage") as excinfo:
            ExecutionEngine._guard_zero_footage(empty, timed_out=True)
        assert "wall-clock budget" in str(excinfo.value)

    def test_gate_reads_the_raw_spec_tree_recursively(self):
        """Playwright nests specs inside the per-`Feature:` describe suite bddgen
        emits. If the summary somehow reads 0 but the raw tree holds recordable
        specs, the gate must still fire — and a top-level-only `sum()` over
        `raw["suites"]` would see 0 here."""
        with pytest.raises(RuntimeError, match="no usable footage"):
            ExecutionEngine._guard_zero_footage(
                {
                    "summary": {"total": 0, "skipped": 0},
                    "scenario_clips": [],
                    "raw": _nested_report_with_specs(2, video_path=None),
                }
            )

    def test_gate_ignores_an_all_skipped_raw_spec_tree(self):
        """The recursive raw count must not resurrect the false positive the
        summary's skipped-exemption exists to prevent."""
        ExecutionEngine._guard_zero_footage(
            {
                "summary": {"total": 3, "skipped": 3},
                "scenario_clips": [],
                "raw": _nested_report_with_specs(3, video_path=None, status="skipped"),
            }
        )

    def test_expect_recording_false_skips_the_gate(self):
        ExecutionEngine._guard_zero_footage(
            {"summary": {"total": 4, "skipped": 0}, "scenario_clips": []},
            timed_out=True,
            expect_recording=False,
        )

    def test_all_skipped_suite_completes_normally(self):
        """A suite that declared tests but skipped every one has no footage to
        produce; failing it would be a false alarm."""
        ExecutionEngine._guard_zero_footage(
            {"summary": {"total": 3, "skipped": 3}, "scenario_clips": []}
        )

    def test_message_names_declared_and_persisted_counts(self):
        with pytest.raises(RuntimeError) as excinfo:
            ExecutionEngine._guard_zero_footage(
                {"summary": {"total": 4, "skipped": 1}, "scenario_clips": []}
            )
        msg = str(excinfo.value)
        assert "3 recordable test(s)" in msg
        assert "0 scenario clips" in msg
        assert "no usable footage" in msg

    def test_any_persisted_clip_passes_the_gate(self):
        ExecutionEngine._guard_zero_footage(
            {"summary": {"total": 4, "skipped": 0}, "scenario_clips": [{"clip_key": "clip-1"}]}
        )


class TestPersistArtifacts:
    """_persist_artifacts — streams the report tree into the shared object store
    and publishes stable object-key-backed URLs (multi-replica safety)."""

    async def test_no_report_dir(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        engine = ExecutionEngine()
        engine._storage = make_local_storage(tmp_path / "store")
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        persist_dir = tmp_path / "persist"

        result = await engine._persist_artifacts(
            package_dir,
            persist_dir,
            "/api/test-runner/persist/artifacts",
            raw={},
        )
        # Only the backend marker when there is no report and no clips.
        assert result["artifact_paths"] == {"storage_backend": "local"}
        assert result["report_url"] == ""
        assert result["report_path"] == ""
        assert result["scenario_clips"] == []

    async def test_stores_report_and_publishes_url(self, tmp_path: Path):
        from tests.conftest import make_local_storage

        engine = ExecutionEngine()
        store = make_local_storage(tmp_path / "store")
        engine._storage = store
        package_dir = tmp_path / "pkg"
        (package_dir / "playwright-report").mkdir(parents=True)
        (package_dir / "playwright-report" / "index.html").write_text("<html>report</html>")
        persist_dir = tmp_path / "job-1"

        result = await engine._persist_artifacts(
            package_dir,
            persist_dir,
            "/api/test-runner/job-1/artifacts",
            raw={},
        )
        assert (
            result["report_url"] == "/api/test-runner/job-1/artifacts/playwright-report/index.html"
        )
        assert result["artifact_paths"]["has_report"] is True
        assert result["artifact_paths"]["report_index"] == "playwright-report/index.html"
        assert result["report_path"] == "playwright-report/index.html"
        # The report lives in the shared store, keyed under the job prefix.
        assert (
            await store.retrieve("job-1", "playwright-report/index.html") == b"<html>report</html>"
        )

    async def test_report_dir_without_index(self, tmp_path: Path):
        """If playwright-report exists but no index.html, URL is empty but the
        report assets are still stored."""
        from tests.conftest import make_local_storage

        engine = ExecutionEngine()
        store = make_local_storage(tmp_path / "store")
        engine._storage = store
        package_dir = tmp_path / "pkg"
        (package_dir / "playwright-report").mkdir(parents=True)
        (package_dir / "playwright-report" / "style.css").write_text("body{}")
        persist_dir = tmp_path / "job-1"

        result = await engine._persist_artifacts(
            package_dir,
            persist_dir,
            "/api/test-runner/job-1/artifacts",
            raw={},
        )
        assert result["report_url"] == ""
        assert result["report_path"] == ""
        assert result["artifact_paths"]["has_report"] is True
        # The asset is still uploaded to the shared store.
        assert await store.retrieve("job-1", "playwright-report/style.css") == b"body{}"


class TestClipKey:
    """Scenario-clip join-key derivation."""

    def test_prefers_clip_tag(self):
        spec = {"title": "Successful login", "tags": ["@smoke", "@clip-3"]}
        assert _clip_key_for_spec(spec, "Login") == "clip-3"

    def test_clip_tag_without_at_prefix(self):
        spec = {"title": "x", "tags": ["clip-7"]}
        assert _clip_key_for_spec(spec, "F") == "clip-7"

    def test_clip_tag_in_title(self):
        spec = {"title": "Successful login @clip-2", "tags": []}
        assert _clip_key_for_spec(spec, "Login") == "clip-2"

    def test_slug_fallback(self):
        spec = {"title": "Successful Login!", "tags": ["@smoke"]}
        assert _clip_key_for_spec(spec, "User Login") == "user-login__successful-login"


class TestSummarizeTestRun:
    """Attempt-level status roll-up for one Playwright test.

    A wall-timeout SALVAGE stamps synthesized attempts ``"timedOut"`` to flag a
    force-killed (blank) run. Those must NOT roll up to ``passed`` or the job is
    wrongly marked ``completed`` with pass_rate=100%.
    """

    def test_all_passed_stays_passed(self):
        results: list[dict[str, Any]] = [{"status": "passed", "duration": 1000}]
        status, error, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "passed"
        assert error == ""

    def test_timed_out_is_failure_not_passed(self):
        results: list[dict[str, Any]] = [{"status": "timedOut", "duration": 500}]
        status, _, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "failed"

    def test_interrupted_is_failure_not_passed(self):
        results: list[dict[str, Any]] = [{"status": "interrupted", "duration": 500}]
        status, _, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "failed"

    def test_timed_out_then_passed_is_flaky(self):
        results: list[dict[str, Any]] = [
            {"status": "timedOut", "duration": 500},
            {"status": "passed", "duration": 1000},
        ]
        status, _, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "flaky"


class TestCollectClipSpecs:
    """Walking the JSON reporter for one video clip per scenario."""

    def _spec(self, title, tag, video_path, status="passed", annotations=None):
        test: dict = {
            "results": [
                {
                    "status": status,
                    "duration": 1000,
                    "attachments": [{"name": "video", "path": str(video_path)}],
                }
            ]
        }
        if annotations is not None:
            test["annotations"] = annotations
        return {
            "title": title,
            "tags": [tag] if tag else [],
            "tests": [test],
        }

    def test_soft_fail_annotation_demotes_clip_status(self, tmp_path: Path):
        # A soft-failed selector keeps the clip non-blank but Playwright still
        # reports ``passed``; the clip must be marked not-passed.
        v = tmp_path / "a.webm"
        v.write_bytes(b"v")
        raw = {
            "suites": [
                {
                    "title": "Login",
                    "specs": [
                        self._spec(
                            "soft",
                            "@clip-1",
                            v,
                            status="passed",
                            annotations=[{"type": "soft-fail", "description": '{"selector":"#x"}'}],
                        ),
                    ],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw)
        assert specs[0]["status"] == "failed"

    def test_clean_clip_status_unchanged(self, tmp_path: Path):
        # Regression guard: no annotations ⇒ passed stays passed.
        v = tmp_path / "a.webm"
        v.write_bytes(b"v")
        raw = {
            "suites": [
                {
                    "title": "Login",
                    "specs": [self._spec("clean", "@clip-1", v, status="passed")],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw)
        assert specs[0]["status"] == "passed"

    def test_one_clip_per_scenario_first_wins(self, tmp_path: Path):
        v1 = tmp_path / "a.webm"
        v1.write_bytes(b"v1")
        v2 = tmp_path / "b.webm"
        v2.write_bytes(b"v2")
        raw = {
            "suites": [
                {
                    "title": "Login",
                    "specs": [
                        self._spec("first", "@clip-1", v1),
                        # Outline second test reuses clip-1 → must be deduped.
                        self._spec("first-again", "@clip-1", v2),
                        self._spec("second", "@clip-2", v2),
                    ],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw)
        keys = [s["clip_key"] for s in specs]
        assert keys == ["clip-1", "clip-2"]
        assert specs[0]["video_path"] == v1  # first wins

    def test_outline_footage_and_status_same_example(self, tmp_path: Path):
        # Scenario Outline: two examples share ONE @clip tag under a single spec
        # (spec.tests has two entries). Example #1 PASSES with video; example #2
        # FAILS with video. The kept clip's footage and status must describe the
        # SAME example — regression for footage(first) vs status(tests[-1]) skew.
        v_pass = tmp_path / "example1.webm"
        v_pass.write_bytes(b"pass")
        v_fail = tmp_path / "example2.webm"
        v_fail.write_bytes(b"fail")
        spec = {
            "title": "Login outline",
            "tags": ["@clip-1"],
            "tests": [
                {
                    "results": [
                        {
                            "status": "passed",
                            "duration": 1000,
                            "attachments": [{"name": "video", "path": str(v_pass)}],
                        }
                    ]
                },
                {
                    "results": [
                        {
                            "status": "failed",
                            "duration": 1000,
                            "attachments": [{"name": "video", "path": str(v_fail)}],
                        }
                    ]
                },
            ],
        }
        raw = {"suites": [{"title": "Login", "specs": [spec], "suites": []}]}
        specs = ExecutionEngine()._collect_clip_specs(raw)
        assert len(specs) == 1
        # Footage is example #1's (first test with a video); status must match it.
        assert specs[0]["video_path"] == v_pass
        assert specs[0]["status"] == "passed"

    def test_skips_specs_without_video(self, tmp_path: Path):
        raw = {
            "suites": [
                {
                    "title": "F",
                    "specs": [
                        {"title": "no-video", "tags": ["@clip-1"], "tests": [{"results": []}]}
                    ],
                    "suites": [],
                }
            ]
        }
        assert ExecutionEngine()._collect_clip_specs(raw) == []

    def test_paint_lead_annotation_is_carried_as_seconds(self, tmp_path: Path):
        # bdd_playwright_service's nav hook records how long, in ms, the clip
        # sat on an unpainted about:blank before its first real frame — the
        # blank LEAD, the mirror image of the failed-scenario tail this same
        # annotation channel (test.info().annotations.push) already carries for
        # soft-fail. _collect_clip_specs must surface it in SECONDS so the
        # transcode step can act on it.
        v = tmp_path / "a.webm"
        v.write_bytes(b"v")
        raw = {
            "suites": [
                {
                    "title": "Login",
                    "specs": [
                        self._spec(
                            "opens blank",
                            "@clip-1",
                            v,
                            status="passed",
                            annotations=[{"type": "paint-lead", "description": "2280"}],
                        ),
                    ],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw)
        assert specs[0]["lead_trim_seconds"] == pytest.approx(2.28)

    def test_no_paint_lead_annotation_yields_zero(self, tmp_path: Path):
        # Regression guard: an older package (predating the hook) or a
        # non-recording stream must not have to opt out of a trim explicitly.
        v = tmp_path / "a.webm"
        v.write_bytes(b"v")
        raw = {
            "suites": [
                {
                    "title": "Login",
                    "specs": [self._spec("clean", "@clip-1", v, status="passed")],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw)
        assert specs[0]["lead_trim_seconds"] == 0.0


class TestPaintLeadSeconds:
    """`_paint_lead_seconds` reads the `paint-lead` annotation bdd's nav hook
    pushes. Malformed/absent input must degrade to "no trim" (0.0), never to a
    guess or a raise — a clip that cannot prove its lead publishes untrimmed."""

    def test_reads_a_well_formed_annotation(self) -> None:
        test = {"annotations": [{"type": "paint-lead", "description": "430"}]}
        assert ExecutionEngine._paint_lead_seconds(test) == pytest.approx(0.43)

    def test_no_annotations_key_is_zero(self) -> None:
        assert ExecutionEngine._paint_lead_seconds({}) == 0.0

    def test_annotations_not_a_list_is_zero(self) -> None:
        assert ExecutionEngine._paint_lead_seconds({"annotations": "oops"}) == 0.0

    def test_unrelated_annotation_type_is_zero(self) -> None:
        test = {"annotations": [{"type": "soft-fail", "description": "1234"}]}
        assert ExecutionEngine._paint_lead_seconds(test) == 0.0

    def test_non_numeric_description_is_zero_not_a_raise(self) -> None:
        test = {"annotations": [{"type": "paint-lead", "description": "not-a-number"}]}
        assert ExecutionEngine._paint_lead_seconds(test) == 0.0

    def test_negative_description_is_zero(self) -> None:
        # Should never happen (Date.now() deltas cannot go negative) but a
        # malformed/adversarial value must not resolve to a negative trim.
        test = {"annotations": [{"type": "paint-lead", "description": "-50"}]}
        assert ExecutionEngine._paint_lead_seconds(test) == 0.0

    def test_missing_description_is_zero(self) -> None:
        test = {"annotations": [{"type": "paint-lead"}]}
        assert ExecutionEngine._paint_lead_seconds(test) == 0.0


class TestResolveLeadTrimSeconds:
    """Clamping bounds for the blank-lead trim (see the constants and
    docstring next to `_resolve_lead_trim_seconds` in execution_engine.py)."""

    def test_below_the_floor_trims_nothing(self) -> None:
        assert _resolve_lead_trim_seconds(0.05, duration_seconds=30.0) == 0.0

    def test_typical_measured_lead_passes_through(self) -> None:
        # kb3 acceptance run measured ~2.28s of blank lead on a ~37s clip.
        assert _resolve_lead_trim_seconds(2.28, duration_seconds=37.3) == pytest.approx(2.28)

    def test_capped_at_the_max_ceiling(self) -> None:
        assert _resolve_lead_trim_seconds(30.0, duration_seconds=60.0) == pytest.approx(6.0)

    def test_never_leaves_less_than_min_keep(self) -> None:
        # A 3s clip with a "measured" 2.5s lead must keep at least 1.0s, not be
        # trimmed to near-nothing.
        assert _resolve_lead_trim_seconds(2.5, duration_seconds=3.0) == pytest.approx(2.0)

    def test_a_too_short_clip_trims_nothing(self) -> None:
        # duration - MIN_KEEP goes negative -> clamps to 0 -> below the floor.
        assert _resolve_lead_trim_seconds(2.0, duration_seconds=0.5) == 0.0

    def test_unknown_duration_trims_nothing(self) -> None:
        # duration_seconds < 0 is the sentinel for "ffprobe could not tell us" —
        # trimming without knowing how much footage exists risks emptying the
        # clip, so this must refuse rather than guess.
        assert _resolve_lead_trim_seconds(2.0, duration_seconds=-1.0) == 0.0


class TestSubprocessTimeoutBestEffort:
    """A hung ffmpeg/ffprobe must degrade ONE clip (kill + documented fallback),
    not orphan the process nor abort the whole harvest by propagating
    TimeoutError up through _persist_scenario_clips → execute."""

    class _HangingProc:
        def __init__(self) -> None:
            self.returncode: int | None = None
            self.killed = False

        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(3600)  # would hang past the timeout
            return b"", b""

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

        async def wait(self) -> int:
            return self.returncode or 0

    @staticmethod
    def _patch(monkeypatch, proc: TestSubprocessTimeoutBestEffort._HangingProc) -> None:
        async def _fake_create(*_a: Any, **_k: Any) -> Any:
            return proc

        async def _fake_wait_for(coro: Any, timeout: float) -> Any:
            coro.close()  # never awaited → suppress un-awaited-coroutine warning
            raise TimeoutError

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)
        monkeypatch.setattr("asyncio.wait_for", _fake_wait_for)

    async def test_ffprobe_meta_timeout_kills_and_returns_zeroed(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        proc = self._HangingProc()
        self._patch(monkeypatch, proc)
        media = tmp_path / "clip.webm"
        media.write_bytes(b"x")
        result = await _ffprobe_meta(media)
        # documented zeroed fallback, not raise; probe_ok=False marks it an
        # ENVIRONMENT fault so the degenerate-clip guard does not treat a hung
        # ffprobe as proof the footage is a dud.
        assert result == (0.0, 0, 0, False, False)
        assert proc.killed is True  # hung process reaped, not orphaned

    async def test_convert_to_mp4_timeout_kills_and_returns_false(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        proc = self._HangingProc()
        self._patch(monkeypatch, proc)
        src = tmp_path / "in.webm"
        src.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        result = await _convert_to_mp4(src, dst)
        assert result is False  # documented fallback → caller keeps raw clip
        assert proc.killed is True  # hung process reaped, not orphaned


class TestProbeFailuresAreLoud:
    """Every degraded-clip path must LOG. These helpers are best-effort by
    design (they zero the metadata rather than raise), which means a silent
    return is indistinguishable downstream from a genuinely empty clip — the
    exact condition that hid a missing ffmpeg in the image for two days."""

    class _Proc:
        def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
            self.returncode = returncode
            self._out = stdout
            self._err = stderr

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._out, self._err

    @staticmethod
    def _patch(monkeypatch, proc: TestProbeFailuresAreLoud._Proc) -> None:
        async def _fake_create(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)

    async def test_ffprobe_nonzero_returncode_logs_path_rc_and_stderr(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        self._patch(monkeypatch, self._Proc(1, stderr=b"moov atom not found"))
        media = tmp_path / "clip.webm"
        media.write_bytes(b"x")
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            result = await _ffprobe_meta(media)
        # 5-tuple: trailing probe_ok=False — the probe RAN but failed, so its
        # zeroes are not evidence the clip is a dud. Best-effort contract is
        # otherwise unchanged.
        assert result == (0.0, 0, 0, False, False)
        text = caplog.text
        assert "ffprobe failed" in text
        assert "clip.webm" in text  # names the clip
        assert "rc=1" in text  # names the returncode
        assert "moov atom not found" in text  # surfaces ffprobe's own diagnosis

    async def test_ffprobe_unparseable_json_logs(self, tmp_path: Path, monkeypatch, caplog) -> None:
        self._patch(monkeypatch, self._Proc(0, stdout=b"not json at all"))
        media = tmp_path / "clip.webm"
        media.write_bytes(b"x")
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            result = await _ffprobe_meta(media)
        # probe_ok=False — unparseable JSON means ffprobe told us nothing.
        assert result == (0.0, 0, 0, False, False)
        assert "unparseable JSON" in caplog.text
        assert "clip.webm" in caplog.text

    async def test_convert_rc_zero_but_missing_output_is_failure(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        self._patch(monkeypatch, self._Proc(0))
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"  # never created
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            assert await _convert_to_mp4(src_file, dst) is False
        assert "no usable output" in caplog.text

    async def test_convert_rc_zero_but_empty_output_is_failure(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """rc==0 with a 0-byte MP4 must NOT be published — falling back to the
        raw WebM at least leaves decodable pixels for the downstream QA."""
        self._patch(monkeypatch, self._Proc(0))
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"")  # header-only/truncated transcode
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            assert await _convert_to_mp4(src_file, dst) is False
        assert "no usable output" in caplog.text

    async def test_convert_rc_zero_with_real_bytes_succeeds(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The happy path must stay unchanged by the new size guard."""
        self._patch(monkeypatch, self._Proc(0))
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")
        assert await _convert_to_mp4(src_file, dst) is True


class TestSalvageRawFromDisk:
    """Footage salvage when a run is killed before the JSON reporter flushes."""

    def test_prefers_results_json_when_present(self, tmp_path: Path):
        pkg = tmp_path
        flushed = {"suites": [{"title": "real", "specs": [], "suites": []}]}
        (pkg / "results.json").write_text(json.dumps(flushed))
        out = ExecutionEngine()._salvage_raw_from_disk(pkg)
        assert out == flushed

    def test_scans_webm_when_no_results_json(self, tmp_path: Path):
        pkg = tmp_path
        # Two scenarios landed video before the kill; one dir name keeps the tag.
        d1 = pkg / "test-results" / "login-clip-3-chromium"
        d1.mkdir(parents=True)
        (d1 / "video.webm").write_bytes(b"v")
        d2 = pkg / "test-results" / "checkout-no-tag-chromium"
        d2.mkdir(parents=True)
        (d2 / "video.webm").write_bytes(b"v")

        raw = ExecutionEngine()._salvage_raw_from_disk(pkg)
        # Routes through the normal harvest path → clips are collected.
        clips = ExecutionEngine()._collect_clip_specs(raw)
        keys = sorted(c["clip_key"] for c in clips)
        # The tagged dir joins cleanly; the untagged one falls back to a slug.
        assert "clip-3" in keys
        assert any("checkout-no-tag" in k for k in keys)
        assert len(clips) == 2

    def test_empty_when_no_footage(self, tmp_path: Path):
        raw = ExecutionEngine()._salvage_raw_from_disk(tmp_path)
        assert ExecutionEngine()._collect_clip_specs(raw) == []

    def test_corrupt_results_json_falls_back_to_scan(self, tmp_path: Path):
        pkg = tmp_path
        (pkg / "results.json").write_text("{not json")
        d = pkg / "test-results" / "s-clip-1-chromium"
        d.mkdir(parents=True)
        (d / "video.webm").write_bytes(b"v")
        clips = ExecutionEngine()._collect_clip_specs(ExecutionEngine()._salvage_raw_from_disk(pkg))
        assert [c["clip_key"] for c in clips] == ["clip-1"]

    def test_zero_footage_salvage_logs_error_with_cause(self, caplog):
        # An empty-but-well-formed salvage (browser never launched) must surface
        # at ERROR with the actionable PLAYWRIGHT_BROWSERS_PATH/bddgen hint so it
        # can't masquerade as a healthy run that merely captured nothing.
        empty = {"suites": [{"title": "salvaged", "specs": [], "suites": []}]}
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            n = ExecutionEngine._log_salvage_outcome(empty)
        assert n == 0
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert errors and "never launched" in errors[0].getMessage()

    def test_counts_specs_in_nested_suites(self, caplog):
        """D1: `_salvage_raw_from_disk` prefers a reporter-flushed results.json,
        and Playwright NESTS specs inside the describe block bddgen emits per
        `Feature:`. A top-level-only count read a partial report holding 3 real
        clips as 0 — and the old guard then hard-failed a run with good footage."""
        nested = {
            "suites": [
                {
                    "title": "Login.feature",
                    "specs": [],
                    "suites": [
                        {
                            "title": "Login",
                            "specs": [{"title": "a"}, {"title": "b"}],
                            "suites": [
                                {"title": "deeper", "specs": [{"title": "c"}], "suites": []}
                            ],
                        }
                    ],
                }
            ]
        }
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            n = ExecutionEngine._log_salvage_outcome(nested)
        assert n == 3
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]

    def test_nonzero_footage_salvage_logs_warning_not_error(self, caplog):
        salvaged = {"suites": [{"title": "salvaged", "specs": [{"x": 1}, {"y": 2}], "suites": []}]}
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            n = ExecutionEngine._log_salvage_outcome(salvaged)
        assert n == 2
        assert not [r for r in caplog.records if r.levelno == logging.ERROR]
        assert [r for r in caplog.records if r.levelno == logging.WARNING]


class TestWaitForReportOrExit:
    """Finishing as soon as the report is written, not when npx finally exits."""

    async def test_returns_when_process_exits(self, tmp_path: Path):
        class _Proc:
            async def wait(self):
                return 0

        # Exits immediately → returns fast even with no report on disk.
        await ExecutionEngine()._wait_for_report_or_exit(
            cast(asyncio.subprocess.Process, _Proc()),
            tmp_path / "results.json",
            5,
            poll_interval=0.02,
        )

    async def test_returns_when_report_stable_even_if_proc_hangs(self, tmp_path: Path):
        rf = tmp_path / "results.json"
        rf.write_text('{"suites":[]}')  # report present + stable

        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)  # npx hung on cleanup

        await asyncio.wait_for(
            ExecutionEngine()._wait_for_report_or_exit(
                cast(asyncio.subprocess.Process, _Proc()), rf, 5, poll_interval=0.02
            ),
            timeout=3,
        )

    async def test_raises_at_wall_when_no_report(self, tmp_path: Path):
        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)

        with pytest.raises(TimeoutError):
            await ExecutionEngine()._wait_for_report_or_exit(
                cast(asyncio.subprocess.Process, _Proc()),
                tmp_path / "results.json",
                0.1,
                poll_interval=0.02,
            )

    async def test_drains_video_before_returning(self, tmp_path: Path):
        """Report is stable immediately, but a clip is still being muxed — the
        wait must not return until the .webm size has settled."""
        rf = tmp_path / "results.json"
        rf.write_text('{"suites":[]}')
        video_dir = tmp_path / "test-results" / "scenario"
        video_dir.mkdir(parents=True)
        webm = video_dir / "video.webm"
        webm.write_bytes(b"a")

        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)  # npx hung on cleanup, as usual

        # ORDERING is the whole assertion. An earlier version of this test ended
        # on `webm.read_bytes() == b"ab"`, which the `finally: await grow_task`
        # below makes trivially true whether or not the drain waited — it passed
        # with `_drain_video_artifacts` deleted outright. Timestamps are what
        # distinguish "returned after the muxer settled" from "returned
        # immediately and the muxer happened to finish first".
        grew_at: list[float] = []

        async def _grow():
            # A muxer writes CONTINUOUSLY and then stops; it does not go quiet and
            # resume. That distinction matters: the drain's contract is "settled ==
            # size unchanged across 2 consecutive polls", so a file merely idle for
            # one gap is legitimately settled and waiting on it would be asserting
            # something the design never promised. Writing every ~10ms for ~250ms
            # keeps the size genuinely moving across many poll windows, which is
            # what the drain exists to sit through.
            for n in range(2, 27):
                await asyncio.sleep(0.01)
                webm.write_bytes(b"a" * n)
            grew_at.append(asyncio.get_running_loop().time())

        grow_task = asyncio.create_task(_grow())
        try:
            outcome = await asyncio.wait_for(
                ExecutionEngine()._wait_for_report_or_exit(
                    cast(asyncio.subprocess.Process, _Proc()),
                    rf,
                    5,
                    poll_interval=0.02,
                    video_drain_timeout_seconds=5,
                ),
                timeout=3,
            )
            returned_at = asyncio.get_running_loop().time()
        finally:
            await grow_task
        assert outcome.video_drain_timed_out is False
        assert grew_at, "the muxer never wrote — test setup is broken, not the drain"
        assert returned_at > grew_at[0], (
            "returned before the muxer's last write: the drain did not wait for the "
            "clip to settle, so a SIGKILL here would truncate it"
        )
        assert webm.read_bytes() == b"a" * 26

    async def test_video_drain_deadline_is_signalled(self, tmp_path: Path):
        """A clip that never stops changing size must not hang the caller
        forever — the drain deadline fires and is reported, not swallowed."""
        rf = tmp_path / "results.json"
        rf.write_text('{"suites":[]}')
        video_dir = tmp_path / "test-results" / "scenario"
        video_dir.mkdir(parents=True)
        webm = video_dir / "video.webm"
        webm.write_bytes(b"a")

        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)

        async def _keep_growing():
            n = 1
            while True:
                await asyncio.sleep(0.02)
                n += 1
                webm.write_bytes(b"a" * n)

        grow_task = asyncio.create_task(_keep_growing())
        try:
            outcome = await asyncio.wait_for(
                ExecutionEngine()._wait_for_report_or_exit(
                    cast(asyncio.subprocess.Process, _Proc()),
                    rf,
                    5,
                    poll_interval=0.02,
                    video_drain_timeout_seconds=0.1,
                ),
                timeout=3,
            )
        finally:
            grow_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await grow_task
        assert outcome.video_drain_timed_out is True

    async def test_no_video_files_skips_drain_entirely(self, tmp_path: Path):
        """Non-recording runs must not pay any drain latency: results.json
        stable + zero .webm files returns as fast as the old, drain-less
        behaviour did."""
        rf = tmp_path / "results.json"
        rf.write_text('{"suites":[]}')

        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)

        outcome = await asyncio.wait_for(
            ExecutionEngine()._wait_for_report_or_exit(
                cast(asyncio.subprocess.Process, _Proc()),
                rf,
                5,
                poll_interval=0.02,
                video_drain_timeout_seconds=30,
            ),
            timeout=0.5,  # would blow past this if drain didn't short-circuit
        )
        assert outcome.video_drain_timed_out is False


class TestDrainVideoArtifacts:
    """Unit coverage of the drain helper in isolation from the report wait."""

    async def test_no_videos_returns_immediately(self, tmp_path: Path):
        class _Proc:
            async def wait(self):
                await asyncio.sleep(60)

        outcome = await asyncio.wait_for(
            ExecutionEngine._drain_video_artifacts(
                cast(asyncio.subprocess.Process, _Proc()), tmp_path, 5, poll_interval=0.02
            ),
            timeout=0.5,
        )
        assert outcome.video_drain_timed_out is False

    async def test_process_exits_during_drain_returns_immediately(self, tmp_path: Path):
        video_dir = tmp_path / "test-results" / "scenario"
        video_dir.mkdir(parents=True)
        (video_dir / "video.webm").write_bytes(b"still writing")

        class _Proc:
            async def wait(self):
                return 0  # exits on its own: the clean path

        outcome = await asyncio.wait_for(
            ExecutionEngine._drain_video_artifacts(
                cast(asyncio.subprocess.Process, _Proc()), tmp_path, 5, poll_interval=0.02
            ),
            timeout=0.5,
        )
        assert outcome.video_drain_timed_out is False


class TestTerminateProcessGroup:
    """Killing the npx→node→browser→ffmpeg tree as a unit."""

    class _Proc:
        __test__ = False
        returncode = 0
        pid = 4242

        async def wait(self):
            return 0

    async def test_noop_when_no_pgid(self, monkeypatch):
        """No captured pgid → nothing to signal; just reap, no killpg."""
        import src.services.execution_engine as mod

        sent: list[int] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda pid: 1)
        monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: sent.append(sig))
        await ExecutionEngine()._terminate_process_group(
            cast(asyncio.subprocess.Process, self._Proc()), None
        )
        assert sent == []

    async def test_escalates_term_then_kill_to_group(self, monkeypatch):
        """Always escalates SIGTERM→SIGKILL to the GROUP: the leader exiting does
        not mean the browser/ffmpeg children did (the B7 leak)."""
        import src.services.execution_engine as mod

        sent: list[int] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda pid: 0)  # FTR's own group = 0
        monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: sent.append(sig))
        await ExecutionEngine()._terminate_process_group(
            cast(asyncio.subprocess.Process, self._Proc()), 4242
        )
        assert sent == [mod.signal.SIGTERM, mod.signal.SIGKILL]

    async def test_never_signals_own_group(self, monkeypatch):
        """A pgid equal to FTR's own process group is refused — never self-kill."""
        import src.services.execution_engine as mod

        sent: list[int] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda pid: 777)
        monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: sent.append(sig))
        await ExecutionEngine()._terminate_process_group(
            cast(asyncio.subprocess.Process, self._Proc()), 777
        )
        assert sent == []

    class _CoercingPidProc:
        """A proc whose ``.pid`` coerces to 1 via ``__index__`` (exactly as a
        ``MagicMock`` attribute does) but is not an ``int`` instance — the mock
        shape behind the broadcast footgun."""

        __test__ = False
        returncode = None

        class _Pid:
            def __index__(self) -> int:
                return 1

        pid = _Pid()

        async def wait(self):
            return 0

    def test_capture_pgid_guards_coercible_mock_pid(self, monkeypatch):
        """A proc.pid that is not a real int (coerces to 1) yields None and never
        reaches os.getpgid — otherwise getpgid(1) + killpg would broadcast
        kill(-1) across the whole process tree (coding-security #80)."""
        import src.services.execution_engine as mod

        called: list[Any] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda pid: called.append(pid) or 1)
        result = ExecutionEngine._capture_pgid(
            cast(asyncio.subprocess.Process, self._CoercingPidProc())
        )
        assert result is None
        assert called == []  # getpgid never called with the coerced pid

    async def test_refuses_pgid_one_broadcast(self, monkeypatch):
        """pgid==1 (init's group) is never signalled: killpg(1) == kill(-1), a
        broadcast to the whole process tree."""
        import src.services.execution_engine as mod

        sent: list[int] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda pid: 0)
        monkeypatch.setattr(mod.os, "killpg", lambda pgid, sig: sent.append(sig))
        await ExecutionEngine()._terminate_process_group(
            cast(asyncio.subprocess.Process, self._Proc()), 1
        )
        assert sent == []


class TestPersistScenarioClips:
    """End-to-end clip persistence (convert disabled → raw copy, no ffmpeg need)."""

    @staticmethod
    def _stub_ffprobe(monkeypatch, meta: tuple[float, int, int, bool]) -> None:
        """Force ffprobe metadata. Real ffprobe on synthetic bytes returns the
        zeroed fallback, which the degenerate-clip guard now (correctly) drops —
        so a persistence test must state the metadata it means to exercise."""

        async def _fake(_path: Path) -> tuple[float, int, int, bool, bool]:
            return (*meta, True)

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake)

    async def test_persists_and_keys_clips(self, tmp_path: Path, monkeypatch):
        from src.config import get_settings
        from tests.conftest import make_local_storage

        self._stub_ffprobe(monkeypatch, (1200.0, 1280, 720, False))
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            src = tmp_path / "rec.webm"
            src.write_bytes(b"fake-webm-bytes")
            raw = {
                "suites": [
                    {
                        "title": "Login",
                        "specs": [
                            {
                                "title": "Successful login",
                                "tags": ["@clip-1"],
                                "tests": [
                                    {
                                        "results": [
                                            {
                                                "status": "passed",
                                                "duration": 1200,
                                                "attachments": [
                                                    {"name": "video", "path": str(src)}
                                                ],
                                            }
                                        ]
                                    }
                                ],
                            }
                        ],
                        "suites": [],
                    }
                ]
            }
            persist_dir = tmp_path / "job-xyz"
            engine = ExecutionEngine()
            store = make_local_storage(tmp_path / "store")
            engine._storage = store
            clips, clip_files, _dropped, _tdrop = await engine._persist_scenario_clips(
                raw, tmp_path, persist_dir
            )
            assert len(clips) == 1
            clip = clips[0]
            assert clip["clip_key"] == "clip-1"
            assert clip["scenario"] == "Successful login"
            assert clip["video_url"] == "/api/test-runner/job-xyz/clips/clip-1/video"
            assert clip["size_bytes"] == len(b"fake-webm-bytes")
            # The raw recording is mapped for the route and STREAMED into the
            # shared store under the job prefix (not left on pod-local disk).
            assert clip_files == {"clip-1": "clip-1.webm"}
            assert await store.retrieve("job-xyz", "clips/clip-1.webm") == b"fake-webm-bytes"
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_warns_when_clip_has_bytes_but_no_decodable_stream(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """`size_bytes > 0` with `duration_ms == 0` / `width == 0` is the exact
        signature of a missing ffprobe or a corrupt recording. It used to be
        published in total silence; the clip must now be named in the log."""
        from src.config import get_settings
        from tests.conftest import make_local_storage

        async def _zeroed_probe(_path: Path) -> tuple[float, int, int, bool, bool]:
            # probe_ok=False — the probe could not run. That is what keeps this
            # clip PUBLISHED past the degenerate guard: "ffprobe told us nothing"
            # is an environment fault, not evidence the footage is a dud.
            return 0.0, 0, 0, False, False

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _zeroed_probe)
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            src_file = tmp_path / "rec.webm"
            src_file.write_bytes(b"fake-webm-bytes")
            raw = {
                "suites": [
                    {
                        "title": "Login",
                        "specs": [
                            {
                                "title": "Successful login",
                                "tags": ["@clip-7"],
                                "tests": [
                                    {
                                        "results": [
                                            {
                                                "status": "passed",
                                                "duration": 1200,
                                                "attachments": [
                                                    {"name": "video", "path": str(src_file)}
                                                ],
                                            }
                                        ]
                                    }
                                ],
                            }
                        ],
                        "suites": [],
                    }
                ]
            }
            engine = ExecutionEngine()
            engine._storage = make_local_storage(tmp_path / "store")
            with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
                clips, _, _dropped, _tdrop = await engine._persist_scenario_clips(
                    raw, tmp_path, tmp_path / "job-zero"
                )
            # Behaviour is UNCHANGED — the clip is still published (a sibling PR
            # owns the gating decision); only the silence is fixed.
            assert len(clips) == 1
            assert clips[0]["size_bytes"] == len(b"fake-webm-bytes")
            assert "no decodable stream" in caplog.text
            assert "clip-7" in caplog.text
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    @staticmethod
    def _raw_for(sources: dict[str, Path]) -> dict[str, Any]:
        """Playwright-JSON with one passing spec per ``{clip_tag: video_path}``."""
        return {
            "suites": [
                {
                    "title": "Login",
                    "specs": [
                        {
                            "title": f"scenario {tag}",
                            "tags": [f"@{tag}"],
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "passed",
                                            "duration": 1200,
                                            "attachments": [{"name": "video", "path": str(path)}],
                                        }
                                    ]
                                }
                            ],
                        }
                        for tag, path in sources.items()
                    ],
                    "suites": [],
                }
            ]
        }

    async def _run_with_dud(
        self,
        tmp_path: Path,
        monkeypatch,
        *,
        dud_bytes: bytes,
        dud_meta: tuple[float, int, int, bool],
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[str], list[str]]:
        from src.config import get_settings
        from tests.conftest import make_local_storage

        good = tmp_path / "good.webm"
        good.write_bytes(b"fake-webm-bytes")
        dud = tmp_path / "dud.webm"
        dud.write_bytes(dud_bytes)

        async def _fake_ffprobe(path: Path) -> tuple[float, int, int, bool, bool]:
            meta = dud_meta if path.name.startswith("clip-2") else (1200.0, 1280, 720, False)
            return (*meta, True)

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe)
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            engine = ExecutionEngine()
            engine._storage = make_local_storage(tmp_path / "store")
            return await engine._persist_scenario_clips(
                self._raw_for({"clip-1": good, "clip-2": dud}),
                tmp_path,
                tmp_path / "job-dud",
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_drops_zero_duration_clip_keeps_good_one(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        # A clip ffprobe reports no duration for is a dud — typically a .webm
        # whose muxer was SIGKILLed mid-write on the salvage path. Attribute it
        # at capture rather than letting the compiler render a black gap.
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips, clip_files, dropped, _tdrop = await self._run_with_dud(
                tmp_path, monkeypatch, dud_bytes=b"partial", dud_meta=(0.0, 0, 0, False)
            )
        assert [c["clip_key"] for c in clips] == ["clip-1"]
        assert clip_files == {"clip-1": "clip-1.webm"}  # dud not published either
        # The drop is RETURNED, not only logged: the caller cannot otherwise tell
        # a short harvest from a clean one (see `_guard_partial_footage`).
        assert dropped == ["clip-2"]
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("clip-2" in r.getMessage() for r in warnings)

    async def test_unprobeable_clips_are_kept_not_silently_dropped(
        self, tmp_path: Path, monkeypatch
    ):
        """An ffprobe that cannot RUN must never empty a healthy harvest.

        `_ffprobe_meta` zeroes its metadata when ffprobe is absent, hangs, exits
        non-zero, or emits unparseable JSON. Keying the degenerate-clip guard on
        `duration_ms == 0` alone would read that environment fault as proof that
        every clip is a dud and silently discard the whole harvest — reintroducing,
        one layer down, exactly the zero-footage-reported-as-success defect this
        guard exists to close. `probe_ok=False` is what keeps the two apart.
        """
        from src.config import get_settings
        from tests.conftest import make_local_storage

        good = tmp_path / "good.webm"
        good.write_bytes(b"fake-webm-bytes")
        other = tmp_path / "other.webm"
        other.write_bytes(b"fake-webm-bytes")

        async def _unprobeable(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 0.0, 0, 0, False, False

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _unprobeable)
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            engine = ExecutionEngine()
            engine._storage = make_local_storage(tmp_path / "store")
            clips, clip_files, _dropped, _tdrop = await engine._persist_scenario_clips(
                self._raw_for({"clip-1": good, "clip-2": other}),
                tmp_path,
                tmp_path / "job-unprobeable",
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True
        assert [c["clip_key"] for c in clips] == ["clip-1", "clip-2"]
        assert set(clip_files) == {"clip-1", "clip-2"}

    async def test_degenerate_mp4_falls_back_to_healthy_raw_webm(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """D5: degeneracy is measured on the POST-conversion artifact, and
        `_convert_to_mp4` returns merely `dst.exists()` — a truncated-input
        transcode can emit a 0-byte mp4 from a healthy raw recording. Publish the
        RAW clip instead of dropping footage the pre-fix code published."""
        from src.config import get_settings
        from tests.conftest import make_local_storage

        src = tmp_path / "rec.webm"
        src.write_bytes(b"fake-webm-bytes")

        async def _fake_convert(_src: Path, dst: Path, *, lead_trim_seconds: float = 0.0) -> bool:
            dst.write_bytes(b"")  # 0-byte mp4, but ffmpeg "succeeded"
            return True

        async def _fake_ffprobe(path: Path) -> tuple[float, int, int, bool, bool]:
            if path.suffix == ".mp4":
                return 0.0, 0, 0, False, True
            return 1200.0, 1280, 720, False, True

        monkeypatch.setattr("src.services.execution_engine._convert_to_mp4", _fake_convert)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe)
        get_settings().execution.convert_clips_to_mp4 = True
        engine = ExecutionEngine()
        store = make_local_storage(tmp_path / "store")
        engine._storage = store
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips, clip_files, _dropped, _tdrop = await engine._persist_scenario_clips(
                self._raw_for({"clip-1": src}), tmp_path, tmp_path / "job-fallback"
            )
        assert [c["clip_key"] for c in clips] == ["clip-1"]
        assert clip_files == {"clip-1": "clip-1.webm"}
        assert clips[0]["size_bytes"] == len(b"fake-webm-bytes")
        assert clips[0]["duration_ms"] == 1200.0
        assert await store.retrieve("job-fallback", "clips/clip-1.webm") == b"fake-webm-bytes"
        assert any("publishing the RAW clip" in r.getMessage() for r in caplog.records)

    async def test_degenerate_mp4_and_degenerate_raw_is_dropped(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """Only when BOTH candidates are degenerate is the clip dropped."""
        from src.config import get_settings
        from tests.conftest import make_local_storage

        src = tmp_path / "rec.webm"
        src.write_bytes(b"truncated")

        async def _fake_convert(_src: Path, dst: Path, *, lead_trim_seconds: float = 0.0) -> bool:
            dst.write_bytes(b"")
            return True

        async def _fake_ffprobe(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 0.0, 0, 0, False, True

        monkeypatch.setattr("src.services.execution_engine._convert_to_mp4", _fake_convert)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe)
        get_settings().execution.convert_clips_to_mp4 = True
        engine = ExecutionEngine()
        engine._storage = make_local_storage(tmp_path / "store")
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips, clip_files, _dropped, _tdrop = await engine._persist_scenario_clips(
                self._raw_for({"clip-1": src}), tmp_path, tmp_path / "job-bothdud"
            )
        assert clips == []
        assert clip_files == {}
        assert _dropped == ["clip-1"]
        assert any("Dropping degenerate scenario clip" in r.getMessage() for r in caplog.records)

    async def test_drops_zero_byte_clip_keeps_good_one(self, tmp_path: Path, monkeypatch, caplog):
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips, clip_files, dropped, _tdrop = await self._run_with_dud(
                tmp_path, monkeypatch, dud_bytes=b"", dud_meta=(1200.0, 1280, 720, False)
            )
        assert [c["clip_key"] for c in clips] == ["clip-1"]
        assert clip_files == {"clip-1": "clip-1.webm"}
        assert dropped == ["clip-2"]
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("clip-2" in r.getMessage() for r in warnings)


class TestUnmeasuredClipOmitsTrackGeometry:
    """A clip ffprobe could not measure must publish NO geometry at all.

    `_is_degenerate` keeps an unprobeable-but-non-empty clip (probe_ok=False is
    an environment fault, not proof of a dud), so `width == height == 0` reaches
    `_normalize_track` on a live path. Guarding only the DENOMINATOR left
    fx = fy = 0.0, so every bbox/cursor coordinate became 0.0 and viewport
    {"w": 0, "h": 0} — confidently-wrong geometry the compiler pins at the
    origin. No test anywhere fed a width=0 probe result WITH a track present,
    which is exactly why it was invisible.
    """

    @staticmethod
    def _raw_with_track(video: Path) -> dict[str, Any]:
        """One passing spec carrying BOTH the video and a w2v-track body attachment."""
        return {
            "suites": [
                {
                    "title": "Login",
                    "specs": [
                        {
                            "title": "Successful login",
                            "tags": ["@clip-3"],
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "passed",
                                            "duration": 1200,
                                            "attachments": [
                                                {"name": "video", "path": str(video)},
                                                _body_attachment(_TRACK),
                                            ],
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                    "suites": [],
                }
            ]
        }

    async def _persist(
        self,
        tmp_path: Path,
        monkeypatch,
        meta: tuple[Any, Any, Any, bool, bool],
        raw: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        from src.config import get_settings
        from tests.conftest import make_local_storage

        video = tmp_path / "rec.webm"
        video.write_bytes(b"fake-webm-bytes")

        async def _fake_ffprobe(_path: Path) -> tuple[Any, Any, Any, bool, bool]:
            return meta

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe)
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            engine = ExecutionEngine()
            engine._storage = make_local_storage(tmp_path / "store")
            clips, _, _dropped, _tdrop = await engine._persist_scenario_clips(
                raw if raw is not None else self._raw_with_track(video),
                tmp_path,
                tmp_path / "job-unmeasured",
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True
        self.last_track_dropped = _tdrop
        return clips

    @staticmethod
    def _assert_geometry_omitted(clip: dict[str, Any]) -> None:
        # Omitted, NOT zeroed: an all-zeros track is indistinguishable from real
        # geometry downstream, so assert the honest absent state on all three.
        geometry = {k: clip[k] for k in ("focus_track", "cursor_track", "viewport")}
        assert geometry == {"focus_track": None, "cursor_track": None, "viewport": None}
        # Specifically NOT the pre-fix shape: every coordinate scaled by fx=fy=0.0
        # and a {"w": 0, "h": 0} viewport.
        assert "0.0" not in json.dumps(geometry)
        assert clip["viewport"] != {"w": 0, "h": 0}

    @pytest.mark.parametrize(
        ("width", "height"),
        [
            (0, 1080),  # ffprobe ran but reported no frame size
            (1920, 0),
            (0, 0),  # the real signature of a missing/hung ffprobe
            (-1920, -1080),  # nonsense dims must degrade, not flip the sign
            # Non-numeric dims are covered at the normalizer level below: they
            # cannot reach here, since ScenarioClip.width/height are `int`.
        ],
    )
    async def test_unmeasured_dims_omit_all_three_tracks(
        self, tmp_path: Path, monkeypatch, caplog, width: Any, height: Any
    ):
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips = await self._persist(
                tmp_path, monkeypatch, (1200.0, width, height, False, False)
            )
        assert len(clips) == 1, "the clip itself must still publish — only geometry is withheld"
        self._assert_geometry_omitted(clips[0])
        assert any(
            "clip-3" in r.getMessage() and "NO measured" in r.getMessage()
            for r in caplog.records
            if r.levelno >= logging.WARNING
        ), "the omission must be diagnosable by clip id"

    async def test_healthy_probe_still_produces_scaled_track(self, tmp_path: Path, monkeypatch):
        """No regression: a measured clip keeps the normal VP-2 scaled geometry."""
        clips = await self._persist(tmp_path, monkeypatch, (1200.0, 2560, 1440, False, True))
        assert len(clips) == 1
        clip = clips[0]
        assert clip["viewport"] == {"w": 2560, "h": 1440}
        # _TRACK was captured at 1280x720 → 2x on both axes.
        assert clip["focus_track"][0]["bbox"] == {"x": 20.0, "y": 40.0, "w": 60.0, "h": 80.0}
        assert clip["cursor_track"][0]["x"] == pytest.approx(20.0)
        assert clip["cursor_track"][0]["y"] == pytest.approx(40.0)

    async def test_unmeasured_dims_report_the_key_to_the_caller(self, tmp_path: Path, monkeypatch):
        """The DISCARD must leave the method as data, not only as a log line.

        The clip publishes with focus_track=None, which downstream is
        byte-identical to a scenario that never touched the page. Only this
        service knows the track existed, so only this service can say so.
        """
        clips = await self._persist(tmp_path, monkeypatch, (1200.0, 0, 0, False, False))
        assert len(clips) == 1
        assert self.last_track_dropped == ["clip-3"]

    async def test_malformed_track_payload_reports_the_key_too(
        self, tmp_path: Path, monkeypatch, caplog
    ):
        """A well-measured clip whose track payload `_normalize_track` rejects.

        `_load_and_normalize_track` is total and swallows its reason, so this
        branch discarded a recorded track in complete silence.
        """
        video = tmp_path / "rec.webm"
        video.write_bytes(b"fake-webm-bytes")
        bad_track = {
            "viewport": {"w": 1280, "h": 720},
            # `float("not-a-number")` -> ValueError inside _normalize_track.
            "focus": [{"t": "not-a-number", "bbox": {"x": 1, "y": 2, "w": 3, "h": 4}}],
            "cursor": [],
        }
        raw = self._raw_with_track(video)
        att = raw["suites"][0]["specs"][0]["tests"][0]["results"][0]["attachments"]
        att[1] = _body_attachment(bad_track)

        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            clips = await self._persist(
                tmp_path, monkeypatch, (1200.0, 1920, 1080, False, True), raw=raw
            )
        assert len(clips) == 1, "the clip still publishes — only its geometry is withheld"
        self._assert_geometry_omitted(clips[0])
        assert self.last_track_dropped == ["clip-3"]
        assert any("could not be normalized" in r.getMessage() for r in caplog.records)

    async def test_healthy_track_reports_no_discard(self, tmp_path: Path, monkeypatch):
        """No false positive: a normalized track must not be reported as discarded."""
        clips = await self._persist(tmp_path, monkeypatch, (1200.0, 2560, 1440, False, True))
        assert clips[0]["cursor_track"]
        assert self.last_track_dropped == []

    @pytest.mark.parametrize(
        ("width", "height"),
        [(0, 1080), (1920, 0), (0, 0), (-1, -1), (None, 1080), (1920, None)],
    )
    def test_normalizer_rejects_unmeasured_recorded_dims(self, width: Any, height: Any):
        assert _normalize_track(_TRACK, width, height) is None
        assert _load_and_normalize_track(_TRACK, width, height) is None


class TestParseResults:
    """Test the Playwright JSON reporter parsing logic."""

    def setup_method(self):
        self.engine = ExecutionEngine()

    def test_empty_suites(self, tmp_path):
        raw = {"suites": []}
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["total"] == 0
        assert result["summary"]["passed"] == 0
        assert result["summary"]["pass_rate"] == 0.0
        assert result["test_results"] == []

    def test_single_passing_test(self, tmp_path):
        raw = {
            "suites": [
                {
                    "title": "Login Feature",
                    "specs": [
                        {
                            "title": "should login successfully",
                            "tests": [{"results": [{"status": "passed", "duration": 1500}]}],
                        }
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["total"] == 1
        assert result["summary"]["passed"] == 1
        assert result["summary"]["pass_rate"] == 100.0
        assert len(result["test_results"]) == 1
        assert result["test_results"][0]["name"] == "should login successfully"
        assert result["test_results"][0]["feature"] == "Login Feature"
        assert result["test_results"][0]["status"] == "passed"
        assert result["test_results"][0]["duration_ms"] == 1500

    def test_soft_fail_annotation_not_counted_passed(self, tmp_path):
        # Two tests both report Playwright status ``passed``; one carries a
        # ``soft-fail`` annotation and must be demoted so pass_rate < 100.
        raw = {
            "suites": [
                {
                    "title": "Tests",
                    "specs": [
                        {
                            "title": "clean pass",
                            "tests": [{"results": [{"status": "passed", "duration": 500}]}],
                        },
                        {
                            "title": "soft failed",
                            "tests": [
                                {
                                    "annotations": [
                                        {"type": "soft-fail", "description": '{"selector":"#x"}'}
                                    ],
                                    "results": [{"status": "passed", "duration": 500}],
                                }
                            ],
                        },
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["total"] == 2
        assert result["summary"]["passed"] == 1
        assert result["summary"]["failed"] == 1
        assert result["summary"]["pass_rate"] == 50.0

    def test_no_annotations_pass_rate_unchanged(self, tmp_path):
        # Regression guard: a clean all-passing report stays at 100%.
        raw = {
            "suites": [
                {
                    "title": "Tests",
                    "specs": [
                        {
                            "title": "a",
                            "tests": [{"results": [{"status": "passed", "duration": 100}]}],
                        },
                        {
                            "title": "b",
                            "tests": [{"results": [{"status": "passed", "duration": 100}]}],
                        },
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["passed"] == 2
        assert result["summary"]["failed"] == 0
        assert result["summary"]["pass_rate"] == 100.0

    def test_mixed_results(self, tmp_path):
        raw = {
            "suites": [
                {
                    "title": "Tests",
                    "specs": [
                        {
                            "title": "passes",
                            "tests": [{"results": [{"status": "passed", "duration": 500}]}],
                        },
                        {
                            "title": "fails",
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "failed",
                                            "duration": 2000,
                                            "error": {"message": "Element not found"},
                                        }
                                    ]
                                }
                            ],
                        },
                        {
                            "title": "skipped",
                            "tests": [{"results": [{"status": "skipped", "duration": 0}]}],
                        },
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["total"] == 3
        assert result["summary"]["passed"] == 1
        assert result["summary"]["failed"] == 1
        assert result["summary"]["skipped"] == 1
        assert result["summary"]["pass_rate"] == 33.3

    def test_nested_suites(self, tmp_path):
        raw = {
            "suites": [
                {
                    "title": "Outer",
                    "specs": [],
                    "suites": [
                        {
                            "title": "Inner",
                            "specs": [
                                {
                                    "title": "nested test",
                                    "tests": [{"results": [{"status": "passed", "duration": 100}]}],
                                }
                            ],
                            "suites": [],
                        }
                    ],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["total"] == 1
        assert result["summary"]["passed"] == 1
        assert result["test_results"][0]["feature"] == "Inner"

    def test_failed_test_with_dict_error(self, tmp_path):
        raw = {
            "suites": [
                {
                    "title": "Suite",
                    "specs": [
                        {
                            "title": "broken test",
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "failed",
                                            "duration": 3000,
                                            "error": {"message": "Timeout waiting for selector"},
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["summary"]["failed"] == 1
        assert "Timeout" in result["test_results"][0]["error"]

    def test_failed_test_with_string_error(self, tmp_path):
        raw = {
            "suites": [
                {
                    "title": "Suite",
                    "specs": [
                        {
                            "title": "broken test",
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "failed",
                                            "duration": 1000,
                                            "error": "plain string error",
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        assert result["test_results"][0]["error"] == "plain string error"

    def test_multiple_results_per_test(self, tmp_path):
        """Test with retries — multiple results per test."""
        raw = {
            "suites": [
                {
                    "title": "Flaky",
                    "specs": [
                        {
                            "title": "flaky test",
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "failed",
                                            "duration": 500,
                                            "error": {"message": "first attempt"},
                                        },
                                        {"status": "passed", "duration": 600},
                                    ]
                                }
                            ],
                        }
                    ],
                    "suites": [],
                }
            ]
        }
        result = self.engine._parse_results(raw, tmp_path)

        # Duration is the DECIDING attempt's own wall time, NOT the sum of all
        # attempts (which reported ~3x the real duration at retries=2).
        assert result["test_results"][0]["duration_ms"] == 600
        # Multiple results with first=failed, last=passed → flaky
        assert result["test_results"][0]["status"] == "flaky"

    def test_report_path_exists(self, tmp_path):
        report_dir = tmp_path / "playwright-report"
        report_dir.mkdir()
        (report_dir / "index.html").write_text("<html></html>")

        raw = {"suites": []}
        result = self.engine._parse_results(raw, tmp_path)

        assert "playwright-report" in result["report_path"]

    def test_report_path_missing(self, tmp_path):
        raw = {"suites": []}
        result = self.engine._parse_results(raw, tmp_path)

        assert result["report_path"] == ""


class TestExtractTestsFromSuite:
    def setup_method(self):
        self.engine = ExecutionEngine()

    def test_empty_suite(self):
        results: list[TestCaseResult] = []
        self.engine._extract_tests_from_suite(
            {"title": "Empty", "specs": [], "suites": []},
            results,
        )
        assert results == []

    def test_suite_with_no_results(self):
        """A spec with a test but no run results still produces an entry with defaults."""
        results: list[TestCaseResult] = []
        self.engine._extract_tests_from_suite(
            {
                "title": "NoResults",
                "specs": [{"title": "test", "tests": [{"results": []}]}],
                "suites": [],
            },
            results,
        )
        assert len(results) == 1
        assert results[0].name == "test"
        assert results[0].status == "passed"
        assert results[0].duration_ms == 0.0


class _FakeStream:
    """Async-context stream returning a fixed byte payload."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, _chunk: int = 65536):
        yield self._payload


class _FakeAsyncClient:
    """Captures constructor kwargs (headers) and serves a fixed payload."""

    captured: ClassVar[dict[str, Any]] = {}

    def __init__(self, **kwargs: Any) -> None:
        type(self).captured = kwargs

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def stream(self, _method: str, _url: str) -> _FakeStream:
        return _FakeStream(b"PK\x05\x06" + b"\x00" * 18)  # minimal empty-zip EOCD


class TestDownloadZipLicenseForwarding:
    """The S2S package download from bdd must carry the caller's license token
    (the bdd download endpoint is fail-closed on unauthenticated in-cluster
    traffic), but must NEVER leak it to an arbitrary external host.
    """

    @pytest.mark.asyncio
    async def test_forwards_license_token_to_internal_bdd(self, tmp_path, monkeypatch):
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeAsyncClient)
        engine = ExecutionEngine()
        await engine._download_zip(
            "http://bdd-playwright-service.w2v.svc.cluster.local:6313/api/bdd/j1/download",
            tmp_path,
            allow_private=True,
            license_token="opaque-license-abc",
        )
        headers = _FakeAsyncClient.captured["headers"]
        assert headers.get("X-Tenant-Id") == "Bearer opaque-license-abc"

    @pytest.mark.asyncio
    async def test_omits_token_for_external_url(self, tmp_path, monkeypatch):
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeAsyncClient)
        engine = ExecutionEngine()
        # allow_private=False is the external path; even if a token is somehow
        # supplied it must not be attached (credential-leak guard).
        await engine._download_zip(
            "https://example.com/pkg.zip",
            tmp_path,
            allow_private=False,
            license_token="opaque-license-abc",
        )
        headers = _FakeAsyncClient.captured["headers"]
        assert "X-Tenant-Id" not in headers

    @pytest.mark.asyncio
    async def test_no_token_no_auth_header(self, tmp_path, monkeypatch):
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.httpx, "AsyncClient", _FakeAsyncClient)
        engine = ExecutionEngine()
        await engine._download_zip(
            "http://bdd-playwright-service.w2v.svc.cluster.local:6313/api/bdd/j1/download",
            tmp_path,
            allow_private=True,
        )
        headers = _FakeAsyncClient.captured["headers"]
        assert "X-Tenant-Id" not in headers


class TestNoLocalLicenceExtractor:
    """routes.py must NOT re-implement w2v_common's licence-extraction rule.

    It carried a local ``_extract_license_token`` that was a copy of the
    PRE-0.74.0 rule (``X-Tenant-Id: Bearer`` only). w2v_common 0.74.0 widened
    the canonical rule to also accept an opaque licence on ``Authorization:
    Bearer`` — the shape the ``license-authz`` ext-authz gate uses — and the
    copy silently drifted: the licence reached the service, authenticated, and
    then the background job got None. No test failed. The route now reads
    ``request.state.license_token``, which AuthMiddleware binds; this guard
    keeps the copy from coming back.
    """

    def test_routes_has_no_local_extractor(self):
        import src.api.routes as routes

        assert not hasattr(routes, "_extract_license_token"), (
            "routes.py re-declared a local licence extractor — read "
            "request.state.license_token (set by AuthMiddleware) instead; a "
            "local copy drifts from w2v_common's canonical rule silently."
        )

    def test_routes_source_does_not_parse_the_licence_headers(self):
        import inspect

        import src.api.routes as routes

        src = inspect.getsource(routes)
        assert 'headers.get("X-Tenant-Id")' not in src
        assert 'headers.get("Authorization")' not in src


class TestNormalizeTrack:
    """Directed-footage (ORCH-3.2): CSS-px sidecar → recorded-video px."""

    def _track(self) -> dict[str, Any]:
        return {
            "viewport": {"w": 960, "h": 540},
            "focus": [{"t": 0.5, "bbox": {"x": 100, "y": 50, "w": 40, "h": 20}, "action": "click"}],
            "cursor": [
                {"t": 0.0, "x": 10, "y": 10, "click": False},
                {"t": 0.5, "x": 120, "y": 60, "click": True},
            ],
        }

    def test_scales_by_width_over_source_viewport(self):
        # Recorded at 1920x1080 but captured against a 960-wide viewport → 2x.
        out = _normalize_track(self._track(), 1920, 1080)
        assert out is not None
        focus, cursor, viewport = out
        assert focus[0]["bbox"] == {"x": 200.0, "y": 100.0, "w": 80.0, "h": 40.0}
        assert focus[0]["action"] == "click"
        assert cursor[1] == {"t": 0.5, "x": 240.0, "y": 120.0, "click": True}
        # viewport rewritten to the RECORDED dims (what the compiler transform reads).
        assert viewport == {"w": 1920, "h": 1080}

    def test_identity_when_viewport_equals_recorded(self):
        t = self._track()
        t["viewport"] = {"w": 1920, "h": 1080}
        out = _normalize_track(t, 1920, 1080)
        assert out is not None
        focus, _cursor, viewport = out
        assert focus[0]["bbox"]["x"] == 100.0
        assert viewport == {"w": 1920, "h": 1080}

    def test_absent_source_width_defaults_factor_one(self):
        t = {"viewport": {}, "focus": [], "cursor": [{"t": 0, "x": 5, "y": 6}]}
        out = _normalize_track(t, 1280, 720)
        assert out is not None
        focus, cursor, viewport = out
        assert focus == []
        assert cursor[0] == {"t": 0.0, "x": 5.0, "y": 6.0, "click": False}
        assert viewport == {"w": 1280, "h": 720}

    def test_x_and_y_scale_independently(self):
        # A recording whose aspect differs from the captured viewport must scale
        # X by width/vp.w and Y by height/vp.h SEPARATELY (else Y skews).
        t = {
            "viewport": {"w": 1000, "h": 1000},
            "focus": [],
            "cursor": [{"t": 0.0, "x": 100, "y": 100, "click": False}],
        }
        out = _normalize_track(t, 1920, 1080)
        assert out is not None
        _f, cursor, viewport = out
        assert cursor[0]["x"] == pytest.approx(100 * 1920 / 1000)  # 192.0
        assert cursor[0]["y"] == pytest.approx(100 * 1080 / 1000)  # 108.0
        assert viewport == {"w": 1920, "h": 1080}

    def test_malformed_track_returns_none(self):
        assert _normalize_track({"focus": "not-a-list"}, 1920, 1080) is None
        assert _normalize_track({"cursor": [{"x": object()}]}, 1920, 1080) is None

    def test_trackless_clip_serialises_new_fields_as_none(self):
        # A clip with no recorded geometry emits the three additive keys as null,
        # so the orchestrator's `sc.get("focus_track")` is a clean None.
        from src.api.schemas import ScenarioClip

        dumped = ScenarioClip(
            clip_key="clip-1", feature="f", scenario="s", status="passed"
        ).model_dump()
        assert dumped["focus_track"] is None
        assert dumped["cursor_track"] is None
        assert dumped["viewport"] is None


class TestLoadAndNormalizeTrack:
    """The file-reading wrapper degrades to None, never raises."""

    def test_reads_and_normalizes(self, tmp_path: Path):
        p = tmp_path / "clip-1.track.json"
        p.write_text(json.dumps({"viewport": {"w": 1920, "h": 1080}, "focus": [], "cursor": []}))
        out = _load_and_normalize_track(p, 1920, 1080)
        assert out is not None
        assert out[2] == {"w": 1920, "h": 1080}

    def test_missing_file_returns_none(self, tmp_path: Path):
        assert _load_and_normalize_track(tmp_path / "nope.json", 1920, 1080) is None

    def test_non_json_returns_none(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert _load_and_normalize_track(p, 1920, 1080) is None

    def test_non_dict_json_returns_none(self, tmp_path: Path):
        p = tmp_path / "arr.json"
        p.write_text("[1, 2, 3]")
        assert _load_and_normalize_track(p, 1920, 1080) is None

    def test_accepts_an_already_parsed_mapping(self):
        """The body-attachment path never touches disk — it hands over a dict."""
        out = _load_and_normalize_track(
            {"viewport": {"w": 1920, "h": 1080}, "focus": [], "cursor": []}, 1920, 1080
        )
        assert out is not None
        assert out[2] == {"w": 1920, "h": 1080}


def _track_body(track: dict[str, Any]) -> str:
    """Base64 of a track payload — exactly how Playwright's JSON reporter
    serializes a ``testInfo.attach(..., { body })`` attachment."""
    return base64.b64encode(json.dumps(track).encode("utf-8")).decode("ascii")


def _body_attachment(track: dict[str, Any]) -> dict[str, Any]:
    """The REAL Playwright JSON-reporter shape for the fixture's attachment.

    Note what is NOT here: a ``path`` key. Playwright only emits ``path`` for
    attachments created FROM a file (the built-in video). A ``body:`` attachment
    is inlined as base64 and has no path at all — which is why a reader gated on
    ``att.get("path")`` harvested zero tracks while video kept working.
    """
    return {
        "name": "w2v-track",
        "contentType": "application/json",
        "body": _track_body(track),
    }


_TRACK = {
    "viewport": {"w": 1280, "h": 720},
    "focus": [{"t": 1.0, "bbox": {"x": 10, "y": 20, "w": 30, "h": 40}, "action": "click"}],
    "cursor": [{"t": 1.0, "x": 10, "y": 20, "click": True}],
}


def _spec_with(*attachments: dict[str, Any]) -> dict[str, Any]:
    return {"tests": [{"results": [{"attachments": list(attachments)}]}]}


class TestLastTrackAttachment:
    """Harvest the w2v-track payload from BOTH real JSON-report shapes.

    The fixture attaches with ``body:``; the previous hand-mirrored tests only
    ever built ``path:`` attachments — a shape Playwright never produces for it —
    so a reader that required ``path`` shipped green and returned None on 40/40
    production runs.
    """

    def test_body_attachment_is_decoded(self):
        out = ExecutionEngine._last_track_attachment(_spec_with(_body_attachment(_TRACK)))
        assert out == _TRACK

    def test_body_attachment_has_no_path_key(self):
        """Pin the fixture shape itself, so it cannot drift back to a path mirror."""
        assert "path" not in _body_attachment(_TRACK)

    def test_body_attachment_survives_confinement_base_dir(self, tmp_path: Path):
        """An inline body touches no filesystem, so base_dir must not reject it."""
        out = ExecutionEngine._last_track_attachment(_spec_with(_body_attachment(_TRACK)), tmp_path)
        assert out == _TRACK

    def test_body_normalizes_end_to_end(self):
        """The decoded payload feeds the normalizer that _persist_scenario_clips calls."""
        track = ExecutionEngine._last_track_attachment(_spec_with(_body_attachment(_TRACK)))
        assert track is not None
        normalized = _load_and_normalize_track(track, 2560, 1440)
        assert normalized is not None
        focus, cursor, viewport = normalized
        assert viewport == {"w": 2560, "h": 1440}
        assert focus[0]["bbox"]["x"] == pytest.approx(20.0)  # 10 * (2560/1280)
        assert cursor[0]["y"] == pytest.approx(40.0)  # 20 * (1440/720)

    def test_path_attachment_still_works(self, tmp_path: Path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps(_TRACK))
        spec = _spec_with({"name": "w2v-track", "path": str(f)})
        assert ExecutionEngine._last_track_attachment(spec) == _TRACK

    def test_both_shapes_present_prefers_body(self, tmp_path: Path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps({"viewport": {"w": 1, "h": 1}, "focus": [], "cursor": []}))
        att = _body_attachment(_TRACK) | {"path": str(f)}
        assert ExecutionEngine._last_track_attachment(_spec_with(att)) == _TRACK

    def test_malformed_base64_falls_back_to_path(self, tmp_path: Path):
        f = tmp_path / "t.json"
        f.write_text(json.dumps(_TRACK))
        att = {"name": "w2v-track", "contentType": "application/json", "body": "!!not base64!!"}
        assert ExecutionEngine._last_track_attachment(_spec_with(att)) is None
        assert ExecutionEngine._last_track_attachment(_spec_with(att | {"path": str(f)})) == _TRACK

    @pytest.mark.parametrize(
        "body",
        [
            "!!not base64!!",
            base64.b64encode(b"{not json").decode("ascii"),
            base64.b64encode(b"[1, 2, 3]").decode("ascii"),  # JSON, but not an object
            base64.b64encode(b"\xff\xfe").decode("ascii"),  # not UTF-8
            None,
            123,
        ],
    )
    def test_malformed_body_degrades_to_none(self, body: object):
        att = {"name": "w2v-track", "contentType": "application/json", "body": body}
        assert ExecutionEngine._last_track_attachment(_spec_with(att)) is None

    def test_attachment_with_neither_body_nor_path(self):
        att = {"name": "w2v-track", "contentType": "application/json"}
        assert ExecutionEngine._last_track_attachment(_spec_with(att)) is None

    def test_none_when_absent(self):
        spec = _spec_with({"name": "video", "path": "/v.webm"})
        assert ExecutionEngine._last_track_attachment(spec) is None

    def test_none_when_path_missing_on_disk(self, tmp_path: Path):
        spec = _spec_with({"name": "w2v-track", "path": str(tmp_path / "gone.json")})
        assert ExecutionEngine._last_track_attachment(spec) is None

    def test_last_attempt_wins_across_retries(self):
        first = {"viewport": {"w": 1, "h": 1}, "focus": [], "cursor": []}
        spec = {
            "tests": [
                {
                    "results": [
                        {"attachments": [_body_attachment(first)]},
                        {"attachments": [_body_attachment(_TRACK)]},
                    ]
                }
            ]
        }
        assert ExecutionEngine._last_track_attachment(spec) == _TRACK

    def test_collect_clip_specs_carries_the_decoded_track(self, tmp_path: Path):
        """The full call chain: report → spec["track"] → normalizer input."""
        pkg = tmp_path / "pkg"
        (pkg / "test-results").mkdir(parents=True)
        vid = pkg / "test-results" / "a.webm"
        vid.write_bytes(b"v")
        raw = {
            "suites": [
                {
                    "title": "F",
                    "specs": [
                        {
                            "title": "S",
                            "tags": ["@clip-1"],
                            "tests": [
                                {
                                    "results": [
                                        {
                                            "status": "passed",
                                            "duration": 1,
                                            "attachments": [
                                                {"name": "video", "path": str(vid)},
                                                _body_attachment(_TRACK),
                                            ],
                                        }
                                    ]
                                }
                            ],
                        }
                    ],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw, pkg)
        assert len(specs) == 1
        assert specs[0]["track"] == _TRACK


class TestAttachmentPathConfinement:
    """A malicious results.json must not exfiltrate files outside the package dir.

    ``att["path"]`` is written by the (untrusted) test package; without SOURCE-path
    confinement, ``_persist_scenario_clips`` would copy an arbitrary named file
    (e.g. ``/app/.env``) into the shared, caller-readable object store.
    """

    def _video_spec(self, tag: str, path: Path) -> dict:
        return {
            "title": tag,
            "tags": [f"@{tag}"],
            "tests": [
                {
                    "results": [
                        {
                            "status": "passed",
                            "duration": 1000,
                            "attachments": [{"name": "video", "path": str(path)}],
                        }
                    ]
                }
            ],
        }

    def test_video_outside_package_dir_is_skipped(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        secret = tmp_path / "secret.env"  # exists, but OUTSIDE the package dir
        secret.write_text("DB_PASSWORD=hunter2")
        spec = self._video_spec("clip-1", secret)
        # base_dir=None (no confinement) still resolves it — proves the file is
        # otherwise perfectly harvestable, so the skip below is the guard at work.
        assert ExecutionEngine._video_test_for_spec(spec, None) is not None
        assert ExecutionEngine._video_test_for_spec(spec, pkg) is None

    def test_video_inside_package_dir_still_found(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        (pkg / "test-results").mkdir(parents=True)
        vid = pkg / "test-results" / "a.webm"
        vid.write_bytes(b"v")
        spec = self._video_spec("clip-1", vid)
        found = ExecutionEngine._video_test_for_spec(spec, pkg)
        assert found is not None
        assert found[1] == vid

    def test_track_outside_package_dir_is_skipped(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps({"viewport": {"w": 1, "h": 1}}))
        spec = {
            "tests": [{"results": [{"attachments": [{"name": "w2v-track", "path": str(outside)}]}]}]
        }
        # base_dir=None (no confinement) reads it — proving the skip below is the
        # guard at work, not a parse failure.
        assert ExecutionEngine._last_track_attachment(spec, None) == {"viewport": {"w": 1, "h": 1}}
        assert ExecutionEngine._last_track_attachment(spec, pkg) is None

    def test_collect_clip_specs_drops_escaping_keeps_legit(self, tmp_path: Path):
        pkg = tmp_path / "pkg"
        (pkg / "test-results").mkdir(parents=True)
        legit = pkg / "test-results" / "good.webm"
        legit.write_bytes(b"ok")
        secret = tmp_path / "secret.env"
        secret.write_text("secret")
        raw = {
            "suites": [
                {
                    "title": "F",
                    "specs": [
                        self._video_spec("clip-1", secret),  # escapes → dropped
                        self._video_spec("clip-2", legit),  # in-package → kept
                    ],
                    "suites": [],
                }
            ]
        }
        specs = ExecutionEngine()._collect_clip_specs(raw, pkg)
        keys = [s["clip_key"] for s in specs]
        assert keys == ["clip-2"]
        assert specs[0]["video_path"] == legit


class TestEnsureTrackRecorder:
    """Config-injection is idempotent and non-destructive."""

    def _package(self, tmp_path: Path) -> Path:
        (tmp_path / "playwright.config.ts").write_text(
            "import { defineBddConfig } from 'playwright-bdd';\n"
            "const testDir = defineBddConfig({ features: 'f/*.feature', steps: 's/*.ts' });\n"
            "export default { testDir };\n"
        )
        return tmp_path

    def test_writes_fixtures_and_injects_import(self, tmp_path: Path):
        pkg = self._package(tmp_path)
        ExecutionEngine._ensure_track_recorder(pkg)
        fixtures = (pkg / "fixtures.ts").read_text()
        # recorder is inlined (placeholder replaced with a JS string literal).
        assert "__W2V_RECORDER_SOURCE__" not in fixtures
        assert "addInitScript" in fixtures
        cfg = (pkg / "playwright.config.ts").read_text()
        assert "importTestFrom: './fixtures.ts'" in cfg

    def test_idempotent_second_run_does_not_double_inject(self, tmp_path: Path):
        pkg = self._package(tmp_path)
        ExecutionEngine._ensure_track_recorder(pkg)
        ExecutionEngine._ensure_track_recorder(pkg)
        cfg = (pkg / "playwright.config.ts").read_text()
        assert cfg.count("importTestFrom: './fixtures.ts'") == 1

    def test_skips_when_package_already_sets_import_test_from(self, tmp_path: Path):
        (tmp_path / "playwright.config.ts").write_text(
            "import { defineBddConfig } from 'playwright-bdd';\n"
            "defineBddConfig({ features: 'f', steps: 's', importTestFrom: './mine.ts' });\n"
        )
        ExecutionEngine._ensure_track_recorder(tmp_path)
        cfg = (tmp_path / "playwright.config.ts").read_text()
        assert "./mine.ts" in cfg
        assert "./fixtures.ts" not in cfg

    def test_no_define_bdd_config_writes_fixtures_but_no_wire(self, tmp_path: Path):
        (tmp_path / "playwright.config.ts").write_text("export default { testDir: './tests' };\n")
        ExecutionEngine._ensure_track_recorder(tmp_path)
        assert (tmp_path / "fixtures.ts").exists()
        assert "importTestFrom" not in (tmp_path / "playwright.config.ts").read_text()


class TestTrackWiringIsLoudAndChains:
    """A capture that ships with no directed-footage track renders a STATIC film
    (footage_direction forces `zoom="off"`, footage_plans pops both tracks). The
    pre-fix code returned silently -- at most a `logger.warning` nobody reads --
    so the ONLY code path that adds camera movement died by data starvation with
    the job still reporting `completed`. Every non-wire now returns a reason that
    rides out on the job as `capture_warnings`.
    """

    @staticmethod
    def _config(tmp_path: Path, body: str) -> Path:
        (tmp_path / "playwright.config.ts").write_text(
            "import { defineBddConfig } from 'playwright-bdd';\n" + body
        )
        return tmp_path

    def test_missing_define_bdd_config_returns_reason_and_logs_error(
        self, tmp_path: Path, caplog
    ) -> None:
        (tmp_path / "playwright.config.ts").write_text("export default { testDir: './tests' };\n")
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            warnings = ExecutionEngine._ensure_track_recorder(tmp_path)
        assert len(warnings) == 1
        assert "defineBddConfig" in warnings[0]
        assert "camera movement OFF" in warnings[0]
        assert "defineBddConfig" in caplog.text  # loud, not a quiet return

    def test_unchainable_import_test_from_returns_reason_and_logs_error(
        self, tmp_path: Path, caplog
    ) -> None:
        """A BARE specifier is not a file we can read or confine, so we cannot
        vouch for its shape -- that one still refuses, but audibly."""
        pkg = self._config(
            tmp_path,
            "defineBddConfig({ features: 'f', steps: 's', importTestFrom: 'vendor-fixtures' });\n",
        )
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            warnings = ExecutionEngine._ensure_track_recorder(pkg)
        assert len(warnings) == 1
        assert "vendor-fixtures" in warnings[0]
        assert "camera movement OFF" in warnings[0]
        assert "vendor-fixtures" in caplog.text

    def test_escaping_import_test_from_is_refused(self, tmp_path: Path) -> None:
        """A specifier pointing outside the extracted package is never chained."""
        pkg = tmp_path / "pkg"
        pkg.mkdir()
        (tmp_path / "outside.ts").write_text("export const test = 1;\n")
        self._config(
            pkg,
            "defineBddConfig({ features: 'f', steps: 's', importTestFrom: '../outside.ts' });\n",
        )
        warnings = ExecutionEngine._ensure_track_recorder(pkg)
        assert len(warnings) == 1
        assert "../outside.ts" in warnings[0]

    def test_chains_onto_the_packages_own_fixture_module(self, tmp_path: Path) -> None:
        """The case that used to bail: the package keeps its own fixtures AND the
        recorder runs, because our fixture extends THEIR test export."""
        pkg = self._config(
            tmp_path,
            "defineBddConfig({ features: 'f', steps: 's', importTestFrom: './mine.ts' });\n",
        )
        (pkg / "mine.ts").write_text(
            "import { test as base } from 'playwright-bdd';\nexport const test = base;\n"
        )
        assert ExecutionEngine._ensure_track_recorder(pkg) == []
        cfg = (pkg / "playwright.config.ts").read_text()
        assert "importTestFrom: './fixtures.ts'" in cfg
        assert "importTestFrom: './mine.ts'" not in cfg  # repointed, not duplicated
        fixtures = (pkg / "fixtures.ts").read_text()
        assert "from './mine.ts'" in fixtures  # chained
        assert "from 'playwright-bdd'" not in fixtures  # base swapped, not appended
        assert "addInitScript" in fixtures

    def test_chain_is_idempotent(self, tmp_path: Path) -> None:
        pkg = self._config(
            tmp_path,
            "defineBddConfig({ features: 'f', steps: 's', importTestFrom: './mine.ts' });\n",
        )
        (pkg / "mine.ts").write_text("export const test = 1;\n")
        ExecutionEngine._ensure_track_recorder(pkg)
        assert ExecutionEngine._ensure_track_recorder(pkg) == []
        cfg = (pkg / "playwright.config.ts").read_text()
        assert cfg.count("importTestFrom") == 1
        assert (pkg / "fixtures.ts").read_text().count("from './mine.ts'") == 1

    def test_wired_package_returns_no_warnings(self, tmp_path: Path) -> None:
        """The normal path (bdd_playwright_service's generated config: a
        defineBddConfig call, no importTestFrom of its own) stays silent."""
        pkg = self._config(
            tmp_path, "const t = defineBddConfig({\n  features: 'f',\n  steps: 's',\n});\n"
        )
        assert ExecutionEngine._ensure_track_recorder(pkg) == []
        assert "importTestFrom: './fixtures.ts'" in (pkg / "playwright.config.ts").read_text()

    def test_all_clips_trackless_is_reported(self, caplog) -> None:
        """Wiring applied + zero interaction recorded is the SAME static film,
        and used to be inferable only from an absence."""
        clips = [{"clip_key": "clip-1"}, {"clip_key": "clip-2", "cursor_track": []}]
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            warnings = ExecutionEngine._track_coverage_warnings(clips)
        assert len(warnings) == 1
        assert "2 captured clip(s)" in warnings[0]
        assert "camera movement OFF" in warnings[0]
        assert "camera movement OFF" in caplog.text

    def test_one_tracked_clip_clears_the_warning(self) -> None:
        clips = [{"clip_key": "clip-1"}, {"clip_key": "clip-2", "cursor_track": [{"t": 0.1}]}]
        assert ExecutionEngine._track_coverage_warnings(clips) == []

    def test_no_clips_at_all_is_the_zero_footage_gates_job(self) -> None:
        """`_guard_zero_footage` already fails that run; do not double-report."""
        assert ExecutionEngine._track_coverage_warnings([]) == []


class TestDiscardedTrackWarnings:
    """A RECORDED track this service threw away must reach `capture_warnings`.

    `_track_coverage_warnings` answers a DIFFERENT question ("did any clip arrive
    with camera direction") and fails open on this one: one surviving track
    silences it for the whole run. So a capture where ffprobe could not measure
    11 of 12 recordings published 11 track-less clips, emitted zero
    caller-visible warnings, and the orchestrator compiled 11 shots with camera
    movement OFF with nothing to attribute it to.
    """

    def test_discarded_tracks_are_named(self, caplog) -> None:
        result = {
            "scenario_clips": [{"clip_key": f"clip-{i}"} for i in range(1, 4)],
            "track_dropped_clips": ["clip-2", "clip-1"],
        }
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            warnings = ExecutionEngine._discarded_track_warnings(result)
        assert len(warnings) == 1
        assert "2 of 3 published clip(s)" in warnings[0]
        assert "clip-1, clip-2" in warnings[0]
        assert "DISCARDED" in warnings[0]
        assert "clip-1, clip-2" in caplog.text

    def test_no_discards_is_silent(self) -> None:
        result = {"scenario_clips": [{"clip_key": "clip-1"}], "track_dropped_clips": []}
        assert ExecutionEngine._discarded_track_warnings(result) == []
        assert ExecutionEngine._discarded_track_warnings({}) == []

    def test_a_surviving_sibling_track_does_not_hide_the_discards(self, caplog) -> None:
        """THE GAP, stated as an assertion.

        11 clips lost a track they recorded; clip-12 kept its. The coverage
        warning correctly stays silent (the run DOES carry camera direction), so
        it cannot be the channel for this — and before the fix nothing else was.
        """
        clips: list[Any] = [{"clip_key": f"clip-{i}"} for i in range(1, 12)]
        clips.append({"clip_key": "clip-12", "cursor_track": [{"t": 0.1, "x": 1.0, "y": 2.0}]})
        result = {
            "scenario_clips": clips,
            "track_dropped_clips": [c["clip_key"] for c in clips[:11]],
        }

        assert ExecutionEngine._track_coverage_warnings(clips) == []
        warnings = ExecutionEngine._discarded_track_warnings(result)
        assert len(warnings) == 1
        assert "11 of 12 published clip(s)" in warnings[0]
        assert "clip-1," in warnings[0]


class TestCaptureRateIsExplicit:
    """Playwright's recorder hard-codes `const fps = 25` (playwright-core
    `server/videoRecorder.js`) and `recordVideo` exposes no fps option, so this
    service CANNOT capture at the compiler's 30fps conform rate. What it can do
    is stop letting ffmpeg infer the output rate: the transcode must be 1:1 with
    what was captured, never a re-timing that invents frames.
    """

    class _Proc:
        returncode = 0

        def __init__(self, stdout: bytes = b"") -> None:
            self._stdout = stdout

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._stdout, b""

    @staticmethod
    def _record_argv(monkeypatch, calls: list[tuple[str, ...]]) -> None:
        async def _fake_create(*args: str, **_k: Any) -> Any:
            calls.append(args)
            return TestCaptureRateIsExplicit._Proc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)

    @staticmethod
    def _patch_probe(monkeypatch, payload: bytes) -> None:
        async def _fake_create(*_a: str, **_k: Any) -> Any:
            return TestCaptureRateIsExplicit._Proc(payload)

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)

    async def test_output_rate_is_pinned_to_the_measured_source_rate(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        async def _fps(_path: Path) -> float:
            return 25.0

        monkeypatch.setattr("src.services.execution_engine._probe_video_fps", _fps)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst) is True
        argv = calls[0]
        assert "-r" in argv, "the transcode must NAME its output rate, not infer it"
        assert argv[argv.index("-r") + 1] == "25"
        # ...as an OUTPUT option (after -i), or it would reinterpret the input.
        assert argv.index("-r") > argv.index(str(src_file))
        # ...and never as an upsample: 25 -> 30 duplicates one frame in six.
        assert argv[argv.index("-r") + 1] != "30"

    async def test_transcode_pins_explicit_crf_below_the_master_range(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Without an explicit -crf, libx264 defaults to 23 -- worse than the
        eventual master's CRF 18-20 -- making this WebM->MP4 intermediate the
        quality FLOOR of the whole picture chain. The transcode must NAME a
        CRF meaningfully better than the master's range, not merely match it.
        """

        async def _fps(_path: Path) -> float:
            return 25.0

        monkeypatch.setattr("src.services.execution_engine._probe_video_fps", _fps)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst) is True
        argv = calls[0]
        assert "-crf" in argv, "the transcode must NAME its CRF, not take libx264's default of 23"
        crf = int(argv[argv.index("-crf") + 1])
        assert crf < 18, "must be meaningfully better than the master's CRF 18-20 floor"

    async def test_unprobeable_source_falls_back_to_ffmpeg_inference(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A fabricated constant would be worse than ffmpeg's own inference."""

        async def _fps(_path: Path) -> None:
            return None

        monkeypatch.setattr("src.services.execution_engine._probe_video_fps", _fps)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst) is True
        assert "-r" not in calls[0]

    async def test_probe_reads_avg_frame_rate_as_a_rational(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        payload = b'{"streams":[{"avg_frame_rate":"25/1","r_frame_rate":"1000/1"}]}'
        self._patch_probe(monkeypatch, payload)
        assert await _probe_video_fps(tmp_path / "clip.webm") == 25.0

    async def test_probe_falls_back_to_r_frame_rate_when_avg_is_unknown(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """ffprobe writes `0/0` for "unknown" -- a salvaged recording with a
        killed muxer routinely does. That is not a rate; use the nominal one."""
        payload = b'{"streams":[{"avg_frame_rate":"0/0","r_frame_rate":"25/1"}]}'
        self._patch_probe(monkeypatch, payload)
        assert await _probe_video_fps(tmp_path / "clip.webm") == 25.0

    async def test_probe_returns_none_when_no_rate_is_knowable(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        payload = b'{"streams":[{"avg_frame_rate":"0/0","r_frame_rate":"0/0"}]}'
        self._patch_probe(monkeypatch, payload)
        assert await _probe_video_fps(tmp_path / "clip.webm") is None


class TestConvertToMp4LeadTrim:
    """`_convert_to_mp4(..., lead_trim_seconds=...)` cuts a measured blank lead
    off the FRONT of the transcode via an input-side `-ss` — the mirror image
    of the tail-trim `w2v_orchestrator` already applies for a failed scenario's
    frozen timeout tail (#166/#1042), here sourced from bdd_playwright_service's
    `paint-lead` annotation instead of pixel analysis."""

    class _Proc:
        returncode = 0

        def __init__(self, stdout: bytes = b"") -> None:
            self._stdout = stdout

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._stdout, b""

    @staticmethod
    def _record_argv(monkeypatch, calls: list[tuple[str, ...]]) -> None:
        async def _fake_create(*args: str, **_k: Any) -> Any:
            calls.append(args)
            return TestConvertToMp4LeadTrim._Proc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)

    @staticmethod
    def _patch_probes(monkeypatch, *, fps: float | None, duration_ms: float) -> None:
        async def _fake_fps(_path: Path) -> float | None:
            return fps

        async def _fake_ffprobe_meta(_path: Path) -> tuple[float, int, int, bool, bool]:
            return duration_ms, 1920, 1080, True, True

        monkeypatch.setattr("src.services.execution_engine._probe_video_fps", _fake_fps)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe_meta)

    async def test_no_lead_trim_requested_emits_no_ss_flag(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """The untrimmed path (the default, `lead_trim_seconds=0.0`) must stay
        BYTE-IDENTICAL to the ffmpeg invocation before this parameter existed —
        no `-ss`, and no extra probe call spent on a clip painted from frame 0."""
        self._patch_probes(monkeypatch, fps=25.0, duration_ms=10_000.0)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst) is True
        assert len(calls) == 1, "no ffprobe duration probe should run when there is nothing to trim"
        assert "-ss" not in calls[0]

    async def test_measured_lead_emits_ss_before_the_input(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        self._patch_probes(monkeypatch, fps=25.0, duration_ms=37_300.0)  # kb3-shaped
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst, lead_trim_seconds=2.28) is True
        argv = calls[0]
        assert "-ss" in argv
        assert argv[argv.index("-ss") + 1] == "2.280"
        # Input-side seek: -ss must precede -i, or ffmpeg reinterprets it as an
        # (inaccurate, output-side) seek on the wrong stream.
        assert argv.index("-ss") < argv.index("-i")
        assert argv.index("-ss") < argv.index(str(src_file))

    async def test_a_lead_beyond_the_ceiling_is_capped(self, tmp_path: Path, monkeypatch) -> None:
        self._patch_probes(monkeypatch, fps=25.0, duration_ms=60_000.0)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst, lead_trim_seconds=30.0) is True
        argv = calls[0]
        assert argv[argv.index("-ss") + 1] == "6.000"  # _LEAD_TRIM_MAX_SECONDS

    async def test_an_unprobeable_source_skips_the_trim_but_still_converts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # duration_ms=0.0/probe_ok=False -> _resolve_lead_trim_seconds sees the
        # unknown-duration sentinel and refuses to trim; the transcode must
        # still happen (never drop a clip over an unrelated probe failure).
        async def _fake_ffprobe_meta(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 0.0, 0, 0, False, False

        async def _fake_fps(_path: Path) -> None:
            return None

        monkeypatch.setattr("src.services.execution_engine._probe_video_fps", _fake_fps)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _fake_ffprobe_meta)
        calls: list[tuple[str, ...]] = []
        self._record_argv(monkeypatch, calls)
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        dst = tmp_path / "out.mp4"
        dst.write_bytes(b"real-mp4-bytes")

        assert await _convert_to_mp4(src_file, dst, lead_trim_seconds=2.28) is True
        assert "-ss" not in calls[0]


class TestBddgenErrorCapture:
    """The subprocess failure text must carry the DIAGNOSIS, not the first 500
    bytes of a minified node code frame.

    Regression for the 2026-08-15 04:55Z production failure: `bddgen` died with a
    multi-kilobyte single-line minified frame from playwright-bdd's bundled
    `dist`, and the stored error was 630 chars cut mid-token at
    `ImportJSONBindingNotDefa` -- the actual `SyntaxError` was thrown away by
    HEAD truncation. Three sessions were spent not diagnosing it.
    """

    @staticmethod
    def _bdd_package(tmp_path: Path) -> Path:
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text(
            '{"name":"x","version":"1.0.0","devDependencies":{"playwright-bdd":"^7"}}'
        )
        return package_dir

    @staticmethod
    def _stub_proc(monkeypatch, *, stdout: bytes, stderr: bytes) -> None:
        """Stub the tool, writing its output to the REDIRECTED stdio files.

        ``_run_node_tool`` captures npm/bddgen output to files rather than pipes
        (so a wall-timeout can still read it — the pipe version dropped it and
        raised a bare ``TimeoutError``), so the stub writes where the real tool
        would: the file objects handed to ``create_subprocess_exec``.
        """

        async def _stub_create_subprocess_exec(*_args: Any, **kwargs: Any) -> Any:
            for stream, payload in (("stdout", stdout), ("stderr", stderr)):
                sink = kwargs.get(stream)
                if hasattr(sink, "write"):
                    sink.write(payload)
                    sink.flush()

            class _Proc:
                pid = 4242
                returncode = 1

                async def wait(self) -> int:
                    return 1

            return _Proc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _stub_create_subprocess_exec)
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)

    async def test_minified_noise_does_not_bury_the_syntax_error(self, tmp_path, monkeypatch):
        """5000 chars of minified frame FOLLOWED by the real error -> the raised
        message must contain the error, not the noise."""
        noise = "const a=1;" * 500  # single 5000-char minified line, as node prints it
        stderr = f"{noise}\n\nSyntaxError: Unexpected token\n    at Object.compile\n".encode()
        engine = ExecutionEngine()
        self._stub_proc(monkeypatch, stdout=b"", stderr=stderr)

        with pytest.raises(RuntimeError) as exc:
            await engine._run_bddgen(self._bdd_package(tmp_path))

        message = str(exc.value)
        assert "bddgen failed with exit code 1" in message
        assert "SyntaxError: Unexpected token" in message

    async def test_stdout_diagnostic_survives_past_the_old_head_window(self, tmp_path, monkeypatch):
        """Replaces `test_stdout_diagnostic_survives_when_stderr_is_empty`, which
        was VACUOUS: with an empty stderr the old `stderr or stdout` already fell
        back to stdout and the diagnostic sat inside the old 500-char head window,
        so it passed against the unfixed code too. Here the diagnostic is pushed
        PAST that window, which is what the old truncation actually threw away.
        """
        engine = ExecutionEngine()
        self._stub_proc(
            monkeypatch,
            stdout=b"resolving step definitions\n" * 40
            + b"This Cucumber Expression has a problem at column 12\n",
            stderr=b"",
        )

        with pytest.raises(RuntimeError) as exc:
            await engine._run_bddgen(self._bdd_package(tmp_path))

        assert "Cucumber Expression has a problem at column 12" in str(exc.value)

    async def test_both_streams_are_represented(self, tmp_path, monkeypatch):
        """`stderr or stdout` discarded stdout whenever stderr was non-empty."""
        engine = ExecutionEngine()
        self._stub_proc(
            monkeypatch,
            stdout=b"loaded 3 feature files from ./features",
            stderr=b"Error: step definition not found",
        )

        with pytest.raises(RuntimeError) as exc:
            await engine._run_bddgen(self._bdd_package(tmp_path))

        message = str(exc.value)
        assert "loaded 3 feature files" in message
        assert "step definition not found" in message


class TestSubprocessErrorBudgeting:
    """`_subprocess_error_text` must give EACH stream its own head+tail budget.

    The first cut of this helper fixed the direction (tail, not head) but kept a
    single budget applied to the JOINED text, which is two separate defects:
    a fatal line that lands FIRST is pushed out by post-fatal chatter, and a
    chatty stderr evicts stdout entirely, label and all.
    """

    def test_head_diagnosis_survives_5000_lines_of_post_fatal_noise(self):
        """D1: the fatal line comes FIRST; a tail-only window shows only warnings."""
        stderr = (
            b"SyntaxError: Unexpected token '}' in features/x.feature\n"
            + 5000 * b"warn deprecated dep-N\n"
        )

        out = _subprocess_error_text(b"", stderr, max_chars=2000)

        assert "SyntaxError: Unexpected token '}' in features/x.feature" in out
        assert "...[elided]" in out, "dropped content must be marked, not silent"
        assert "warn deprecated dep-N" in out, "the tail must survive too"

    def test_noisy_stderr_cannot_evict_stdout(self):
        """D2: a joined cut let 200 lines of stderr erase the stdout diagnosis.

        This is the `_read_playwright_report` shape: stderr is a whole Chromium
        run's log, stdout is the thing that failed to parse as JSON.
        """
        stdout = b"REAL DIAGNOSIS: cucumber expression problem at column 7\n"
        stderr = b"".join(f"chromium noise line {i}\n".encode() for i in range(200))

        out = _subprocess_error_text(stdout, stderr, max_chars=2000)

        assert "REAL DIAGNOSIS: cucumber expression problem at column 7" in out
        assert "stdout:" in out, "the stream label must survive the cut"
        assert "stderr:" in out
        assert out.index("stdout:") < out.index("stderr:"), "stderr stays LAST"
        assert "chromium noise line 199" in out

    def test_labels_and_line_boundaries_survive_a_tiny_char_budget(self):
        """D4: the char cut used to slice mid-line and could swallow a label."""
        out = _subprocess_error_text(b"stdoutdiag\n", b"x" * 5000, max_chars=100)

        assert "stdout:" in out
        assert "stderr:" in out
        assert "stdoutdiag" in out
        # Every line is either a label, a marker, or a whole/explicitly-marked
        # line -- nothing is silently sliced.
        for line in out.splitlines():
            assert line in {"stdout:", "stderr:", "...[elided]", "stdoutdiag"} or line.startswith(
                ("...[line truncated]", "x")
            ), line

    def test_no_output_is_reported_as_such(self):
        assert _subprocess_error_text(b"", b"  \n", max_chars=500) == "<no subprocess output>"

    def test_short_output_is_not_marked_truncated(self):
        out = _subprocess_error_text(b"", b"Error: boom\n", max_chars=2000)

        assert out == "stderr:\nError: boom"


class TestFfmpegFailureIsNotHeadTruncated:
    """ffmpeg/ffprobe failures must survive the truncation budget.

    Both call sites used to store ``.decode()[:500]`` — a HEAD slice. ffmpeg
    prints its banner and the whole ``configuration:`` flag list FIRST, so 500
    characters was pure banner and the ONE line that explains the failure
    (``moov atom not found``) fell off the end. This sits on the path where the
    clip is then dropped as degenerate, so it is the only record of a lost clip.
    """

    _BANNER = (
        b"ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers\n"
        b"  built with gcc 13 (Debian 13.2.0-13)\n"
        + b"  configuration: --prefix=/usr --extra-version=1 --toolchain=hardened "
        b"--enable-gpl --enable-libaom --enable-libass --enable-libfdk-aac "
        b"--enable-libfreetype --enable-libmp3lame --enable-libopus "
        b"--enable-libvorbis --enable-libvpx --enable-libx264 --enable-libx265\n" * 6
    )

    class _Proc:
        def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b"") -> None:
            self.returncode = returncode
            self._out = stdout
            self._err = stderr

        async def communicate(self) -> tuple[bytes, bytes]:
            return self._out, self._err

    @staticmethod
    def _patch(monkeypatch, proc: TestFfmpegFailureIsNotHeadTruncated._Proc) -> None:
        async def _fake_create(*_a: Any, **_k: Any) -> Any:
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)

    async def test_convert_failure_keeps_cause_after_a_long_banner(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """The cause is the LAST line, far past character 500. It must survive."""
        stderr = self._BANNER + b"clip.webm: Invalid data found when processing input\n"
        assert len(self._BANNER) > 500, "banner must be long enough to reproduce the bug"
        self._patch(monkeypatch, self._Proc(1, stderr=stderr))
        src_file = tmp_path / "in.webm"
        src_file.write_bytes(b"x")
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            assert await _convert_to_mp4(src_file, tmp_path / "out.mp4") is False
        assert "Invalid data found when processing input" in caplog.text
        # ...and the message stays bounded rather than dumping the whole stream.
        assert len(caplog.text) < 6000

    async def test_ffprobe_unparseable_json_keeps_trailing_cause(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """ffprobe's stdout parse failure: the tail carries the malformed region."""
        stdout = b'{\n  "format": {\n' + b'    "filler": "x",\n' * 200 + b"UNPARSEABLE_TAIL"
        self._patch(monkeypatch, self._Proc(0, stdout=stdout))
        media = tmp_path / "clip.webm"
        media.write_bytes(b"x")
        with caplog.at_level(logging.WARNING, logger="src.services.execution_engine"):
            assert await _ffprobe_meta(media) == (0.0, 0, 0, False, False)
        assert "UNPARSEABLE_TAIL" in caplog.text


class TestWallTimeoutSurfacesCapturedOutput:
    """The wall-timeout branch returned without ever opening ``_pw_stdout.log`` /
    ``_pw_stderr.log``, and ``execute()``'s ``finally`` then rmtree'd the package
    dir — so a browser-launch crash loop was reported ONLY as the guessed
    "browser likely never launched" line, with node's real stderr deleted unread.
    """

    def test_read_captured_stdio_keeps_head_and_tail(self, tmp_path: Path) -> None:
        out = tmp_path / "_pw_stdout.log"
        err = tmp_path / "_pw_stderr.log"
        out.write_bytes(b"Running 12 tests using 1 worker\n")
        err.write_bytes(
            b"browserType.launch: Executable doesn't exist at /ms-playwright/chromium/headless\n"
            + b"  npm warn deprecated some-package@1.0.0\n" * 500
            + b"Error: Process from config.webServer exited early.\n"
        )
        text = ExecutionEngine._read_captured_stdio(out, err)
        assert "Executable doesn't exist" in text  # head: the launch failure
        assert "exited early" in text  # tail: the last word
        assert "Running 12 tests" in text  # stdout is not evicted by stderr
        assert len(text) <= 4000  # budgeted, not a raw dump

    def test_read_captured_stdio_reports_a_missing_log(self, tmp_path: Path) -> None:
        """An absent log is itself the diagnosis (the redirect never happened)."""
        text = ExecutionEngine._read_captured_stdio(
            tmp_path / "_pw_stdout.log", tmp_path / "_pw_stderr.log"
        )
        assert "unreadable" in text
        assert "FileNotFoundError" in text

    def test_zero_salvage_log_carries_the_captured_stderr(self, caplog) -> None:
        diagnostics = "stderr:\nbrowserType.launch: Executable doesn't exist at /ms-playwright"
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            assert ExecutionEngine._log_salvage_outcome({"suites": []}, diagnostics) == 0
        assert "browser likely never launched" in caplog.text  # the guess, still useful
        assert "Executable doesn't exist" in caplog.text  # ...and now the EVIDENCE

    async def test_run_tests_timeout_logs_the_real_stderr(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        """Drive the actual wall-timeout branch of ``_run_tests``.

        Pre-fix this returned ``(salvaged, True)`` having read nothing, so the
        only trace of the crash loop was the guessed cause.
        """

        class _Proc:
            returncode: int | None = None

            async def wait(self) -> int:
                return 0

        async def _fake_create(*_a: Any, **_k: Any) -> Any:
            # The engine redirects stdio to these files; emulate the browser
            # having written its launch failure there before the wall expired.
            (tmp_path / "_pw_stdout.log").write_bytes(b"Running 1 test using 1 worker\n")
            (tmp_path / "_pw_stderr.log").write_bytes(
                b"browserType.launch: Executable doesn't exist at /ms-playwright\n"
            )
            return _Proc()

        async def _boom(*_a: Any, **_k: Any) -> None:
            raise TimeoutError

        async def _noop(*_a: Any, **_k: Any) -> None:
            return None

        monkeypatch.setattr("asyncio.create_subprocess_exec", _fake_create)
        monkeypatch.setattr(ExecutionEngine, "_capture_pgid", staticmethod(lambda _p: None))
        monkeypatch.setattr(ExecutionEngine, "_wait_for_report_or_exit", staticmethod(_boom))
        monkeypatch.setattr(ExecutionEngine, "_terminate_process_group", _noop)

        request = TestExecutionRequest(
            target_url="https://example.com", package_url="https://example.com/pkg.zip"
        )
        with caplog.at_level(logging.ERROR, logger="src.services.execution_engine"):
            raw, timed_out = await ExecutionEngine()._run_tests(tmp_path, request)

        assert timed_out is True
        assert _count_specs(raw) == 0
        assert "browser likely never launched" in caplog.text
        assert "Executable doesn't exist at /ms-playwright" in caplog.text


class TestSoftFailDemotionCarriesItsReason:
    """A soft-failed test is demoted to `failed`, but Playwright reported it
    PASSED — so it has no error text, and the orchestrator rendered the result
    literally as `<no error>`:

        FTR scenario Managing Daily Tasks/Completing a task to track progress
        failed: <no error>

    (orchestrator E2E run 32150382725). The reason existed all along in the
    soft-fail annotation's `description`; nothing read it.
    """

    def _suite(self, annotations: list[dict[str, str]]) -> dict[str, object]:
        return {
            "title": "Managing Daily Tasks",
            "specs": [
                {
                    "title": "Completing a task",
                    "tests": [
                        {
                            "annotations": annotations,
                            "results": [{"status": "passed", "duration": 1200}],
                        }
                    ],
                }
            ],
            "suites": [],
        }

    def _one(self, engine: ExecutionEngine, annotations: list[dict[str, str]]):
        out: list[TestCaseResult] = []
        engine._extract_tests_from_suite(self._suite(annotations), out)
        assert len(out) == 1
        return out[0]

    def test_demoted_test_names_the_selector_that_missed(self) -> None:
        r = self._one(
            ExecutionEngine(),
            [{"type": "soft-fail", "description": "click: .todo-list .toggle not found"}],
        )
        assert r.status == "failed"
        assert r.error, "a demoted soft-fail must not reach the caller with an empty error"
        assert ".todo-list .toggle" in r.error

    def test_every_soft_fail_reason_is_carried(self) -> None:
        r = self._one(
            ExecutionEngine(),
            [
                {"type": "soft-fail", "description": "click: A missing"},
                {"type": "soft-fail", "description": "fill: B missing"},
            ],
        )
        assert "A missing" in r.error
        assert "B missing" in r.error

    def test_a_descriptionless_annotation_still_yields_a_message(self) -> None:
        r = self._one(ExecutionEngine(), [{"type": "soft-fail"}])
        assert r.status == "failed"
        assert r.error.strip(), "must never be empty — that is the <no error> defect"

    def test_unrelated_annotations_do_not_demote_or_invent_an_error(self) -> None:
        r = self._one(ExecutionEngine(), [{"type": "issue", "description": "JIRA-1"}])
        assert r.status == "passed"
        assert r.error == ""

    def test_a_real_playwright_error_is_not_overwritten(self) -> None:
        engine = ExecutionEngine()
        suite = {
            "title": "S",
            "specs": [
                {
                    "title": "T",
                    "tests": [
                        {
                            "annotations": [{"type": "soft-fail", "description": "late miss"}],
                            "results": [
                                {
                                    "status": "failed",
                                    "duration": 10,
                                    "error": {"message": "the REAL failure"},
                                }
                            ],
                        }
                    ],
                }
            ],
            "suites": [],
        }
        out: list[TestCaseResult] = []
        engine._extract_tests_from_suite(suite, out)
        assert "the REAL failure" in out[0].error


class TestAttemptAccounting:
    """`_summarize_test_run` must report the DECIDING attempt, not the sum.

    Playwright's ``results`` are RETRIES of one test. Summing every attempt's
    ``duration`` reported ~3x the real wall time at ``retries=2``, and a
    ``skipped`` first attempt was never cleared by a later ``passed`` — so
    ``[skipped, passed]`` was reported as ``skipped``, dropping a passing test
    out of the pass rate.
    """

    def test_duration_is_not_summed_across_attempts(self):
        results: list[dict[str, Any]] = [
            {"status": "failed", "duration": 1000},
            {"status": "failed", "duration": 1100},
            {"status": "failed", "duration": 1200},
        ]
        _, _, duration_ms = ExecutionEngine._summarize_test_run(results)
        # The deciding (last) attempt, NOT 3300.
        assert duration_ms == 1200.0

    def test_flaky_reports_the_passing_attempts_duration(self):
        results: list[dict[str, Any]] = [
            {"status": "timedOut", "duration": 30000},
            {"status": "passed", "duration": 900},
        ]
        status, _, duration_ms = ExecutionEngine._summarize_test_run(results)
        assert status == "flaky"
        assert duration_ms == 900.0

    def test_failure_reports_the_failing_attempts_duration(self):
        results: list[dict[str, Any]] = [
            {"status": "passed", "duration": 900},
            {"status": "failed", "duration": 4000},
        ]
        status, _, duration_ms = ExecutionEngine._summarize_test_run(results)
        assert status == "failed"
        assert duration_ms == 4000.0

    def test_skipped_then_passed_reports_passed(self):
        results: list[dict[str, Any]] = [
            {"status": "skipped", "duration": 0},
            {"status": "passed", "duration": 1500},
        ]
        status, _, duration_ms = ExecutionEngine._summarize_test_run(results)
        assert status == "passed"
        assert duration_ms == 1500.0

    def test_skipped_only_stays_skipped(self):
        results: list[dict[str, Any]] = [{"status": "skipped", "duration": 0}]
        status, _, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "skipped"

    def test_flaky_rule_still_holds_for_failed_then_passed(self):
        """The pre-existing flaky rule must survive the terminal-pass change."""
        results: list[dict[str, Any]] = [
            {"status": "failed", "duration": 500},
            {"status": "passed", "duration": 1000},
        ]
        status, _, _ = ExecutionEngine._summarize_test_run(results)
        assert status == "flaky"

    def test_single_attempt_duration_is_unchanged(self):
        results: list[dict[str, Any]] = [{"status": "passed", "duration": 1234}]
        status, _, duration_ms = ExecutionEngine._summarize_test_run(results)
        assert status == "passed"
        assert duration_ms == 1234.0


class TestSalvagePrefersThePassingRetry:
    """Attempt 0's dir sorts before ``-retry1``, and `_collect_clip_specs` dedupes
    first-wins on ``@clip-<n>`` — so plain lexicographic salvage kept the FAILED
    attempt's footage and threw away the passing retry's."""

    @staticmethod
    def _package_with_retry(tmp_path: Path) -> Path:
        pkg = tmp_path
        first = pkg / "test-results" / "login-clip-3-chromium"
        first.mkdir(parents=True)
        (first / "video.webm").write_bytes(b"attempt0")
        retry = pkg / "test-results" / "login-clip-3-chromium-retry1"
        retry.mkdir(parents=True)
        (retry / "video.webm").write_bytes(b"attempt1")
        return pkg

    def test_highest_attempt_wins_the_clip_key(self, tmp_path: Path):
        pkg = self._package_with_retry(tmp_path)
        raw = ExecutionEngine()._salvage_raw_from_disk(pkg)
        clips = ExecutionEngine()._collect_clip_specs(raw)
        assert [c["clip_key"] for c in clips] == ["clip-3"]
        assert str(clips[0]["video_path"]).endswith("login-clip-3-chromium-retry1/video.webm")

    def test_double_digit_retry_beats_single_digit(self, tmp_path: Path):
        """Ordinal is parsed as an INT — a string sort would rank retry9 > retry10."""
        pkg = tmp_path
        for suffix in ("", "-retry9", "-retry10"):
            d = pkg / "test-results" / f"login-clip-3-chromium{suffix}"
            d.mkdir(parents=True)
            (d / "video.webm").write_bytes(b"v")
        clips = ExecutionEngine()._collect_clip_specs(ExecutionEngine()._salvage_raw_from_disk(pkg))
        assert str(clips[0]["video_path"]).endswith("login-clip-3-chromium-retry10/video.webm")

    def test_unrelated_tests_keep_their_own_clips(self, tmp_path: Path):
        pkg = self._package_with_retry(tmp_path)
        other = pkg / "test-results" / "checkout-clip-4-chromium"
        other.mkdir(parents=True)
        (other / "video.webm").write_bytes(b"v")
        clips = ExecutionEngine()._collect_clip_specs(ExecutionEngine()._salvage_raw_from_disk(pkg))
        assert sorted(c["clip_key"] for c in clips) == ["clip-3", "clip-4"]


class TestSubprocessReapedOnSuccess:
    """`_install_dependencies` / `_run_bddgen` reaped the process group ONLY in
    their ``except TimeoutError`` branch. npm/npx exiting 0 does not mean the
    group is empty — an esbuild/tsx daemon outlives the leader and orphans under
    FTR's PID 1. Both paths now reap in a ``finally``, the shape `_run_tests` uses.
    """

    @staticmethod
    def _spy_reaps(monkeypatch, engine: ExecutionEngine, *, returncode: int) -> list[Any]:
        calls: list[Any] = []

        async def _stub_create_subprocess_exec(*_args: Any, **kwargs: Any) -> Any:
            sink = kwargs.get("stdout")
            if hasattr(sink, "write"):
                sink.write(b"ok")

            class _Proc:
                pid = 4242
                returncode = 0

                async def wait(self) -> int:
                    return 0

            proc = _Proc()
            proc.returncode = returncode
            return proc

        monkeypatch.setattr("asyncio.create_subprocess_exec", _stub_create_subprocess_exec)
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)

        async def _record(proc: Any, pgid: int | None) -> None:
            calls.append((proc, pgid))

        monkeypatch.setattr(engine, "_terminate_process_group", _record)
        return calls

    @staticmethod
    def _package(tmp_path: Path, *, bdd: bool) -> Path:
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        deps = '"devDependencies":{"playwright-bdd":"^7"}' if bdd else '"devDependencies":{}'
        (package_dir / "package.json").write_text(f'{{"name":"x","version":"1.0.0",{deps}}}')
        return package_dir

    async def test_install_dependencies_reaps_on_success(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        calls = self._spy_reaps(monkeypatch, engine, returncode=0)
        await engine._install_dependencies(self._package(tmp_path, bdd=False))
        assert len(calls) == 1

    async def test_install_dependencies_reaps_on_nonzero_exit(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        calls = self._spy_reaps(monkeypatch, engine, returncode=1)
        with pytest.raises(RuntimeError):
            await engine._install_dependencies(self._package(tmp_path, bdd=False))
        assert len(calls) == 1

    async def test_run_bddgen_reaps_on_success(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        calls = self._spy_reaps(monkeypatch, engine, returncode=0)
        await engine._run_bddgen(self._package(tmp_path, bdd=True))
        assert len(calls) == 1

    async def test_run_bddgen_reaps_on_nonzero_exit(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        calls = self._spy_reaps(monkeypatch, engine, returncode=1)
        with pytest.raises(RuntimeError):
            await engine._run_bddgen(self._package(tmp_path, bdd=True))
        assert len(calls) == 1


class TestPlaywrightErrorKeepsItsCallLog:
    """`error[:1000]` head-truncated the stored Playwright error, cutting the
    trailing ``Call log:`` block — the only part naming the failing locator.
    Same defect class as #93, routed through the same head+tail clamp."""

    @staticmethod
    def _suite_with_error(message: str) -> dict[str, Any]:
        return {
            "title": "Checkout",
            "specs": [
                {
                    "title": "adds an item",
                    "tests": [
                        {
                            "results": [
                                {"status": "failed", "duration": 5, "error": {"message": message}}
                            ]
                        }
                    ],
                }
            ],
            "suites": [],
        }

    def test_call_log_survives_a_long_error(self):
        message = (
            "TimeoutError: locator.click: Timeout 30000ms exceeded.\n"
            + "\n".join(f"  filler line {i} of post-fatal chatter" for i in range(200))
            + "\nCall log:\n  - waiting for getByRole('button', { name: 'Buy now' })\n"
        )
        assert len(message) > 1000
        results: list[TestCaseResult] = []
        ExecutionEngine()._extract_tests_from_suite(self._suite_with_error(message), results)
        stored = results[0].error
        # ``max_chars`` budgets the BODY; the elision marker is headroom on top,
        # exactly as every other `_subprocess_error_text` call site behaves.
        assert len(stored) < len(message)
        assert len(stored) <= 1100
        assert "Call log:" in stored
        assert "getByRole('button', { name: 'Buy now' })" in stored
        # The head fingerprint is kept too -- this is head+tail, not tail-only.
        assert "TimeoutError: locator.click" in stored

    def test_short_error_is_passed_through_unchanged(self):
        results: list[TestCaseResult] = []
        ExecutionEngine()._extract_tests_from_suite(
            self._suite_with_error("Error: expected 2 got 3"), results
        )
        assert results[0].error == "Error: expected 2 got 3"

    def test_empty_error_stays_empty(self):
        results: list[TestCaseResult] = []
        ExecutionEngine()._extract_tests_from_suite(
            {
                "title": "S",
                "specs": [{"title": "t", "tests": [{"results": [{"status": "passed"}]}]}],
                "suites": [],
            },
            results,
        )
        assert results[0].error == ""


class TestCleanupSurvivesCancellation:
    """A4. The reap's grace ``await proc.wait()`` sits inside
    ``contextlib.suppress(TimeoutError, ProcessLookupError, ChildProcessError)``,
    and ``CancelledError`` is a ``BaseException`` — none of those catch it. Under
    a REAL cancellation (client disconnect / SIGTERM drain) SIGTERM landed and the
    SIGKILL escalation plus the reap were skipped, leaking the whole
    npx→node→browser→ffmpeg group. ``asyncio.shield`` detaches the reap so it
    finishes anyway; the cancellation still propagates."""

    class _Proc:
        __test__ = False
        returncode = None
        pid = 4242

        def __init__(self) -> None:
            self.waits_completed = 0

        async def wait(self) -> int:
            # Park, so the cancellation lands DURING the grace wait — and count
            # COMPLETIONS, not entries: a wait cancelled half-way is not a reap.
            await asyncio.sleep(0.05)
            self.waits_completed += 1
            return 0

    @staticmethod
    def _spy_killpg(monkeypatch) -> list[int]:
        import src.services.execution_engine as mod

        sent: list[int] = []
        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)  # FTR's own group = 0
        monkeypatch.setattr(mod.os, "killpg", lambda _pgid, sig: sent.append(sig))
        return sent

    async def test_sigkill_and_reap_still_run_after_cancellation(self, monkeypatch):
        import src.services.execution_engine as mod

        sent = self._spy_killpg(monkeypatch)
        proc = self._Proc()
        task = asyncio.create_task(
            ExecutionEngine()._terminate_process_group(cast(asyncio.subprocess.Process, proc), 4242)
        )
        # Let it reach the grace wait (SIGTERM sent, parked in proc.wait()).
        for _ in range(50):
            if sent:
                break
            await asyncio.sleep(0)
        assert sent == [mod.signal.SIGTERM], "reap never reached the grace wait"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The detached reap must still finish: SIGKILL + both waits reaped.
        for _ in range(60):
            if len(sent) >= 2 and proc.waits_completed >= 2:
                break
            await asyncio.sleep(0.01)
        assert sent == [mod.signal.SIGTERM, mod.signal.SIGKILL]
        assert proc.waits_completed == 2

    async def test_cancellation_is_reraised_not_swallowed(self, monkeypatch):
        """The caller must still see the cancellation — a shielded cleanup that
        swallows it turns a cancelled request into a successful one."""
        self._spy_killpg(monkeypatch)
        task = asyncio.create_task(
            ExecutionEngine()._terminate_process_group(
                cast(asyncio.subprocess.Process, self._Proc()), 4242
            )
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_pending_reap_is_strongly_referenced(self, monkeypatch):
        """asyncio keeps only a WEAK reference to a running task; a reap detached
        during teardown can otherwise be GC'd mid-await and never kill anything."""
        import src.services.execution_engine as mod

        self._spy_killpg(monkeypatch)
        proc = self._Proc()
        task = asyncio.create_task(
            ExecutionEngine()._terminate_process_group(cast(asyncio.subprocess.Process, proc), 4242)
        )
        await asyncio.sleep(0.01)
        assert mod._PENDING_REAPS, "no strong reference held to the in-flight reap"
        await task
        assert not mod._PENDING_REAPS, "done callback must release the reference"


class TestClampIsAHardUpperBound:
    """A3. ``_clamp_error_text(max_chars=N)`` delegated to ``_render_stream``,
    which budgeted only the BODY at ``N - 1`` and then appended ``_ELISION`` plus
    a newline — returning up to ~N+11. The old ``error[:1000]`` was a hard <=1000,
    so any consumer sized to the declared budget silently overflowed."""

    @pytest.mark.parametrize("max_chars", [24, 60, 200, 1000, 4000])
    @pytest.mark.parametrize("multiplier", [0.5, 0.9, 1.0, 1.05, 50])
    def test_never_exceeds_declared_budget(self, max_chars: int, multiplier: float):
        # Many short lines: exercises whole-line dropping AND the line window.
        line_count = max(int(max_chars * multiplier) // 11, 1)
        text = "\n".join(f"line-{i:04d}" for i in range(line_count))
        out = _clamp_error_text(text, max_chars=max_chars)
        assert len(out) <= max_chars, f"{len(out)} > {max_chars}"

    @pytest.mark.parametrize("max_chars", [24, 60, 1000])
    def test_never_exceeds_budget_on_one_huge_line(self, max_chars: int):
        """Node's minified code frame: a single multi-kilobyte line."""
        out = _clamp_error_text("x" * 50_000, max_chars=max_chars)
        assert len(out) <= max_chars

    def test_boundary_inputs_around_the_budget(self):
        for length in (999, 1000, 1001, 1011, 1012, 20_000):
            body = "\n".join("abcdefgh" for _ in range(length // 9 + 1))[:length]
            out = _clamp_error_text(body, max_chars=1000)
            assert len(out) <= 1000, f"len={len(out)} for input {length}"

    def test_still_keeps_the_tail_diagnosis(self):
        """The bound must not cost the property the clamp exists for: Playwright
        puts the failing locator LAST, and it must survive."""
        text = "noise\n" * 500 + 'Call log: waiting for getByRole("button", { name: "Save" })'
        out = _clamp_error_text(text, max_chars=1000)
        assert 'name: "Save"' in out
        assert len(out) <= 1000

    def test_subprocess_error_text_also_bounded(self):
        out = _subprocess_error_text(b"a\n" * 5000, b"b\n" * 5000, max_chars=500)
        assert len(out) <= 500


class TestSalvageRetryGroupingWithRetryLikeName:
    """A2. ``_salvage_sort_key`` stripped ``-retry<N>$`` unconditionally, so a
    scenario whose sanitized dir name LEGITIMATELY ends ``-retry3`` grouped its
    attempt-0 take under ``…-user-can-`` and its retry under ``…-user-can-retry3``
    — two different groups, so the FAILED attempt-0 footage still sorted first and
    still won the first-wins dedupe on ``clip-<n>``. The retry marker is now only
    read as one when the stripped dir ALSO exists on disk."""

    @staticmethod
    def _package(tmp_path: Path, base: str) -> Path:
        for suffix in ("", "-retry1"):
            d = tmp_path / "test-results" / f"{base}{suffix}"
            d.mkdir(parents=True)
            (d / "video.webm").write_bytes(b"v")
        return tmp_path

    def test_scenario_name_ending_in_retry_still_prefers_the_retry(self, tmp_path: Path):
        base = "login-clip-3-user-can-retry3"
        pkg = self._package(tmp_path, base)
        clips = ExecutionEngine()._collect_clip_specs(ExecutionEngine()._salvage_raw_from_disk(pkg))
        assert [c["clip_key"] for c in clips] == ["clip-3"]
        assert str(clips[0]["video_path"]).endswith(f"{base}-retry1/video.webm")

    def test_lone_retry_like_dir_is_not_treated_as_a_retry(self, tmp_path: Path):
        """With no attempt-0 sibling on disk, ``-retry3`` is part of the NAME."""
        d = tmp_path / "test-results" / "login-clip-3-user-can-retry3"
        d.mkdir(parents=True)
        (d / "video.webm").write_bytes(b"v")
        key = _salvage_sort_key(d / "video.webm", {"login-clip-3-user-can-retry3"})
        assert key == ("login-clip-3-user-can-retry3", 0, "video.webm")


class TestFootageYieldIsAttributed:
    """Short footage was completely invisible; only ABSENT footage was reported.

    `_guard_zero_footage` returns on `if clips: return`, so a run that declared 6
    recordable scenarios and persisted 1 completes clean. Per-clip, the only
    check is `_is_degenerate`, which tests `duration_ms == 0` — a 0.4s clip
    publishes as healthy.

    That is measurably the live defect: the orchestrator sizes its capture order
    at ~4.0s per scenario, a footage beat is PINNED to its measured clip length
    (so a short clip cannot be grown downstream), and roughly one demo_video
    render in three fails GROSSLY UNDERFULL as a result.

    These pin the attribution. It is reporting, never a gate — a wrong threshold
    must produce a misleading warning, never a failed run.
    """

    @staticmethod
    def _clip(key: str, ms: float) -> dict[str, object]:
        return {"clip_key": key, "duration_ms": ms}

    def test_a_partial_harvest_is_named(self) -> None:
        clips = [self._clip("c1", 4200.0)]

        warnings = ExecutionEngine._footage_yield_warnings(clips, declared=6, expect_recording=True)

        assert any("SHORTFALL" in w for w in warnings)
        assert any("1 clip(s) persisted against 6" in w for w in warnings)
        assert any("17%" in w for w in warnings), "the fraction actually delivered must be stated"

    def test_clips_shorter_than_the_assumption_are_named_with_the_numbers(self) -> None:
        clips = [self._clip("c1", 900.0), self._clip("c2", 1100.0), self._clip("c3", 700.0)]

        warnings = ExecutionEngine._footage_yield_warnings(clips, declared=3, expect_recording=True)

        yield_warning = next(w for w in warnings if "YIELD" in w)
        assert "mean=0.90s" in yield_warning
        assert "4.0s" in yield_warning, "the assumption being measured must be quoted"
        assert "c3=0.70s" in yield_warning, "the shortest clips must be named"

    def test_a_healthy_harvest_says_nothing(self) -> None:
        clips = [self._clip("c1", 4200.0), self._clip("c2", 4400.0)]

        assert (
            ExecutionEngine._footage_yield_warnings(clips, declared=2, expect_recording=True) == []
        )

    def test_a_non_recording_package_is_never_warned_about(self) -> None:
        assert ExecutionEngine._footage_yield_warnings([], declared=0, expect_recording=False) == []
        assert (
            ExecutionEngine._footage_yield_warnings(
                [self._clip("c1", 100.0)], declared=4, expect_recording=False
            )
            == []
        )

    def test_a_clip_with_no_duration_does_not_crash_the_capture(self) -> None:
        """Reporting, not a gate — this runs on the way OUT of a real capture.

        `duration_ms` is absent whenever ffprobe was missing or hung, which is
        exactly when the run most needs its other warnings to survive. A
        `None`/missing duration must read as 0.0 and feed the mean, not raise.
        """
        clips: list[dict[str, object]] = [{"clip_key": "c1", "duration_ms": None}, {}]

        warnings = ExecutionEngine._footage_yield_warnings(clips, declared=2, expect_recording=True)

        yield_warning = next(w for w in warnings if "YIELD" in w)
        assert "mean=0.00s" in yield_warning


# ── 2026-08-21 bug-hunt ─────────────────────────────────────


class TestPartialFootageIsObservable:
    """#1. ``_select_publishable_artifact`` DROPS a degenerate clip and
    ``_guard_zero_footage`` returns early the moment ``clips`` is non-empty, so a
    15-spec suite with 3 duds completed with 12 clips and NOTHING in the response
    said 3 shots were missing (``capture_warnings`` only covered track coverage).
    The orchestrator then joins narration onto clip keys that never arrive.
    """

    @staticmethod
    def _probe_degenerate_keys(monkeypatch, degenerate: set[str]) -> None:
        """``_ffprobe_meta`` stub: named clip keys probe as 0ms, the rest healthy."""

        async def _probe(path: Path) -> tuple[float, int, int, bool, bool]:
            if any(path.name.startswith(f"{key}.") for key in degenerate):
                return 0.0, 0, 0, False, True
            return 1200.0, 1280, 720, False, True

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _probe)

    async def _run(
        self,
        tmp_path: Path,
        monkeypatch,
        *,
        n_specs: int,
        degenerate: set[str],
    ) -> dict[str, Any]:
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        video = package_dir / "rec.webm"
        video.write_bytes(b"a real recording")
        report = _nested_report_with_specs(n_specs, video_path=video)

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        self._probe_degenerate_keys(monkeypatch, degenerate)

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            return await TestZeroFootageGate()._execute(engine, tmp_path, package_dir)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_a_dropped_clip_is_named_in_capture_warnings(self, tmp_path: Path, monkeypatch):
        """1 dud of 4 must NOT fail the job (that would void 3 good clips
        permanently under the Idempotency-Key semantics) but it MUST be visible."""
        result = await self._run(tmp_path, monkeypatch, n_specs=4, degenerate={"clip-2"})
        assert [c["clip_key"] for c in result["scenario_clips"]] == ["clip-1", "clip-3", "clip-4"]
        dropped_warnings = [w for w in result["capture_warnings"] if "clip-2" in w]
        assert dropped_warnings, result["capture_warnings"]
        assert "1 of 4" in dropped_warnings[0]

    async def test_losing_the_majority_of_the_footage_fails_the_run(
        self, tmp_path: Path, monkeypatch
    ):
        """3 duds of 4: the film would be missing most of its shots, which is the
        zero-footage case in all but name. Must fail, naming the dropped keys."""
        with pytest.raises(RuntimeError, match="clip-2"):
            await self._run(
                tmp_path, monkeypatch, n_specs=4, degenerate={"clip-2", "clip-3", "clip-4"}
            )

    async def test_exactly_half_dropped_still_completes(self, tmp_path: Path, monkeypatch):
        """Boundary: the gate fires on a STRICT majority, so 2 of 4 completes."""
        result = await self._run(tmp_path, monkeypatch, n_specs=4, degenerate={"clip-3", "clip-4"})
        assert [c["clip_key"] for c in result["scenario_clips"]] == ["clip-1", "clip-2"]
        assert any("2 of 4" in w for w in result["capture_warnings"]), result["capture_warnings"]

    async def test_clean_full_capture_carries_no_drop_warning(self, tmp_path: Path, monkeypatch):
        """Vacuity guard: the warning must not appear when nothing was dropped."""
        result = await self._run(tmp_path, monkeypatch, n_specs=3, degenerate=set())
        assert len(result["scenario_clips"]) == 3
        assert not [w for w in result["capture_warnings"] if "dropp" in w.lower()]


class TestDiscardedTrackReachesCaptureWarnings:
    """WIRING: the discard must ride out of `execute()` on `capture_warnings`.

    Collecting the keys in `_persist_scenario_clips` and never joining them into
    `capture_warnings` would ship this fix INERT — `_track_coverage_warnings`
    goes silent the moment ONE sibling clip keeps its track, which is exactly the
    shape this covers (4 specs, 3 unprobeable, the 4th healthy).
    """

    async def _run(self, tmp_path: Path, monkeypatch, *, unmeasured: set[str]) -> dict[str, Any]:
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        video = package_dir / "rec.webm"
        video.write_bytes(b"a real recording")
        report = _nested_report_with_specs(4, video_path=video)
        # Every scenario recorded a track (they all interacted with the page).
        for spec in report["suites"][0]["suites"][0]["specs"]:
            spec["tests"][0]["results"][0]["attachments"].append(_body_attachment(_TRACK))

        async def _probe(path: Path) -> tuple[float, int, int, bool, bool]:
            # Unmeasured dims, but a NON-ZERO duration: `_is_degenerate` keeps
            # such a clip, so it publishes — track-less — rather than dropping.
            if any(path.name.startswith(f"{key}.") for key in unmeasured):
                return 1200.0, 0, 0, False, False
            return 1200.0, 1280, 720, False, True

        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _probe)

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            return await TestZeroFootageGate()._execute(engine, tmp_path, package_dir)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_discarded_tracks_reach_the_caller(self, tmp_path: Path, monkeypatch):
        result = await self._run(tmp_path, monkeypatch, unmeasured={"clip-1", "clip-2", "clip-3"})
        assert len(result["scenario_clips"]) == 4, "all four clips still publish"
        # The surviving 4th track is what silences `_track_coverage_warnings`.
        assert result["scenario_clips"][3]["cursor_track"]
        assert not [
            w for w in result["capture_warnings"] if "camera movement OFF" in w and "EMPTY" in w
        ]
        discard = [w for w in result["capture_warnings"] if "DISCARDED" in w]
        assert discard, result["capture_warnings"]
        assert "3 of 4 published clip(s)" in discard[0]
        assert "clip-1, clip-2, clip-3" in discard[0]

    async def test_clean_capture_carries_no_discard_warning(self, tmp_path: Path, monkeypatch):
        """Vacuity guard: every track normalized ⇒ nothing reported."""
        result = await self._run(tmp_path, monkeypatch, unmeasured=set())
        assert len(result["scenario_clips"]) == 4
        assert not [w for w in result["capture_warnings"] if "DISCARDED" in w]


class TestNodeToolTimeoutNamesTheFailure:
    """#2. On the 120s ``npm``/``bddgen`` timeout the captured output was
    DISCARDED and a bare ``TimeoutError`` re-raised; ``routes`` stored
    ``"TimeoutError: "`` and ``execute()``'s ``finally`` rmtree'd the package dir,
    so the evidence was gone. Same ``<no error>`` class #93/#97 fixed on the
    Playwright path with ``_read_captured_stdio``; never applied here.
    """

    @staticmethod
    def _package(tmp_path: Path, *, bdd: bool) -> Path:
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        deps = '"devDependencies":{"playwright-bdd":"^7"}' if bdd else '"devDependencies":{}'
        (package_dir / "package.json").write_text(f'{{"name":"x","version":"1.0.0",{deps}}}')
        return package_dir

    @staticmethod
    def _stub_hanging_proc(monkeypatch, *, stdout: bytes, stderr: bytes) -> None:
        """The tool writes diagnostics, then hangs until the wall budget kills it."""

        async def _stub_create_subprocess_exec(*_args: Any, **kwargs: Any) -> Any:
            for stream, payload in (("stdout", stdout), ("stderr", stderr)):
                sink = kwargs.get(stream)
                if hasattr(sink, "write"):
                    sink.write(payload)
                    sink.flush()

            class _Proc:
                pid = 4242
                returncode = None

                async def communicate(self) -> tuple[bytes, bytes]:
                    await asyncio.sleep(3600)
                    raise AssertionError("unreachable")

                async def wait(self) -> int:
                    await asyncio.sleep(3600)
                    raise AssertionError("unreachable")

            return _Proc()

        monkeypatch.setattr("asyncio.create_subprocess_exec", _stub_create_subprocess_exec)
        import src.services.execution_engine as mod

        monkeypatch.setattr(mod.os, "getpgid", lambda _pid: 0)
        monkeypatch.setattr(mod.asyncio, "wait_for", _instant_timeout_wait_for)

    async def test_npm_install_timeout_carries_the_captured_output(
        self, tmp_path: Path, monkeypatch
    ):
        engine = ExecutionEngine()
        self._stub_hanging_proc(
            monkeypatch,
            stdout=b"npm warn deprecated foo@1\n",
            stderr=b"npm error network request to https://registry.internal failed, ETIMEDOUT\n",
        )
        with pytest.raises(Exception) as exc:
            await engine._install_dependencies(self._package(tmp_path, bdd=False))
        message = str(exc.value)
        assert "ETIMEDOUT" in message, message
        assert "npm warn deprecated" in message, message

    async def test_bddgen_timeout_carries_the_captured_output(self, tmp_path: Path, monkeypatch):
        engine = ExecutionEngine()
        self._stub_hanging_proc(
            monkeypatch,
            stdout=b"resolving step definitions\n",
            stderr=b"Error: cannot find module 'playwright-bdd/dist/gen'\n",
        )
        with pytest.raises(Exception) as exc:
            await engine._run_bddgen(self._package(tmp_path, bdd=True))
        message = str(exc.value)
        assert "cannot find module" in message, message
        assert "resolving step definitions" in message, message


async def _instant_timeout_wait_for(awaitable: Any, timeout: float | None = None) -> Any:
    """``asyncio.wait_for`` stand-in that fires the timeout immediately.

    Keeps the 120s budget out of the test's wall clock while still driving the
    REAL ``except TimeoutError`` branch (and cancelling the awaitable, as the
    real one does).
    """
    task = asyncio.ensure_future(awaitable)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    raise TimeoutError


class TestTraceAndScreenshotOptionsAreNotSilentNoOps:
    """#3. ``screenshot_on_failure`` / ``trace_on_failure`` were referenced
    NOWHERE in ``src/``: ``trace_on_failure=False`` still recorded traces
    (``trace:'on-first-retry'`` is hardcoded in the bdd-generated config) and
    ``_store_report_tree`` uploaded them to the SHARED object store.
    """

    @staticmethod
    def _capture_cmd(engine: ExecutionEngine, monkeypatch, tmp_path: Path) -> dict[str, Any]:
        ctx = TestWallBudgetOverride._stub_run(engine, monkeypatch, tmp_path)
        captured: dict[str, Any] = ctx["captured"]
        real = asyncio.create_subprocess_exec

        async def _spy(*args: Any, **kwargs: Any) -> Any:
            captured["cmd"] = list(args)
            return await real(*args, **kwargs)

        monkeypatch.setattr("asyncio.create_subprocess_exec", _spy)
        return ctx

    async def test_trace_off_reaches_the_playwright_command(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        ctx = self._capture_cmd(engine, monkeypatch, tmp_path)
        request = TestExecutionRequest(
            bdd_job_id="j", target_url="https://example.com", trace_on_failure=False
        )
        await engine._run_tests(ctx["package_dir"], request)
        cmd = ctx["captured"]["cmd"]
        assert "--trace" in cmd, cmd
        assert cmd[cmd.index("--trace") + 1] == "off", cmd

    async def test_trace_on_is_still_requested_explicitly(self, tmp_path, monkeypatch):
        engine = ExecutionEngine()
        ctx = self._capture_cmd(engine, monkeypatch, tmp_path)
        request = TestExecutionRequest(
            bdd_job_id="j", target_url="https://example.com", trace_on_failure=True
        )
        await engine._run_tests(ctx["package_dir"], request)
        cmd = ctx["captured"]["cmd"]
        assert cmd[cmd.index("--trace") + 1] == "on-first-retry", cmd

    async def test_screenshot_option_is_declared_unimplemented_to_the_caller(
        self, tmp_path: Path, monkeypatch
    ):
        """No Playwright CLI lever exists for ``screenshot`` and the config is
        generated by another service, so this one cannot be wired from here. It
        must at least tell the caller it was ignored, on the response."""
        from src.config import get_settings

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        video = package_dir / "rec.webm"
        video.write_bytes(b"a real recording")
        report = _nested_report_with_specs(1, video_path=video)

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict[str, Any], bool]:
            return report, False

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        TestZeroFootageGate._healthy_probe(monkeypatch)

        from tests.conftest import make_local_storage

        engine._storage = make_local_storage(tmp_path / "store")
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            screenshot_on_failure=False,
        )
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            result = await engine.execute(
                request,
                persist_dir=tmp_path / "job-screenshot",
                public_artifact_base="/api/test-runner/job-screenshot/artifacts",
                work_dir=package_dir,
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True
        assert any("screenshot_on_failure" in w for w in result["capture_warnings"]), result[
            "capture_warnings"
        ]

    async def test_no_warning_when_the_caller_leaves_the_default(self, tmp_path: Path, monkeypatch):
        """Vacuity guard: the default request must not emit the notice."""
        assert (
            ExecutionEngine._unimplemented_option_warnings(
                TestExecutionRequest(bdd_job_id="j", target_url="https://example.com")
            )
            == []
        )


class TestSoftFailDemotionSurvivesRetries:
    """#4. Demotion gated on ``status == "passed"``. With ``retries >= 1`` a
    fail-then-pass carrying a soft-failed selector summarizes as ``"flaky"``,
    escapes demotion, is counted under ``flaky`` instead of ``failed``, and the
    clip ships ``status="flaky"``.
    """

    @staticmethod
    def _flaky_soft_fail_report(video: Path | None) -> dict[str, Any]:
        attachments = [{"name": "video", "path": str(video)}] if video else []
        return {
            "suites": [
                {
                    "title": "Login.feature",
                    "specs": [],
                    "suites": [
                        {
                            "title": "Login",
                            "specs": [
                                {
                                    "title": "Successful login",
                                    "tags": ["@clip-1"],
                                    "tests": [
                                        {
                                            "annotations": [
                                                {
                                                    "type": "soft-fail",
                                                    "description": 'getByRole("button", '
                                                    '{ name: "Save" }) never appeared',
                                                }
                                            ],
                                            "results": [
                                                {
                                                    "status": "failed",
                                                    "duration": 900,
                                                    "attachments": [],
                                                },
                                                {
                                                    "status": "passed",
                                                    "duration": 800,
                                                    "attachments": attachments,
                                                },
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "suites": [],
                        }
                    ],
                }
            ]
        }

    def test_flaky_with_a_soft_fail_is_demoted_in_the_summary(self, tmp_path: Path):
        engine = ExecutionEngine()
        parsed = engine._parse_results(self._flaky_soft_fail_report(None), tmp_path)
        assert parsed["summary"]["failed"] == 1, parsed["summary"]
        assert parsed["summary"]["flaky"] == 0, parsed["summary"]
        assert parsed["summary"]["pass_rate"] == 0.0
        assert "never appeared" in parsed["test_results"][0]["error"]

    def test_flaky_with_a_soft_fail_is_demoted_on_the_clip(self, tmp_path: Path):
        video = tmp_path / "rec.webm"
        video.write_bytes(b"v")
        specs = ExecutionEngine()._collect_clip_specs(self._flaky_soft_fail_report(video))
        assert [s["status"] for s in specs] == ["failed"], specs

    def test_a_clean_flaky_test_is_still_flaky(self, tmp_path: Path):
        """Vacuity guard: demotion must key on the soft-fail annotation, not on
        the status being non-``passed``."""
        report = self._flaky_soft_fail_report(None)
        report["suites"][0]["suites"][0]["specs"][0]["tests"][0]["annotations"] = []
        parsed = ExecutionEngine()._parse_results(report, tmp_path)
        assert parsed["summary"]["flaky"] == 1, parsed["summary"]
        assert parsed["summary"]["failed"] == 0, parsed["summary"]


class TestASignalKilledRunIsVisible:
    """A process group killed by signal read as a completely clean run.

    `returncode` was interpolated into ONE error string and used in NO decision
    anywhere. So a run that `_wait_for_report_or_exit` SIGKILLed — which it does
    as soon as `results.json` looks stable, aimed by its own docstring at "a stuck
    ffmpeg video muxer" — reported success as long as the partial `results.json`
    happened to parse.

    A NEGATIVE returncode is categorically different from a non-zero one:
    positive means "some tests failed", which is normal here; negative means
    killed by signal N, which means the harvest may be truncated. This does not
    fail the job; it makes the condition visible in `capture_warnings`.
    """

    @staticmethod
    def _warn(returncode: int) -> str | None:
        """The exact predicate the engine applies, exercised directly."""
        if returncode < 0:
            return (
                f"Playwright process group was KILLED by signal {-returncode} rather than exiting."
            )
        return None

    def test_a_sigkilled_group_is_named_with_its_signal(self) -> None:
        warning = self._warn(-9)

        assert warning is not None
        assert "signal 9" in warning, "the operator needs to know WHICH signal"

    def test_a_sigterm_is_also_caught(self) -> None:
        assert "signal 15" in (self._warn(-15) or "")

    def test_a_normal_test_failure_is_not_warned_about(self) -> None:
        """Playwright exits non-zero when tests fail. That is expected here and
        must not produce a truncation warning — crying wolf on every failing
        suite would make the real signal unreadable."""
        assert self._warn(1) is None
        assert self._warn(0) is None

    def test_the_engine_actually_applies_this_predicate(self) -> None:
        """Guard the wiring, not just the arithmetic.

        The predicate above is a restatement; this asserts the engine's own source
        carries it, so deleting the check cannot leave these tests green.
        """
        import inspect

        from src.services.execution_engine import ExecutionEngine

        src = inspect.getsource(ExecutionEngine)
        assert "if returncode < 0:" in src, (
            "the engine no longer distinguishes a signal death from a test failure"
        )
        assert "_signal_kill_warning" in src, (
            "the warning is no longer carried to the capture_warnings assembly"
        )


def _feature_source(clip_numbers: list[int]) -> str:
    """A Gherkin source shaped the way bdd_playwright_service stamps one."""
    body = "\n".join(
        f"\n  @clip-{n}\n  Scenario: Step {n}\n    Given the app is open\n" for n in clip_numbers
    )
    return f"Feature: Login\n{body}"


class TestPartialSalvageIsReportedAsPartial:
    """A wall-timeout salvage must say "N of M", never silently redefine M.

    `_salvage_raw_from_disk` synthesizes one spec per SURVIVING clip, so with 12
    scenarios and 9 `.webm` on disk the report declares 9. Every report-derived
    gate then compares 9 against 9: `_guard_zero_footage` returns on the first
    clip, `_guard_partial_footage` reads only clips that RECORDED then got
    dropped, and `_footage_yield_warnings` sees no shortfall. The job lands
    `completed` and "12 scenarios, 9 clips" is indistinguishable from
    "9 scenarios" — while the orchestrator narrates clips 10-12 over footage that
    never arrives.
    """

    @staticmethod
    def _stage(
        engine: ExecutionEngine,
        monkeypatch,
        tmp_path: Path,
        *,
        declared: list[int],
        recorded: list[int],
        write_feature: bool = True,
    ) -> Path:
        package_dir = TestZeroClipSalvageFailsTheRun._stub_timeout_run(
            engine, monkeypatch, tmp_path
        )
        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        TestZeroFootageGate._healthy_probe(monkeypatch)
        if write_feature:
            features = package_dir / "features"
            features.mkdir(parents=True, exist_ok=True)
            (features / "login.feature").write_text(_feature_source(declared), encoding="utf-8")
        for n in recorded:
            clip_dir = package_dir / "test-results" / f"login-clip-{n}-chromium"
            clip_dir.mkdir(parents=True)
            (clip_dir / "video.webm").write_bytes(b"a real recording")
        return package_dir

    async def _run(
        self, engine: ExecutionEngine, tmp_path: Path, package_dir: Path, **kwargs: Any
    ) -> dict[str, Any]:
        from src.config import get_settings

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            return await TestZeroFootageGate()._execute(engine, tmp_path, package_dir, **kwargs)
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_twelve_declared_nine_salvaged_reports_the_shortfall(
        self, tmp_path: Path, monkeypatch
    ):
        """THE HEADLINE. 12 declared, 9 salvaged ⇒ the run still completes (the 9
        good clips are worth more to the orchestrator than a permanent failure
        under the same Idempotency-Key), but the gap is NAMED."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 13)),
            recorded=list(range(1, 10)),
        )

        result = await self._run(engine, tmp_path, package_dir)

        assert len(result["scenario_clips"]) == 9
        shortfall = [w for w in result["capture_warnings"] if "PARTIAL SALVAGE" in w]
        assert shortfall, (
            "a 9-of-12 salvage reported as a complete run: "
            f"capture_warnings={result['capture_warnings']}"
        )
        message = shortfall[0]
        assert "9 of 12" in message
        for missing in ("clip-10", "clip-11", "clip-12"):
            assert missing in message, f"{missing} never recorded but is not named"
        assert "clip-1," not in message, "a clip that DID record must not be named as missing"

    async def test_a_clean_twelve_of_twelve_run_is_unchanged(self, tmp_path: Path, monkeypatch):
        """The regression guard that matters most: a full harvest must stay
        silent. A gate that warns here would fire on every healthy capture."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 13)),
            recorded=list(range(1, 13)),
        )

        result = await self._run(engine, tmp_path, package_dir)

        assert len(result["scenario_clips"]) == 12
        assert [w for w in result["capture_warnings"] if "PARTIAL SALVAGE" in w] == []

    async def test_a_genuine_nine_scenario_package_delivering_nine_is_silent(
        self, tmp_path: Path, monkeypatch
    ):
        """A SHORT package is not a broken one. 9 declared, 9 delivered must not
        borrow a denominator from anywhere else."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 10)),
            recorded=list(range(1, 10)),
        )

        result = await self._run(engine, tmp_path, package_dir)

        assert len(result["scenario_clips"]) == 9
        assert [w for w in result["capture_warnings"] if "PARTIAL SALVAGE" in w] == []

    async def test_zero_footage_on_timeout_still_hits_the_zero_footage_gate(
        self, tmp_path: Path, monkeypatch
    ):
        """`_guard_zero_footage` owns the empty harvest and must keep owning it —
        the new gate must not pre-empt it with a softer verdict."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine, monkeypatch, tmp_path, declared=list(range(1, 13)), recorded=[]
        )

        with pytest.raises(RuntimeError, match="no usable footage"):
            await self._run(engine, tmp_path, package_dir)

    async def test_a_majority_missing_fails_the_job(self, tmp_path: Path, monkeypatch):
        """Same policy as `_guard_partial_footage`: a strict majority missing is
        the zero-footage case in all but name."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine, monkeypatch, tmp_path, declared=list(range(1, 13)), recorded=[1, 2]
        )

        with pytest.raises(RuntimeError, match="most of the capture"):
            await self._run(engine, tmp_path, package_dir)

    async def test_an_unmeasurable_salvage_says_so_rather_than_going_quiet(
        self, tmp_path: Path, monkeypatch
    ):
        """No @clip tags in the package ⇒ the completeness of the salvage is
        UNMEASURED, which is not the same as fine. Fail-open on a measurement gap
        is the defect class this whole change exists to close."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=[],
            recorded=[1, 2],
            write_feature=False,
        )

        result = await self._run(engine, tmp_path, package_dir)

        assert len(result["scenario_clips"]) == 2
        assert [w for w in result["capture_warnings"] if "UNKNOWN completeness" in w]

    async def test_expect_recording_false_opts_out(self, tmp_path: Path, monkeypatch):
        """A non-recording package has no footage to lose — same escape hatch the
        two sibling gates honour."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 13)),
            recorded=list(range(1, 10)),
        )

        result = await self._run(engine, tmp_path, package_dir, expect_recording=False)

        assert [w for w in result["capture_warnings"] if "SALVAGE" in w] == []

    async def _run_with_grep(
        self,
        engine: ExecutionEngine,
        tmp_path: Path,
        package_dir: Path,
        *,
        grep_pattern: str,
    ) -> dict[str, Any]:
        from src.config import get_settings

        get_settings().execution.convert_clips_to_mp4 = False
        try:
            return await TestZeroFootageGate()._execute(
                engine, tmp_path, package_dir, grep_pattern=grep_pattern
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

    async def test_a_filtered_run_that_records_every_selected_clip_is_silent(
        self, tmp_path: Path, monkeypatch
    ):
        """5 declared, grep excludes 2 (clip-4, clip-5) ⇒ the denominator is the 3
        SELECTED clips, not all 5. Recording exactly those 3 must not be reported
        as a 3-of-5 shortfall — the other 2 were never supposed to run."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 6)),
            recorded=[1, 2, 3],
        )

        result = await self._run_with_grep(
            engine, tmp_path, package_dir, grep_pattern=r"@clip-[123]\b"
        )

        assert len(result["scenario_clips"]) == 3
        assert [w for w in result["capture_warnings"] if "PARTIAL SALVAGE" in w] == []

    async def test_a_filtered_run_that_drops_a_selected_clip_still_raises(
        self, tmp_path: Path, monkeypatch
    ):
        """Same filter (clip-1..3 selected, clip-4/5 excluded), but only 2 of the 3
        SELECTED clips actually recorded ⇒ the shortfall check must still catch a
        genuine recording failure among the clips the filter kept."""
        engine = ExecutionEngine()
        package_dir = self._stage(
            engine,
            monkeypatch,
            tmp_path,
            declared=list(range(1, 6)),
            recorded=[1, 2],
        )

        result = await self._run_with_grep(
            engine, tmp_path, package_dir, grep_pattern=r"@clip-[123]\b"
        )

        shortfall = [w for w in result["capture_warnings"] if "PARTIAL SALVAGE" in w]
        assert shortfall, (
            "clip-3 was selected by the filter and never recorded, but no "
            f"shortfall was raised: capture_warnings={result['capture_warnings']}"
        )
        message = shortfall[0]
        assert "2 of 3" in message
        assert "clip-3" in message
        assert "clip-4" not in message, "an excluded clip must not be named as missing"
        assert "clip-5" not in message, "an excluded clip must not be named as missing"


class TestDeclaredClipKeys:
    """`_declared_clip_keys` — the one denominator the salvaged report cannot
    influence, because it is read from the package sources."""

    def test_reads_tags_from_nested_feature_files(self, tmp_path: Path) -> None:
        (tmp_path / "a").mkdir()
        (tmp_path / "a" / "one.feature").write_text("@clip-1\nScenario: x\n")
        (tmp_path / "a" / "two.feature").write_text("@smoke @clip-2\nScenario: y\n")
        assert ExecutionEngine._declared_clip_keys(tmp_path) == {"clip-1", "clip-2"}

    def test_ignores_generated_and_vendored_trees(self, tmp_path: Path) -> None:
        """`.features-gen` is what bddgen COMPILES from the sources, so counting
        it double-counts; `node_modules` carries other people's fixtures."""
        for bad in (".features-gen", "node_modules"):
            d = tmp_path / bad / "nested"
            d.mkdir(parents=True)
            (d / "x.feature").write_text("@clip-99\nScenario: x\n")
        (tmp_path / "real.feature").write_text("@clip-1\nScenario: x\n")
        assert ExecutionEngine._declared_clip_keys(tmp_path) == {"clip-1"}

    def test_a_bare_clip_word_in_prose_is_not_a_declaration(self, tmp_path: Path) -> None:
        """Without the leading `@` it is step text, not a tag. Counting it would
        inflate the denominator and invent a shortfall."""
        (tmp_path / "x.feature").write_text(
            "@clip-1\nScenario: x\n  Given the user opens clip-7 in the player\n"
        )
        assert ExecutionEngine._declared_clip_keys(tmp_path) == {"clip-1"}

    def test_no_feature_files_is_an_empty_set_not_an_error(self, tmp_path: Path) -> None:
        assert ExecutionEngine._declared_clip_keys(tmp_path) == set()

    def test_no_grep_pattern_is_unchanged_from_before(self, tmp_path: Path) -> None:
        """Backward compatibility: omitting `grep_pattern` must behave exactly
        like the pre-fix, whole-file scan."""
        (tmp_path / "x.feature").write_text(
            "Feature: Demo\n\n  @clip-1\n  Scenario: one\n    Given a step\n\n"
            "  @clip-2\n  Scenario: two\n    Given a step\n"
        )
        assert ExecutionEngine._declared_clip_keys(tmp_path) == {"clip-1", "clip-2"}

    def test_grep_pattern_excludes_a_scenario_the_filter_did_not_select(
        self, tmp_path: Path
    ) -> None:
        (tmp_path / "x.feature").write_text(
            "Feature: Demo\n\n  @clip-1\n  Scenario: one\n    Given a step\n\n"
            "  @clip-2\n  Scenario: two\n    Given a step\n\n"
            "  @clip-3\n  Scenario: three\n    Given a step\n"
        )
        assert ExecutionEngine._declared_clip_keys(tmp_path, r"@clip-[12]\b") == {
            "clip-1",
            "clip-2",
        }

    def test_grep_pattern_clip_1_does_not_also_select_clip_10(self, tmp_path: Path) -> None:
        """The substring hazard: an UNANCHORED `@clip-1` would also match
        `@clip-10`. A correctly anchored pattern (word-boundary or `$`) must
        not."""
        (tmp_path / "x.feature").write_text(
            "Feature: Demo\n\n  @clip-1\n  Scenario: one\n    Given a step\n\n"
            "  @clip-10\n  Scenario: ten\n    Given a step\n"
        )
        assert ExecutionEngine._declared_clip_keys(tmp_path, r"@clip-1\b") == {"clip-1"}

    def test_feature_level_tags_apply_to_every_scenario(self, tmp_path: Path) -> None:
        """Gherkin ``Feature:``-level tags are inherited by every scenario in the
        file, so a filter on a feature-level tag must select all of them."""
        (tmp_path / "x.feature").write_text(
            "@smoke\nFeature: Demo\n\n  @clip-1\n  Scenario: one\n    Given a step\n\n"
            "  @clip-2\n  Scenario: two\n    Given a step\n"
        )
        assert ExecutionEngine._declared_clip_keys(tmp_path, r"@smoke\b") == {
            "clip-1",
            "clip-2",
        }

    def test_grep_pattern_matching_nothing_yields_empty_declared_set(self, tmp_path: Path) -> None:
        (tmp_path / "x.feature").write_text("@clip-1\nScenario: x\n")
        assert ExecutionEngine._declared_clip_keys(tmp_path, r"@clip-99\b") == set()


# Fake credentials (not real secrets).
_FAKE_KC = "kc-pw-0001"  # pragma: allowlist secret
_FAKE_TW = "tw-pw-0002"  # pragma: allowlist secret


class TestDotenvBeforeInstall:
    """Item 4 (merqi showcase spike): `execute()` installed dependencies BEFORE
    `_configure_target` injected `dotenv` into package.json, so an auth_config
    package that did not itself list dotenv died at `import 'dotenv/config'`."""

    async def test_dotenv_is_injected_before_install_and_install_sees_it(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        (package_dir / "package.json").write_text('{"name":"x","version":"1.0.0"}')
        (package_dir / "package-lock.json").write_text("{}")
        (package_dir / "playwright.config.ts").write_text("export default {};")
        seen: dict[str, Any] = {}

        async def _resolve(_request: Any, _tmp: Path, _token: str | None) -> Path:
            return package_dir

        async def _run_node_tool(cmd: list[str], pkg: Path, **_k: Any) -> None:
            seen["cmd"] = cmd
            seen["deps"] = json.loads((pkg / "package.json").read_text()).get("dependencies", {})
            raise RuntimeError("stop after install")

        monkeypatch.setattr(engine, "_resolve_package", _resolve)
        monkeypatch.setattr(engine, "_run_node_tool", _run_node_tool)
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            auth_config={"login_url": "https://example.com/login", "password": _FAKE_KC},
        )
        with pytest.raises(RuntimeError, match="stop after install"):
            await engine.execute(request, work_dir=package_dir)
        assert "dotenv" in seen["deps"]
        # The shipped lockfile no longer matches package.json: `npm ci` would exit 1.
        assert seen["cmd"][:2] == ["npm", "install"]

    async def test_untouched_lockfile_still_uses_npm_ci(self, tmp_path: Path, monkeypatch) -> None:
        engine = ExecutionEngine()
        (tmp_path / "package.json").write_text('{"dependencies":{"dotenv":"^16"}}')
        (tmp_path / "package-lock.json").write_text("{}")
        seen: list[list[str]] = []

        async def _run_node_tool(cmd: list[str], _pkg: Path, **_k: Any) -> None:
            seen.append(cmd)

        monkeypatch.setattr(engine, "_run_node_tool", _run_node_tool)
        await engine._install_dependencies(tmp_path)
        await engine._install_dependencies(tmp_path, lockfile_stale=True)
        assert [c[1] for c in seen] == ["ci", "install"]


class TestMultipleLogins:
    """Item 5: several logins per run, keyed by origin — not '|'-joined."""

    @staticmethod
    def _request() -> TestExecutionRequest:
        return TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://auth.example.com/strategy",
            auth_config=[
                {
                    "login_url": "https://example.com/login",
                    "username": "tw@example.com",
                    "password": _FAKE_TW,
                },
                {
                    "origin": "https://auth.example.com",
                    "login_url": "https://auth.example.com/strategy",
                    "username": "kc-user",
                    "password": _FAKE_KC,
                    "sso_button": "Keycloak",
                },
            ],
        )

    def test_logins_file_keyed_by_origin_and_primary_follows_target(self, tmp_path: Path) -> None:
        ExecutionEngine()._configure_target(tmp_path, self._request())
        env = (tmp_path / ".env").read_text()
        # AUTH_* = the login for target_url's origin, so a single-login scaffold still works.
        assert "AUTH_USERNAME=kc-user" in env and f"AUTH_PASSWORD={_FAKE_KC}" in env
        assert "|" not in env
        assert "AUTH_LOGINS_FILE=.auth/logins.json" in env
        logins_file = tmp_path / ".auth" / "logins.json"
        assert logins_file.stat().st_mode & 0o777 == 0o600
        logins = json.loads(logins_file.read_text())
        assert set(logins) == {"https://example.com", "https://auth.example.com"}
        assert logins["https://example.com"]["password"] == _FAKE_TW
        assert logins["https://auth.example.com"]["sso_button"] == "Keycloak"

    def test_every_login_url_is_ssrf_guarded(self, tmp_path: Path) -> None:
        req = self._request()
        assert isinstance(req.auth_config, list)
        req.auth_config[0]["login_url"] = "https://internal-thing.example.com/login"
        with pytest.raises(ValueError):
            ExecutionEngine()._configure_target(tmp_path, req)

    def test_duplicate_origin_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="twice"):
            TestExecutionRequest(
                bdd_job_id="j",
                target_url="https://example.com",
                auth_config=[
                    {"login_url": "https://example.com/a"},
                    {"origin": "https://EXAMPLE.com"},
                ],
            )

    def test_stored_job_config_carries_no_secret(self) -> None:
        from src.api.schemas import without_auth_secrets

        dumped = self._request().model_dump(exclude={"target_url", "bdd_job_id", "package_url"})
        stored = without_auth_secrets(dumped)
        text = json.dumps(stored)
        assert _FAKE_KC not in text and _FAKE_TW not in text
        logins = stored["auth_config"]
        assert isinstance(logins, list)
        assert logins[1]["username"] == "kc-user"
        single = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            auth_config={"login_url": "https://example.com/login", "password": _FAKE_KC},
        ).model_dump()
        assert without_auth_secrets(single)["auth_config"] == {
            "login_url": "https://example.com/login"
        }


# ── Login runs: parser differentials + nothing shipped carries the password ──────

_PW = "Hunter2-login-secret"  # pragma: allowlist secret
# The re-audit repro: urlparse reads host twenty.example; Chromium (WHATWG reads "\"
# as "/") navigates to attacker.example.
_BACKSLASH = "https://attacker.example\\@twenty.example/"


class TestLoginUrlParserDifferential:
    @pytest.mark.parametrize(
        "url",
        [
            _BACKSLASH,
            "https://attacker.example%5C@twenty.example/",
            "https://attacker.example%40twenty.example/",
            "https://user@twenty.example/",
            "https://twenty.example /",
            "http://twenty.example/",
        ],
    )
    def test_login_url_is_rejected_at_the_schema(self, url: str) -> None:
        with pytest.raises(ValueError):
            TestExecutionRequest(
                bdd_job_id="j",
                target_url="https://twenty.example/",
                auth_config={"login_url": url, "password": _PW},
            )

    def test_login_bound_target_url_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            TestExecutionRequest(
                bdd_job_id="j", target_url=_BACKSLASH, auth_config={"password": _PW}
            )

    def test_ssrf_check_rejects_the_differential_before_parsing(self) -> None:
        # Pre-fix, urlparse saw twenty.example and the address check passed.
        with pytest.raises(ValueError, match="backslash"):
            _validate_url(_BACKSLASH)
        with pytest.raises(ValueError, match="userinfo"):
            _validate_url("https://attacker.example%40twenty.example/x.zip")

    def test_target_url_without_a_login_keeps_its_old_contract(self) -> None:
        req = TestExecutionRequest(bdd_job_id="j", target_url="http://example.com")
        assert req.target_url == "http://example.com"


class TestLoginRunShipsNoSecret:
    """No artifact, report, result field, error or log line of a login run may
    contain the password — Playwright titles `fill()` steps with the typed value."""

    @staticmethod
    def _report(package_dir: Path) -> dict[str, Any]:
        setup_video = package_dir / "setup.webm"
        setup_video.write_bytes(b"setup-video")
        scenario_video = package_dir / "scenario.webm"
        scenario_video.write_bytes(b"scenario-video")
        step = {"title": f'Fill "{_PW}" locator("#password")', "duration": 5}
        return {
            "suites": [
                {
                    "title": "auth.setup.ts",
                    "file": "auth.setup.ts",
                    "specs": [
                        {
                            "title": "authenticate",
                            "file": "auth.setup.ts",
                            "tests": [
                                {
                                    "projectName": "setup",
                                    "results": [
                                        {
                                            "status": "passed",
                                            "duration": 900,
                                            "steps": [step],
                                            "attachments": [
                                                {"name": "video", "path": str(setup_video)}
                                            ],
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                    "suites": [],
                },
                {
                    "title": "Demo.feature",
                    "specs": [],
                    "suites": [
                        {
                            "title": "Demo",
                            "specs": [
                                {
                                    "title": "Strategy studio",
                                    "tags": ["@clip-1"],
                                    "tests": [
                                        {
                                            "projectName": "chromium",
                                            "results": [
                                                {
                                                    "status": "failed",
                                                    "duration": 900,
                                                    "steps": [step],
                                                    "error": {
                                                        "message": f"fill({_PW!r}) timed out"
                                                    },
                                                    "attachments": [
                                                        {
                                                            "name": "video",
                                                            "path": str(scenario_video),
                                                        }
                                                    ],
                                                }
                                            ],
                                        }
                                    ],
                                }
                            ],
                            "suites": [],
                        }
                    ],
                },
            ]
        }

    async def test_no_artifact_or_result_contains_the_password(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        from src.config import get_settings
        from tests.conftest import make_local_storage

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        report_dir = package_dir / "playwright-report"
        report_dir.mkdir()
        (report_dir / "index.html").write_text(f"<html>Fill &quot;{_PW}&quot;</html>")
        report = self._report(package_dir)
        (package_dir / "results.json").write_text(json.dumps(report))
        seen: dict[str, Any] = {}

        async def _run_tests(_pkg: Path, request: TestExecutionRequest) -> tuple[dict, bool]:
            seen["trace_on_failure"] = request.trace_on_failure
            logger_ = logging.getLogger("src.services.execution_engine")
            logger_.error("playwright stderr: fill(%r) failed", _PW)
            return report, False

        async def _probe(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 5000.0, 2560, 1440, False, True

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _probe)
        store_root = tmp_path / "store"
        engine._storage = make_local_storage(store_root)
        caplog.set_level(logging.DEBUG)
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            auth_config={"login_url": "https://example.com/login", "password": _PW},
        )
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            result = await engine.execute(
                request,
                persist_dir=tmp_path / "job-login",
                public_artifact_base="/api/test-runner/job-login/artifacts",
                work_dir=package_dir,
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True

        assert seen["trace_on_failure"] is False  # a trace records fill()'s argument
        assert _PW not in json.dumps(result, default=str)
        assert not result.get("report_url")  # the HTML report is withheld
        stored = [p for p in store_root.rglob("*") if p.is_file()]
        assert not any("playwright-report" in p.as_posix() for p in stored)
        for path in stored:
            assert _PW.encode() not in path.read_bytes(), path
            assert b"setup-video" not in path.read_bytes(), path  # setup never harvested
        assert [c["clip_key"] for c in result["scenario_clips"]] == ["clip-1"]
        assert all("authenticate" not in str(t) for t in result["test_results"])
        assert _PW not in caplog.text

    async def test_a_raised_error_never_carries_the_password(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict, bool]:
            raise RuntimeError(f"Playwright produced no parseable output: fill('{_PW}')")

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            auth_config={"login_url": "https://example.com/login", "password": _PW},
        )
        with pytest.raises(RuntimeError) as exc:
            await engine.execute(request, work_dir=package_dir)
        assert _PW not in str(exc.value)
        assert "[redacted]" in str(exc.value)

    async def test_salvage_without_results_json_ships_no_footage(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """No results.json ⇒ the report is a salvage nobody can split into setup vs
        scenario footage ⇒ a login run uploads no clip, video or screenshot."""
        from src.config import get_settings
        from tests.conftest import make_local_storage

        engine = ExecutionEngine()
        package_dir = tmp_path / "pkg"
        package_dir.mkdir()
        report = self._report(package_dir)  # what a salvage would synthesise
        assert not (package_dir / "results.json").exists()

        async def _run_tests(*_a: Any, **_k: Any) -> tuple[dict, bool]:
            return report, True

        async def _probe(_path: Path) -> tuple[float, int, int, bool, bool]:
            return 5000.0, 2560, 1440, False, True

        TestZeroFootageGate._stub_pipeline(engine, monkeypatch, package_dir)
        monkeypatch.setattr(engine, "_run_tests", _run_tests)
        monkeypatch.setattr("src.services.execution_engine._ffprobe_meta", _probe)
        store_root = tmp_path / "store"
        engine._storage = make_local_storage(store_root)
        request = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://example.com",
            auth_config={"login_url": "https://example.com/login", "password": _PW},
            expect_recording=False,
        )
        get_settings().execution.convert_clips_to_mp4 = False
        try:
            result = await engine.execute(
                request,
                persist_dir=tmp_path / "job-salvage",
                public_artifact_base="/api/test-runner/job-salvage/artifacts",
                work_dir=package_dir,
            )
        finally:
            get_settings().execution.convert_clips_to_mp4 = True
        assert result["scenario_clips"] == []
        assert [p for p in store_root.rglob("*") if p.is_file()] == []


_ENCODED_PW = 'p@ss "w\\ord"/+&é'  # pragma: allowlist secret


class TestSecretScrubberCoversEncodings:
    @pytest.mark.parametrize(
        "encoded",
        [
            pytest.param(_ENCODED_PW, id="raw"),
            pytest.param(repr(_ENCODED_PW)[1:-1], id="repr"),
            pytest.param(json.dumps(_ENCODED_PW)[1:-1], id="json"),
            pytest.param(quote(_ENCODED_PW, safe=""), id="urlencoded"),
            pytest.param(quote_plus(_ENCODED_PW), id="form-urlencoded"),
            pytest.param(base64.b64encode(_ENCODED_PW.encode()).decode(), id="base64"),
            pytest.param(
                base64.urlsafe_b64encode(_ENCODED_PW.encode()).decode().rstrip("="),
                id="base64url-unpadded",
            ),
        ],
    )
    def test_each_encoded_form_is_removed(self, encoded: str) -> None:
        from src.services.execution_engine import _scrub_obj, _secret_variants

        variants = _secret_variants([_ENCODED_PW])
        blob = {"error": f"step failed: {encoded}", "steps": [f"title {encoded}"]}
        scrubbed = json.dumps(_scrub_obj(blob, variants))
        assert encoded not in scrubbed
        assert "[redacted]" in scrubbed


class TestLoginOriginAgreement:
    def test_origin_and_login_url_must_agree(self) -> None:
        with pytest.raises(ValueError, match="same origin"):
            TestExecutionRequest(
                bdd_job_id="j",
                target_url="https://twenty.example/",
                auth_config={
                    "origin": "https://twenty.example",
                    "login_url": "https://attacker.example/login",
                },
            )

    def test_agreeing_origin_and_login_url_pass(self) -> None:
        req = TestExecutionRequest(
            bdd_job_id="j",
            target_url="https://twenty.example/",
            auth_config={
                "origin": "https://twenty.example",
                "login_url": "https://twenty.example:443/login",
            },
        )
        assert isinstance(req.auth_config, dict)


class TestSameOriginLabelledLogins:
    """Several logins on ONE origin, told apart by label (requester + gm on one app)."""

    _URL = "https://auth.example.com"

    def _logins(self) -> list[dict]:
        return [
            {"origin": self._URL, "label": "requester", "username": "req", "password": _FAKE_TW},
            {"origin": self._URL, "label": "gm", "username": "gm", "password": _FAKE_KC},
        ]

    def test_same_origin_with_labels_is_accepted_and_keyed_by_label(self, tmp_path: Path) -> None:
        req = TestExecutionRequest(bdd_job_id="j", target_url=self._URL, auth_config=self._logins())
        ExecutionEngine()._configure_target(tmp_path, req)
        logins = json.loads((tmp_path / ".auth" / "logins.json").read_text())
        assert set(logins) == {f"{self._URL}#requester", f"{self._URL}#gm"}
        assert logins[f"{self._URL}#gm"]["label"] == "gm"
        assert logins[f"{self._URL}#gm"]["origin"] == self._URL
        assert logins[f"{self._URL}#gm"]["password"] == _FAKE_KC
        assert logins[f"{self._URL}#requester"]["username"] == "req"

    def test_single_labelled_login_stays_keyed_by_origin(self, tmp_path: Path) -> None:
        other = {"origin": "https://example.com", "username": "o", "password": _FAKE_KC}
        req = TestExecutionRequest(
            bdd_job_id="j", target_url=self._URL, auth_config=[self._logins()[0], other]
        )
        ExecutionEngine()._configure_target(tmp_path, req)
        logins = json.loads((tmp_path / ".auth" / "logins.json").read_text())
        assert set(logins) == {self._URL, "https://example.com"}
        assert logins[self._URL]["label"] == "requester"

    def test_unlabelled_duplicate_origin_still_rejected(self) -> None:
        dup = [{"origin": self._URL, "username": "a"}, {"origin": self._URL, "username": "b"}]
        with pytest.raises(ValueError, match="label"):
            TestExecutionRequest(bdd_job_id="j", target_url=self._URL, auth_config=dup)

    def test_mixed_labelled_and_unlabelled_on_one_origin_rejected(self) -> None:
        mixed = [{"origin": self._URL, "label": "gm"}, {"origin": self._URL}]
        with pytest.raises(ValueError, match="needs a label"):
            TestExecutionRequest(bdd_job_id="j", target_url=self._URL, auth_config=mixed)

    def test_same_origin_same_label_rejected(self) -> None:
        dup = [{"origin": self._URL, "label": "gm"}, {"origin": self._URL, "label": "gm"}]
        with pytest.raises(ValueError, match="twice"):
            TestExecutionRequest(bdd_job_id="j", target_url=self._URL, auth_config=dup)

    def test_bad_label_slug_rejected(self) -> None:
        with pytest.raises(ValueError, match="label"):
            TestExecutionRequest(
                bdd_job_id="j",
                target_url=self._URL,
                auth_config=[{"origin": self._URL, "label": "Not A Slug"}],
            )
