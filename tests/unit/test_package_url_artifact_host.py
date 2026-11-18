"""The SSRF guard must still admit OUR OWN artifact store.

PR #75 closed a real hole (hostnames were never resolved, so any ``*.internal``
name walked straight through). Closing it by name, however, also rejected the
presigned MinIO URLs the orchestrator legitimately hands this service as
``package_url`` — the live showcase/authored recording path sends
``http://minio.internal:8389/w2v-orchestrator/artifacts/...?X-Amz-Signature=...``
on the *untrusted* branch (``allow_private=False``), so it hit the new
name-based block and the recording package could no longer be downloaded.

The fix trusts exactly one thing: the MinIO endpoint THIS service is already
configured to use (``STORAGE_MINIO_ENDPOINT``). It is not an open operator
allowlist — an attacker-supplied ``*.internal`` host that is not our artifact
store is still refused, and every other range check is untouched.
"""

from __future__ import annotations

import ipaddress
import socket

import pytest

from src.config import clear_settings_cache
from src.services.execution_engine import _validate_url

_PRESIGNED = (
    "http://minio.internal:8389/w2v-orchestrator/artifacts/video-packages/"
    "recording-package/specra-recording.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=svc-w2v-orchestrator%2F20260711%2Fus-east-1%2Fs3%2Faws4_request"
    "&X-Amz-Date=20260711T054703Z&X-Amz-Expires=604800"
    "&X-Amz-SignedHeaders=host&X-Amz-Signature=7359d4ef439a36ae0c922aee0806b04fd16420cc"
)


def _literal_only_getaddrinfo(host, port, *args, **kwargs):
    """Resolve IP literals to themselves; refuse to touch the network otherwise.

    Keeps the suite hermetic — a test that accidentally depends on real DNS
    fails loudly instead of quietly reaching out (or flaking in CI).
    """
    text = str(host).strip("[]")
    try:
        parsed = ipaddress.ip_address(text)
    except ValueError:
        raise AssertionError(f"test attempted real DNS resolution of {host!r}") from None
    family = socket.AF_INET6 if parsed.version == 6 else socket.AF_INET
    return [(family, socket.SOCK_STREAM, 6, "", (text, port or 0))]


@pytest.fixture(autouse=True)
def _hermetic_dns_and_config(monkeypatch: pytest.MonkeyPatch):
    """Point the service at the live artifact endpoint and block real DNS."""
    monkeypatch.setattr(socket, "getaddrinfo", _literal_only_getaddrinfo)
    monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "minio.internal:8389")
    clear_settings_cache()
    yield
    clear_settings_cache()


class TestOwnArtifactHostIsReachable:
    def test_presigned_package_url_at_our_minio_is_allowed(self):
        """The live showcase package_url must survive the SSRF guard."""
        _validate_url(_PRESIGNED)

    def test_plain_url_at_our_minio_is_allowed(self):
        _validate_url("http://minio.internal:8389/bucket/pkg.zip")

    def test_scheme_is_still_enforced_for_the_allowed_host(self):
        """Trusting the host must not trust arbitrary schemes."""
        with pytest.raises(ValueError, match="http or https"):
            _validate_url("file://minio.internal:8389/etc/passwd")


class TestTheAllowlistIsNarrow:
    """Trusting our artifact store must not reopen the estate."""

    def test_other_internal_hosts_are_still_blocked(self):
        for url in (
            "http://postgres.internal:5432/x",
            "http://registry.internal:5000/v2/_catalog",
            "http://minio-evil.internal:8389/pkg.zip",
            "http://bdd.svc.cluster.local/pkg.zip",
        ):
            with pytest.raises(ValueError):
                _validate_url(url)

    def test_tailnet_literals_are_still_blocked(self):
        """The CGNAT /10 stays closed — this is what #75 was for."""
        for ip in ("100.64.0.5", "100.64.0.6", "100.64.0.7", "100.64.0.1"):
            with pytest.raises(ValueError):
                _validate_url(f"http://{ip}/x")

    def test_public_addresses_still_pass(self):
        """The deny is the CIDR, not a ``startswith("100.")`` prefix match."""
        for ip in ("8.8.8.8", "1.1.1.1", "100.63.255.255", "100.128.0.0"):
            _validate_url(f"http://{ip}/x")

    def test_a_different_configured_endpoint_moves_the_trust(self, monkeypatch: pytest.MonkeyPatch):
        """The allowlist follows config — it is not a hardcoded 'minio.internal'."""
        monkeypatch.setenv("STORAGE_MINIO_ENDPOINT", "artifacts.internal:9000")
        clear_settings_cache()
        _validate_url("http://artifacts.internal:9000/bucket/pkg.zip")
        with pytest.raises(ValueError):
            _validate_url("http://minio.internal:8389/bucket/pkg.zip")
