"""CI-enforced invariant: the RUNTIME image stage must bake ffmpeg/ffprobe.

This gate exists because the absence of these two binaries is *invisible* at
runtime. ``_convert_to_mp4`` and ``_ffprobe_meta`` are both best-effort by
design — a missing binary makes them degrade to "keep the raw WebM, report
zeroed metadata" rather than raise. The damage only surfaces several services
downstream, where the orchestrator relabels that WebM as ``video_format="mp4"``,
the video analyzer cannot decode it, and the blank-clip gate hard-fails the run
with no mention of ffmpeg anywhere in the trace.

The Playwright base image is the trap: it bundles a PRIVATE ``ffmpeg-linux``
under ``/ms-playwright`` for its own recorder, so "the image has ffmpeg" reads
true while ``which ffmpeg`` is empty and ``ffprobe`` does not exist at all.
The Debian ``ffmpeg`` package provides BOTH binaries on PATH, which is what the
service actually needs. Verified missing in the live pod on 2026-07-31.
"""

from __future__ import annotations

import re
import shlex
from pathlib import Path

_DOCKERFILE = Path(__file__).resolve().parent.parent / "Dockerfile"


def _final_stage_lines() -> list[str]:
    """Return the Dockerfile's last build stage, comment-free, continuations joined.

    Comments are stripped FIRST so a mere mention of a package in prose (such as
    this repo's own "why" comments) can never satisfy the assertion — only a real
    instruction counts.
    """
    text = _DOCKERFILE.read_text(encoding="utf-8")
    code_lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
    # Join backslash continuations into single logical instructions.
    logical: list[str] = []
    buf = ""
    for line in code_lines:
        stripped = line.rstrip()
        if stripped.endswith("\\"):
            buf += stripped[:-1] + " "
            continue
        logical.append((buf + stripped).strip())
        buf = ""
    if buf:
        logical.append(buf.strip())

    stage_starts = [i for i, ln in enumerate(logical) if re.match(r"(?i)^FROM\s", ln)]
    assert stage_starts, "Dockerfile declares no FROM instruction"
    return logical[stage_starts[-1] :]


def _apt_installed_packages(lines: list[str]) -> set[str]:
    """Collect every package name passed to an ``apt-get install`` in ``lines``."""
    packages: set[str] = set()
    for line in lines:
        if "apt-get install" not in line:
            continue
        for command in re.split(r"&&|\|\||;", line):
            if "apt-get install" not in command:
                continue
            tokens = shlex.split(command)
            idx = tokens.index("install")
            packages.update(t for t in tokens[idx + 1 :] if not t.startswith("-"))
    return packages


class TestRuntimeStageBakesFfmpeg:
    def test_final_stage_installs_ffmpeg(self) -> None:
        packages = _apt_installed_packages(_final_stage_lines())
        assert "ffmpeg" in packages, (
            "The runtime stage must apt-get install `ffmpeg` (it provides BOTH "
            "`ffmpeg` and `ffprobe` on PATH). Without it every recorded clip is "
            "published as an undecodable WebM with duration_ms=0 / 0x0 and the "
            f"failure is SILENT. Packages found in the final stage: {sorted(packages)}"
        )

    def test_apt_hygiene_is_preserved(self) -> None:
        """The ffmpeg addition must not have cost the image-size hygiene."""
        lines = [ln for ln in _final_stage_lines() if "apt-get install" in ln]
        assert lines, "final stage runs no apt-get install"
        for line in lines:
            assert "--no-install-recommends" in line, line
            assert "rm -rf /var/lib/apt/lists/*" in line, line
