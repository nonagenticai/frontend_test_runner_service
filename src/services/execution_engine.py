"""Playwright test execution engine.

Manages the lifecycle of running Playwright tests in a subprocess:
1. Resolve the test package (download from BDD service or URL)
2. Extract to a sandboxed temp directory
3. Install dependencies (npm ci)
4. Run npx playwright test with JSON reporter
5. Parse results into structured models
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import html
import json
import logging
import mimetypes
import os
import re
import shutil
import signal
import tempfile
import zipfile
from collections.abc import Collection, Iterable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote, quote_plus, urlparse

import httpx
from w2v_common.auth.identity_context import identity_event_hook
from w2v_common.net import validate_public_url

from src.api.schemas import (
    AuthLogin,
    ScenarioClip,
    TestCaseResult,
    TestExecutionRequest,
    TestSummary,
    auth_logins,
    login_label,
    login_origin,
    login_secrets,
    logins_file_key,
    reject_url_parser_differentials,
    strict_login_url,
)
from src.config import get_settings
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)

#: Secret values of the login run executing in THIS task (ContextVar: concurrent jobs
#: on one pod never see each other's). Read by the log filter below.
_RUN_SECRETS: ContextVar[tuple[str, ...]] = ContextVar("ftr_run_secrets", default=())
_SCRUBBED = "[redacted]"
#: Playwright's conventional setup-file name (``auth.setup.ts``, ``login.setup.mjs``).
_SETUP_FILE_RE = re.compile(r"(^|/)[^/]*\.setup\.[cm]?[jt]s$")


def _secret_variants(secrets: Iterable[str]) -> tuple[str, ...]:
    """Each secret plus every encoded form it can surface in: Python repr and JSON
    string escaping, URL-encoding (path and form), base64 (std/url-safe, padded or
    not) and HTML escaping. A scrubber that matches only the raw value misses
    ``"pa\\"ss"`` in a JSON report or ``pa%40ss`` in a logged URL."""
    out: set[str] = set()
    for secret in secrets:
        raw = secret.encode()
        for b64 in (base64.b64encode(raw), base64.urlsafe_b64encode(raw)):
            text = b64.decode()
            out.update((text, text.rstrip("=")))
        out.update(
            (
                secret,
                repr(secret)[1:-1],
                json.dumps(secret)[1:-1],
                json.dumps(secret, ensure_ascii=False)[1:-1],
                quote(secret, safe=""),
                quote(secret),
                quote_plus(secret),
                html.escape(secret),
            )
        )
    return tuple(v for v in out if v)


def _scrub_text(text: str, secrets: Iterable[str]) -> str:
    for secret in sorted(secrets, key=len, reverse=True):
        text = text.replace(secret, _SCRUBBED)
    return text


def _scrub_obj[T](value: T, secrets: tuple[str, ...]) -> T:
    """Deep copy of a JSON-ish value with every secret replaced in every string."""
    if isinstance(value, str):
        return cast("T", _scrub_text(value, secrets))
    if isinstance(value, dict):
        return cast("T", {k: _scrub_obj(v, secrets) for k, v in value.items()})
    if isinstance(value, list):
        return cast("T", [_scrub_obj(v, secrets) for v in value])
    return value


def _is_setup_spec(spec: Mapping[str, Any]) -> bool:
    if _SETUP_FILE_RE.search(str(spec.get("file") or "")):
        return True
    return any(str(t.get("projectName") or "") == "setup" for t in spec.get("tests", []))


def _without_setup_specs(raw: _JsonObj) -> _JsonObj:
    """The JSON report minus every auth-setup spec (and suites left empty by it)."""

    def prune(suite: _JsonObj) -> _JsonObj | None:
        if _SETUP_FILE_RE.search(str(suite.get("file") or "")):
            return None
        specs = [sp for sp in suite.get("specs", []) if not _is_setup_spec(sp)]
        children = [c for c in (prune(ch) for ch in suite.get("suites", [])) if c is not None]
        return {**suite, "specs": specs, "suites": children}

    suites = [c for c in (prune(su) for su in raw.get("suites", [])) if c is not None]
    return {**raw, "suites": suites}


class _SecretScrubFilter(logging.Filter):
    """Scrub the current login run's secrets from every record this module logs —
    message AND traceback — so stdio a failing step echoes never reaches pod logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        secrets = _RUN_SECRETS.get()
        if not secrets:
            return True
        message = record.getMessage()
        if any(secret in message for secret in secrets):
            record.msg, record.args = _scrub_text(message, secrets), None
        if record.exc_info:
            record.exc_text = _scrub_text(
                logging.Formatter().formatException(record.exc_info), secrets
            )
            record.exc_info = None
        return True


logger.addFilter(_SecretScrubFilter())

#: Seconds of footage the ORCHESTRATOR assumes one recorded scenario yields when
#: it derives how many to order (`demo_video/gen.py::_SECONDS_PER_SCENARIO`).
#: Mirrored here so this service can say, on every capture, whether the
#: assumption held. It is deliberately a REPORTING threshold and never a gate:
#: getting it wrong must produce a misleading warning, never a failed run.
#:
#: ⚠️ The orchestrator's own value is justified by exactly two hand-observed
#: samples from one E2E run. That is precisely why measuring it here is worth
#: doing — nothing else feeds realised yield back.
_EXPECTED_SECONDS_PER_SCENARIO = 4.0

#: One harvested scenario clip, as it travels through this module's warning
#: helpers. A real pydantic model is the right end state, but the shape is
#: assembled incrementally across `_persist_scenario_clips` and read by several
#: independent helpers — so this alias is the honest intermediate: it names the
#: thing once instead of repeating `dict[str, Any]` at every boundary.
ClipDict = dict[str, Any]

# Playwright JSON-report objects (suites/specs/tests/results/attachments) are
# untyped nested dicts; alias the shape so new helpers don't grow the raw
# ``dict[str, Any]`` surface the Any-ratchet caps.
_JsonObj = dict[str, Any]

_MAX_ZIP_BYTES = 100 * 1024 * 1024  # 100 MB download cap
_MAX_UNCOMPRESSED_BYTES = 500 * 1024 * 1024  # 500 MB extraction cap

_CLIP_TAG_RE = re.compile(r"@?(clip-\d+)")

#: The ``@clip-<n>`` tag as it appears in a Gherkin ``.feature`` SOURCE file.
#: Deliberately anchored on the leading ``@`` (unlike ``_CLIP_TAG_RE``, which
#: also has to recover the tag from an on-disk directory name where the ``@`` is
#: stripped): in a feature file the ``@`` is what makes it a tag rather than a
#: substring of prose, and a bare ``clip-3`` in a step's text must not be read as
#: declared work.
_FEATURE_CLIP_TAG_RE = re.compile(r"@(clip-\d+)")

#: Directory names never scanned for declared work. ``.features-gen`` holds the
#: specs ``bddgen`` COMPILES from the sources, so counting it would double-count;
#: ``node_modules`` can contain a dependency's own fixture ``.feature`` files.
_NON_SOURCE_DIRS = frozenset({".features-gen", "node_modules"})

#: A tag line (``@clip-3``, ``@smoke @clip-3``, ...) or a ``Scenario``/
#: ``Scenario Outline`` header, used to walk a ``.feature`` file scenario by
#: scenario so a grep filter can be applied per scenario rather than to the
#: file as a whole. Deliberately loose about leading whitespace — Gherkin
#: sources are indented.
_GHERKIN_TAG_LINE_RE = re.compile(r"^\s*@\S")
_GHERKIN_SCENARIO_HEADER_RE = re.compile(r"^\s*Scenario(?: Outline)?:\s*(.*)$")
_GHERKIN_TAG_RE = re.compile(r"@(\S+)")

# Retention budgets for :func:`_subprocess_error_text`.
#
# ``_ERROR_TAIL_LINES`` is the TOTAL line budget; it is divided among the
# non-empty streams rather than applied to their concatenation. A joined cut
# always lets the chattier stream evict the other one entirely — and at
# ``_read_playwright_report`` the chatty stream is stderr (a whole Chromium run's
# log, routinely hundreds of lines) while the diagnosis is in stdout (the thing
# that failed to parse as JSON).
_ERROR_TAIL_LINES = 40
# Lines kept from the START of each stream. Node prints an uncaught exception as
# ``file:line`` + source line + caret + blank + ``SyntaxError: ...`` — the whole
# fingerprint lands inside the first handful of lines, and anything that follows
# may be thousands of lines of post-fatal chatter (npm warnings, Chromium spew)
# that would otherwise push it out of a tail-only window. 8 covers that frame
# with slack; the remaining 32 go to the tail, where a rethrown error's real
# message and stack land.
_ERROR_HEAD_LINES = 8
# The head gets 1/4 of a stream's CHARACTER budget. It is a fingerprint (one
# short ``SyntaxError:`` line is usually all of it); the tail carries the
# message, so it keeps the other 3/4. The split also stops a single minified
# multi-kilobyte first line from consuming the whole budget.
_ERROR_HEAD_CHAR_FRACTION = 4
_ELISION = "...[elided]"
_LINE_ELISION = "...[line truncated]"

# Strong references to in-flight shielded process-group reaps. asyncio holds only
# a WEAK reference to a running task, so a reap detached during teardown can be
# garbage-collected mid-await and never finish killing the group.
_PENDING_REAPS: set[asyncio.Task[None]] = set()


@dataclass(frozen=True, slots=True)
class _ReportWaitOutcome:
    """What happened while waiting for ``results.json`` + draining video.

    ``video_drain_timed_out`` is the only field the caller needs: it is False
    on every clean path (process exited on its own, no video files existed, or
    every ``.webm`` went stable within the drain deadline) and True only when
    the drain deadline expired while at least one clip was still changing
    size — i.e. the process group is about to be killed while a muxer may
    still be mid-flush. The caller folds this into the existing
    ``_signal_kill_warning`` channel rather than a new one.
    """

    video_drain_timed_out: bool = False


def _fit_line(line: str, budget: int, *, keep_head: bool) -> str:
    """Cut a single over-long line with an EXPLICIT marker (never a silent slice)."""
    if len(line) <= budget:
        return line
    room = budget - len(_LINE_ELISION)
    if room <= 0:
        return _LINE_ELISION
    return line[:room] + _LINE_ELISION if keep_head else _LINE_ELISION + line[-room:]


def _take_lines(lines: list[str], budget: int, *, from_end: bool) -> tuple[list[str], bool]:
    """Greedily keep WHOLE lines within ``budget`` characters.

    Trimming happens on line boundaries so a cut can never eat a stream label or
    leave a half-token. The only exception is a line that alone exceeds the whole
    budget (node's minified code frame); that one is cut with a visible marker,
    keeping the end of the line when we are working backwards from the tail.
    """
    ordered = list(reversed(lines)) if from_end else list(lines)
    kept: list[str] = []
    used = 0
    dropped = False
    for line in ordered:
        room = budget - used - (1 if kept else 0)  # newline join cost
        if room <= 0:
            dropped = True
            break
        if len(line) <= room:
            fitted = line
        elif kept:
            # Something already fits; drop this line whole rather than slice it.
            dropped = True
            break
        else:
            fitted = _fit_line(line, room, keep_head=not from_end)
            dropped = True
        used += len(fitted) + (1 if kept else 0)
        kept.append(fitted)
    if from_end:
        kept.reverse()
    return kept, dropped


def _head_tail_body(
    lines: list[str], *, budget: int, max_lines: int
) -> tuple[list[str], list[str], bool]:
    """Split ``lines`` into the kept HEAD and TAIL within ``budget`` characters.

    Returns ``(head_kept, tail_kept, elided)``; ``elided`` is True when anything
    at all was dropped (whole lines outside the line window, or lines/parts of
    lines that did not fit the character budget). The joined body is never longer
    than ``budget`` characters, elision marker EXCLUDED — the caller reserves
    room for that marker itself (see :func:`_render_stream`).
    """
    head_n = min(_ERROR_HEAD_LINES, max(max_lines // 2, 1))
    tail_n = max(max_lines - head_n, 1)
    tail = lines[-tail_n:]
    head = lines[: max(len(lines) - len(tail), 0)][:head_n]
    elided = len(head) + len(tail) < len(lines)

    head_kept, head_dropped = _take_lines(head, budget // _ERROR_HEAD_CHAR_FRACTION, from_end=False)
    used = sum(len(line) + 1 for line in head_kept)
    tail_kept, tail_dropped = _take_lines(tail, max(budget - used, 1), from_end=True)
    return head_kept, tail_kept, elided or head_dropped or tail_dropped


def _render_stream(label: str, raw: bytes, *, max_lines: int, max_chars: int) -> str:
    """Render ONE stream as ``label:`` plus its head and tail, elision marked.

    An EMPTY ``label`` renders the head+tail body alone, with no header line —
    used by :func:`_clamp_error_text` for text that is already one named thing
    (a Playwright error), where a ``stdout:``/``stderr:`` header would lie.

    ``max_chars`` is a HARD upper bound on the returned string, elision marker
    included. It used to bound only the *body*, then append ``_ELISION`` plus a
    newline on top — so a 1000-char budget returned up to 1011 characters and a
    consumer that sized a column/column-type to the declared budget overflowed.
    The marker is therefore RESERVED before the lines are taken, not added after.
    """
    text = raw.decode(errors="replace").strip()
    if not text:
        return ""
    header = f"{label}:" if label else ""
    budget = max(max_chars - len(header) - 1, 1)
    lines = text.splitlines()

    head_kept, tail_kept, elided = _head_tail_body(lines, budget=budget, max_lines=max_lines)
    if elided:
        # Re-take with the marker's own cost (marker + its newline) held back, so
        # header + body + marker together still fit inside ``max_chars``.
        head_kept, tail_kept, _ = _head_tail_body(
            lines, budget=max(budget - len(_ELISION) - 1, 1), max_lines=max_lines
        )
    body = [*head_kept, _ELISION, *tail_kept] if elided else [*head_kept, *tail_kept]
    rendered = "\n".join([header, *body] if header else body)
    # Absolute floor: for a ``max_chars`` smaller than the marker itself no
    # reservation can fit, and the contract is still a hard bound.
    return rendered if len(rendered) <= max_chars else rendered[:max_chars]


def _subprocess_error_text(
    stdout: bytes,
    stderr: bytes,
    *,
    max_chars: int,
    max_lines: int = _ERROR_TAIL_LINES,
) -> str:
    """Render a subprocess failure into the PART THAT ACTUALLY DIAGNOSES IT.

    Three bugs this exists to prevent, all reproduced against real output:

    1. **HEAD truncation.** Node prints the uncaught-exception code frame FIRST,
       and for playwright-bdd's bundled ``dist`` that frame is a single
       multi-kilobyte MINIFIED line. So ``error[:500]`` stored 500 characters of
       pure noise, cut mid-token, and DISCARDED the ``SyntaxError:`` /
       ``This Cucumber Expression has a problem at column N`` that follows it.
       A 2026-08-15 failure was stored as 630 chars ending at
       ``ImportJSONBindingNotDefa`` — undiagnosable, and it cost three sessions.
    2. **TAIL-ONLY truncation** (the first fix for #1, also wrong). When the fatal
       line comes FIRST and thousands of post-fatal warnings follow, a tail-only
       window shows 40 ``npm warn deprecated`` lines and no diagnosis at all. So
       each stream keeps a HEAD *and* a TAIL, with an explicit elision marker.
    3. **One budget for the joined text.** ``stderr or stdout`` used to discard
       stdout outright; joining the two and cutting the joint merely moved the
       bug — stderr, being last, evicted stdout label and all past 40 lines.
       Each non-empty stream now gets its OWN line and character budget, so a
       noisy stderr cannot erase stdout. stderr is still rendered LAST.
    """
    streams = [
        (label, raw) for label, raw in (("stdout", stdout), ("stderr", stderr)) if raw.strip()
    ]
    if not streams:
        return "<no subprocess output>"
    per_lines = max(max_lines // len(streams), 1)
    # The joined result also carries one newline per extra stream; charge those to
    # the budget so ``max_chars`` bounds the RETURNED string, not just its parts.
    per_chars = max((max_chars - (len(streams) - 1)) // len(streams), 1)
    rendered = [
        _render_stream(label, raw, max_lines=per_lines, max_chars=per_chars)
        for label, raw in streams
    ]
    return "\n".join(part for part in rendered if part)


def _clamp_error_text(text: str, *, max_chars: int, max_lines: int = _ERROR_TAIL_LINES) -> str:
    """Clamp an ALREADY-DECODED error string through the same head+tail budget.

    Playwright puts the failing locator in the ``Call log:`` block at the END of
    an error, so the old ``error[:1000]`` head-slice cut off the only part that
    names what missed — the exact class #93 fixed for subprocess output. This is
    that same clamp, not a second scheme: it delegates to :func:`_render_stream`
    with no stream label.

    ``max_chars`` is a HARD bound: ``len(result) <= max_chars`` always, elision
    marker included.
    """
    return _render_stream("", text.encode(), max_lines=max_lines, max_chars=max_chars)


# Versioned directed-footage recorder assets (injected into the downloaded
# playwright-bdd package at run time — see ExecutionEngine._ensure_track_recorder).
_TRACK_RECORDER_DIR = Path(__file__).parent / "track_recorder"
_TRACK_MARKER = "// w2v-test-runner: track recorder injected"
_RECORDER_PLACEHOLDER = '"__W2V_RECORDER_SOURCE__"'
# Files a package might call defineBddConfig() from, in preference order.
_DEFINE_BDD_CONFIG_RE = re.compile(r"defineBddConfig\(\s*\{")
# A package that already declares its OWN fixture module. We CHAIN onto it
# rather than bail (see _ensure_track_recorder), so the package keeps its
# fixtures AND the track recorder still runs.
_IMPORT_TEST_FROM_RE = re.compile(r"""importTestFrom\s*:\s*['"]([^'"]+)['"]""")
# The import statement in w2v.fixtures.ts whose module specifier is rewritten
# when chaining. Kept as a literal so the template stays valid TS on its own.
_FIXTURES_BASE_IMPORT = "from 'playwright-bdd'"
# When we chain, the marker records WHICH module we chained onto, so a second
# pass over an already-wired package rebuilds the identical fixtures.ts instead
# of silently resetting the base back to playwright-bdd.
_TRACK_MARKER_BASE_RE = re.compile(re.escape(_TRACK_MARKER) + r"\s*\(base:\s*([^)]+)\)")
# Module extensions Playwright's transpiler resolves for an extension-less
# relative importTestFrom specifier.
_TS_MODULE_SUFFIXES = (".ts", ".mts", ".cts", ".js", ".mjs", ".cjs")


# Playwright names a retry's output dir ``<test-dir>-retry<N>``; attempt 0 has no
# suffix. Both dirs carry the SAME ``@clip-<n>`` tag in their name, so salvage
# synthesizes two specs with the same clip key and ``_collect_clip_specs``'
# first-wins dedupe keeps whichever sorted first. Plain lexicographic sort put
# ``<name>`` before ``<name>-retry1``, i.e. it kept the FAILED attempt's footage
# and discarded the passing retry's.
_RETRY_DIR_SUFFIX_RE = re.compile(r"-retry(\d+)$")

# Statuses a ``soft-fail`` annotation demotes to ``failed``. BOTH describe a run
# whose DECIDING attempt passed: ``passed`` (single attempt) and ``flaky``
# (fail-then-pass, which is what ``_summarize_test_run`` returns the moment the
# caller sets ``retries >= 1`` — the schema allows 0-10). Demoting only
# ``passed`` meant the identical soft-failed selector escaped the honesty gate
# purely because a retry ladder was enabled.
_SOFT_FAIL_DEMOTABLE = frozenset({"passed", "flaky"})

# Blank-lead trim bounds. Playwright's video capture starts at CONTEXT CREATION
# (about:blank), before bdd_playwright_service's nav `Before` hook navigates and
# waits for a real paint — every acceptance film opens on that dead window
# (measured live: up to ~2.3s of "blank/white frame" scored 2.0/5 by the footage
# critic). The hook records exactly how long that took, in ms, as a `paint-lead`
# annotation (`_paint_lead_seconds` below); these bound how much of it we act on:
#   * MIN — below one frame-pair at Playwright's fixed 25fps capture rate, a trim
#     buys nothing visible and is not worth a second ffmpeg pass.
#   * MAX — a defensive ceiling. The hook's clock starts at ITS OWN entry, not the
#     exact instant the video muxer started, so it is a best-effort proxy (same
#     caveat as the tail-trim's action-track clock) — an anomalously large
#     reading must not be trusted to eat that much real footage.
#   * MIN_KEEP — a trim may never leave less than this much clip; unknown/unprobed
#     duration skips the trim entirely rather than risk emptying it.
_LEAD_TRIM_MIN_SECONDS = 0.12
_LEAD_TRIM_MAX_SECONDS = 6.0
_LEAD_TRIM_MIN_KEEP_SECONDS = 1.0


def _resolve_lead_trim_seconds(requested: float, duration_seconds: float) -> float:
    """Clamp a requested blank-lead trim against the safety bounds above.

    ``duration_seconds < 0`` means the source's duration could not be probed —
    trimming without knowing how much footage exists risks emptying the clip, so
    this returns ``0.0`` (no trim) rather than guess. Returns ``0.0`` whenever the
    clamped result would fall back under the MIN floor (a degenerately short
    clip, or a sub-floor request to begin with).
    """
    if requested < _LEAD_TRIM_MIN_SECONDS or duration_seconds < 0:
        return 0.0
    capped = min(
        requested,
        _LEAD_TRIM_MAX_SECONDS,
        max(0.0, duration_seconds - _LEAD_TRIM_MIN_KEEP_SECONDS),
    )
    return capped if capped >= _LEAD_TRIM_MIN_SECONDS else 0.0


# Fraction of the recordable clips that may be DROPPED as degenerate before the
# run is failed rather than completed-with-a-warning. Deliberately a strict
# majority (``dropped / attempted > 0.5``), and deliberately a code constant
# rather than an env knob: a ConfigMap key is exactly the surface that wedges a
# rollout, and this threshold is a product decision, not an operator dial.
#
# Why not fail on the first drop: one dud must not void its clean siblings, and
# ``job_manager`` hands the SAME row back to every resubmit under an
# ``Idempotency-Key``, so a ``failed`` row is permanent for that key — failing a
# 15-clip capture over 1 missing shot would burn the whole paid run.
# Why fail past half: the orchestrator joins narration 1:1 onto clip keys, so a
# film missing most of its shots is the zero-footage case in all but name, and
# ``_guard_zero_footage`` already fails that. Between the two, the drop is
# reported on the job as a ``capture_warning`` naming every missing clip_key.
_MAX_DROPPED_CLIP_FRACTION = 0.5


def _salvage_sort_key(webm: Path, known_dirs: Collection[str]) -> tuple[str, int, str]:
    """Order salvaged recordings so the HIGHEST attempt is seen FIRST.

    Grouped by the retry-stripped dir name, then by DESCENDING attempt ordinal —
    descending because the consumer (``_collect_clip_specs``) dedupes first-wins
    on ``clip_key``, so "wins" means "sorted first". Ordering between different
    tests stays alphabetical.

    A trailing ``-retry<N>`` is only READ as a retry marker when the stripped name
    is ALSO present on disk (``known_dirs``). Stripping it unconditionally broke
    the very case this ordering exists for: a scenario whose sanitized dir name
    legitimately ENDS ``-retry3`` gave attempt-0 the group ``…-user-can-`` and its
    retry the group ``…-user-can-retry3``, i.e. two different groups, so the failed
    attempt-0 take still sorted first and still won the first-wins dedupe.
    Playwright always names a retry ``<attempt-0 dir>-retry<N>``, so the attempt-0
    dir existing beside it is exactly the evidence that distinguishes the two.
    """
    dir_name = webm.parent.name
    match = _RETRY_DIR_SUFFIX_RE.search(dir_name)
    base = dir_name[: match.start()] if match else dir_name
    if match is None or base not in known_dirs:
        return (dir_name, 0, webm.name)
    return (base, -int(match.group(1)), webm.name)


def _salvage_order(webms: Iterable[Path]) -> list[Path]:
    """Sort salvaged ``.webm`` paths highest-attempt-first within each test.

    Takes the whole set at once because :func:`_salvage_sort_key` needs to know
    which sibling dir names exist to tell a retry suffix from a scenario name
    that merely ends that way.
    """
    paths = list(webms)
    known = {webm.parent.name for webm in paths}
    return sorted(paths, key=lambda webm: _salvage_sort_key(webm, known))


def _attempt_duration_ms(result: Mapping[str, object]) -> float:
    """One attempt's own duration, 0.0 when absent or non-numeric."""
    value = result.get("duration", 0)
    return float(value) if isinstance(value, int | float) else 0.0


def _dims_unmeasured(width: object, height: object) -> bool:
    """True when ffprobe gave us NO usable frame size for the clip.

    ``_ffprobe_meta`` zeroes its metadata whenever the probe could not run
    (ffprobe missing, hung, or an undecodable webm), and ``_is_degenerate``
    deliberately still PUBLISHES such a clip when the file has bytes. So
    ``width``/``height`` reaching the track normalizer as ``0`` is a live,
    reachable state — not a theoretical one.
    """
    if not isinstance(width, int | float) or not isinstance(height, int | float):
        return True
    return width <= 0 or height <= 0


def _normalize_track(
    track: dict[str, Any], width: int, height: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]] | None:
    """Scale a raw CSS-px track sidecar into RECORDED-VIDEO pixel space (VP-2).

    The recorder emits coordinates in CSS pixels relative to the page's
    ``innerWidth``/``innerHeight`` (``track["viewport"]``). The recorded video may
    be a different resolution, so X/W scale by ``fx = width / viewport.w`` and Y/H
    by ``fy = height / viewport.h`` INDEPENDENTLY (both 1.0 for the common
    1920x1080 1:1 case; separate factors keep coords correct even if the recording
    and the captured viewport differ in aspect ratio). Each factor is guarded
    against a zero/absent source dimension AND against an unmeasured RECORDED
    dimension (``_dims_unmeasured`` ⇒ ``None``, never all-zero geometry).
    ``viewport`` is rewritten to the recorded dimensions.

    Pure + total: returns ``(focus_track, cursor_track, viewport)`` or ``None``
    on any malformed input — it never raises.
    """
    if _dims_unmeasured(width, height):
        # Guarding only the DENOMINATOR (src_vp) left the NUMERATOR free: an
        # unmeasured clip gave fx = fy = 0.0, collapsing every bbox and cursor
        # coordinate to 0.0 and the viewport to {"w": 0, "h": 0}. The compiler
        # reads that as real geometry and pins the cursor/auto-zoom at the
        # origin for the whole clip (or divides by it). Omitting the track is
        # the honest "we could not measure this" state.
        return None
    try:
        src_vp = track.get("viewport") or {}
        src_w = float(src_vp.get("w") or 0)
        src_h = float(src_vp.get("h") or 0)
        fx = (width / src_w) if src_w > 0 else 1.0
        fy = (height / src_h) if src_h > 0 else 1.0

        focus_track: list[dict[str, Any]] = []
        for f in track.get("focus") or []:
            bbox = f.get("bbox") or {}
            focus_track.append(
                {
                    "t": float(f.get("t", 0.0)),
                    "bbox": {
                        "x": float(bbox.get("x", 0.0)) * fx,
                        "y": float(bbox.get("y", 0.0)) * fy,
                        "w": float(bbox.get("w", 0.0)) * fx,
                        "h": float(bbox.get("h", 0.0)) * fy,
                    },
                    "action": str(f.get("action", "reveal")),
                }
            )

        cursor_track: list[dict[str, Any]] = []
        for c in track.get("cursor") or []:
            cursor_track.append(
                {
                    "t": float(c.get("t", 0.0)),
                    "x": float(c.get("x", 0.0)) * fx,
                    "y": float(c.get("y", 0.0)) * fy,
                    "click": bool(c.get("click", False)),
                }
            )

        viewport = {"w": int(width), "h": int(height)}
        return focus_track, cursor_track, viewport
    except TypeError, ValueError, AttributeError:
        return None


def _load_and_normalize_track(
    track: Path | str | _JsonObj, width: int, height: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]] | None:
    """Normalize a ``w2v-track`` payload into recorded-video px.

    Accepts EITHER an already-parsed track mapping (the shape harvested from a
    ``body:`` attachment, which Playwright's JSON reporter inlines as base64 and
    never writes to disk) or a path to a sidecar FILE (a ``path:`` attachment).

    Total + defensive: a missing file, unreadable bytes, non-JSON content, or a
    normalize that rejects the shape all degrade to ``None`` — a malformed track
    must never fail a recording that otherwise succeeded."""
    if isinstance(track, dict):
        return _normalize_track(track, width, height)
    try:
        raw = json.loads(Path(track).read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    if not isinstance(raw, dict):
        return None
    return _normalize_track(raw, width, height)


def _slug(text: str) -> str:
    """Lowercase filesystem/URL-safe slug (``[a-z0-9-]``)."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "x"


def _clip_key_for_spec(spec: dict[str, Any], feature_title: str) -> str:
    """Stable join key for a recorded scenario clip.

    Prefers the deterministic ``@clip-<n>`` tag stamped by bdd_playwright_service
    (the pipeline-wide join spine) — looked up in the spec's ``tags`` then its
    title — and falls back to a ``feature__scenario`` slug for plain-Playwright
    packages that carry no clip tag.
    """
    for tag in spec.get("tags") or []:
        m = _CLIP_TAG_RE.fullmatch(str(tag).strip())
        if m:
            return m.group(1)
    title = str(spec.get("title") or "")
    m = _CLIP_TAG_RE.search(title)
    if m:
        return m.group(1)
    return f"{_slug(feature_title)}__{_slug(title)}"


def _is_degenerate(size_bytes: int, duration_ms: float, probe_ok: bool) -> bool:
    """True when a persisted clip file is unusable footage.

    A zero-byte file is always a dud. A zero DURATION only counts when the probe
    actually ran (``probe_ok``): a missing or hung ffprobe zeroes every clip, and
    reading that environment fault as "every clip is a dud" would silently
    discard an entire healthy harvest.
    """
    return size_bytes == 0 or (probe_ok and duration_ms == 0)


def _warn_if_undecodable(
    key: str, path: Path, size_bytes: int, duration_ms: float, width: int, height: int
) -> None:
    """Name a clip being PUBLISHED with real bytes but no decodable stream.

    That combination is the exact signature of a missing/broken ffprobe or a
    corrupt recording, and it is otherwise INVISIBLE: the clip is published, the
    orchestrator relabels it mp4, and the failure only surfaces as a blank-clip
    gate hard-fail three stages later. Name it here so the next occurrence is a
    grep away.

    Only the PUBLISH paths call this. A clip that gets dropped already logs its
    own drop warning carrying the same diagnosis; double-logging one clip would
    only make the real signal harder to grep for.
    """
    if size_bytes > 0 and (duration_ms == 0 or width == 0):
        logger.warning(
            "clip %s has %d bytes but no decodable stream "
            "(duration_ms=%s, %sx%s, file=%s) — ffprobe missing or clip corrupt",
            key,
            size_bytes,
            duration_ms,
            width,
            height,
            path.name,
        )


def _spec_is_recordable(spec: _JsonObj) -> bool:
    """True when a spec actually ran (or tried to) and therefore owed footage.

    A spec whose every attempt reports ``skipped`` produced no browser session
    and so legitimately has no video — counting it as "declared work" would make
    the zero-footage gate fire on an all-skipped suite. A spec carrying no tests
    at all declared nothing either.
    """
    for test in spec.get("tests", []):
        for result in test.get("results", []):
            if str(result.get("status", "")) != "skipped":
                return True
    return False


def _count_specs(raw: _JsonObj, *, recordable_only: bool = False) -> int:
    """Count every spec (scenario) in a Playwright JSON report, RECURSIVELY.

    Playwright nests specs inside child suites — ``bddgen`` emits one describe
    block per ``Feature:``, so a real report's specs live one (or more) levels
    down from ``raw["suites"]``. Any counter that only reads the top level
    reports 0 for a perfectly good report.

    ``recordable_only`` narrows the count to specs that owed footage (see
    ``_spec_is_recordable``); it is what the zero-footage gate reads.
    """
    total = 0

    def walk(suite: _JsonObj) -> None:
        nonlocal total
        for spec in suite.get("specs", []):
            if not recordable_only or _spec_is_recordable(spec):
                total += 1
        for child in suite.get("suites", []):
            walk(child)

    for suite in raw.get("suites", []):
        walk(suite)
    return total


async def _ffprobe_meta(path: Path) -> tuple[float, int, int, bool, bool]:
    """Return ``(duration_ms, width, height, has_audio, probe_ok)`` for a media file.

    Best-effort: a probe failure yields zeroed/false values rather than aborting
    clip persistence.

    ``probe_ok`` is False whenever the probe itself could not run or could not be
    parsed (ffprobe absent, hung, non-zero exit, unparseable JSON). It exists so a
    caller can tell "ffprobe says this file has no duration" (a genuine DUD) apart
    from "ffprobe could not tell us anything" (an environment fault). Without it a
    missing or hung ffprobe zeroes every clip and a degenerate-clip guard would
    silently discard an entire healthy harvest.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffprobe not available; clip metadata will be zeroed")
        return 0.0, 0, 0, False, False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        # Best-effort contract: a single hung ffprobe must degrade THIS clip's
        # metadata (zeroed), never orphan the process or abort the whole harvest.
        proc.kill()
        await proc.wait()
        logger.warning("ffprobe timed out for %s; clip metadata will be zeroed", path)
        return 0.0, 0, 0, False, False
    if proc.returncode != 0:
        # Never silent: a zeroed probe is indistinguishable downstream from a
        # genuinely empty clip, so the returncode + ffprobe's own diagnosis is
        # the ONLY signal that tells "undecodable stream" apart from "no bytes".
        logger.warning(
            "ffprobe failed for %s (rc=%s); clip metadata will be zeroed: %s",
            path,
            proc.returncode,
            _subprocess_error_text(b"", stderr, max_chars=2000),
        )
        return 0.0, 0, 0, False, False
    try:
        data = json.loads(stdout.decode(errors="replace"))
    except json.JSONDecodeError:
        # Head-truncation trap (#93 again): ffprobe's stdout starts with the
        # opening brace + banner-ish keys, so ``[:500]`` reliably kept the part
        # that says nothing and cut the malformed region that explains the parse
        # failure. Both ends are kept, elision marked.
        logger.warning(
            "ffprobe returned unparseable JSON for %s; clip metadata will be zeroed: %s",
            path,
            _subprocess_error_text(stdout, b"", max_chars=2000),
        )
        return 0.0, 0, 0, False, False
    duration_ms = float(data.get("format", {}).get("duration", 0.0) or 0.0) * 1000.0
    width = height = 0
    has_audio = False
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and not width:
            width = int(stream.get("width", 0) or 0)
            height = int(stream.get("height", 0) or 0)
        elif stream.get("codec_type") == "audio":
            has_audio = True
    return duration_ms, width, height, has_audio, True


async def _probe_video_fps(path: Path) -> float | None:
    """The video stream's frame rate, or None when it cannot be established.

    Reads ``avg_frame_rate`` first (the rate actually observed over the file)
    and falls back to ``r_frame_rate`` (the container's nominal rate), because a
    salvaged recording whose muxer was killed mid-write can carry a sane nominal
    rate and a garbage average, or vice versa. A ``0/0`` — ffprobe's "unknown" —
    is not a rate and yields None so the caller degrades instead of pinning the
    transcode to nonsense.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-print_format",
            "json",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate",
            str(path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffprobe not available; %s will be transcoded at its source rate", path)
        return None
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("ffprobe fps probe timed out for %s", path)
        return None
    if proc.returncode != 0:
        return None
    try:
        streams = json.loads(stdout.decode(errors="replace")).get("streams", [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None
    for field in ("avg_frame_rate", "r_frame_rate"):
        raw = str(streams[0].get(field, "") or "")
        num, _, den = raw.partition("/")
        try:
            fps = float(num) / float(den or 1)
        except TypeError, ValueError, ZeroDivisionError:
            continue
        if fps > 0:
            return fps
    return None


async def _convert_to_mp4(src: Path, dst: Path, *, lead_trim_seconds: float = 0.0) -> bool:
    """Transcode a recorded clip (Playwright writes WebM/VP8) to H.264 MP4.

    The compositor + browsers handle MP4 most reliably. Returns True on success;
    a failure leaves ``dst`` absent so the caller can fall back to the raw clip.

    ``lead_trim_seconds`` cuts a measured BLANK LEAD off the front (see
    ``_resolve_lead_trim_seconds`` / ``_paint_lead_seconds``) via an input-side
    ``-ss`` — fast-seeked to the nearest keyframe and then made frame-accurate by
    the re-encode this function already does unconditionally, so it adds no extra
    generation loss. ``0.0`` (the default) emits no ``-ss`` at all: byte-identical
    ffmpeg invocation to before this parameter existed.

    The output rate is PINNED to the source's measured rate. Playwright's video
    recorder hard-codes ``const fps = 25`` (playwright-core
    ``server/videoRecorder.js``) with no ``recordVideo`` option to change it, so
    capture is 25fps and this service cannot make it 30. Naming the rate
    explicitly keeps the transcode 1:1: ffmpeg's default output mode is CFR
    derived from the input timestamps, which on the salvage path (a recording
    whose muxer was SIGKILLed, leaving broken/variable timestamps) invents
    duplicated or dropped frames of its own. Emitting ``-r 30`` here would be
    the same defect deliberately — 25→30 duplicates one frame in six — so the
    remaining capture/conform mismatch has to be closed on the CONFORM side, in
    video_compiler, not by re-timing the master here.

    ``-crf 16``: without an explicit CRF, libx264 defaults to 23 — worse than
    the eventual master's CRF 18-20 — making this intermediate (already a
    second lossy generation over the VP8 capture) the quality FLOOR of the
    whole picture chain. 16 sits meaningfully below the master's range so the
    intermediate stays near-transparent relative to it, instead of merely
    matching 18 and reintroducing the same bottleneck one generation later.
    ``-preset veryfast`` is kept: preset trades encode TIME for compression
    EFFICIENCY at a fixed CRF, not for quality — a slower preset on this
    intermediate would only shrink a file that is deleted once the next stage
    consumes it, with no visual upside, at the cost of slower harvests on the
    ci-runner pool.
    """
    lead_args: list[str] = []
    if lead_trim_seconds > 0:
        probed_ms, _, _, _, probe_ok = await _ffprobe_meta(src)
        duration_seconds = probed_ms / 1000.0 if probe_ok else -1.0
        resolved_trim = _resolve_lead_trim_seconds(lead_trim_seconds, duration_seconds)
        if resolved_trim > 0:
            lead_args = ["-ss", f"{resolved_trim:.3f}"]
    fps = await _probe_video_fps(src)
    # No measurable rate ⇒ do not pin one; ffmpeg's own inference is still
    # better than a fabricated constant.
    # ``-r`` alone forces a CFR output at that rate on every ffmpeg the fleet
    # runs; ``-fps_mode cfr`` would be redundant AND unrecognized on the
    # ffmpeg 4.x a dev box still carries, which would fail the transcode and
    # silently demote every clip to the raw-webm fallback.
    rate_args = ["-r", f"{fps:.6f}".rstrip("0").rstrip(".")] if fps else []
    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-y",
            *lead_args,
            "-i",
            str(src),
            *rate_args,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "16",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(dst),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        logger.warning("ffmpeg not available; keeping raw recording for %s", src)
        return False
    try:
        _, stderr = await asyncio.wait_for(proc.communicate(), timeout=180)
    except TimeoutError:
        # Best-effort contract: a hung ffmpeg transcode must degrade THIS clip
        # (caller falls back to the raw recording), never orphan the process or
        # abort the whole harvest.
        proc.kill()
        await proc.wait()
        logger.warning("ffmpeg mp4 convert timed out for %s; keeping raw recording", src)
        return False
    if proc.returncode != 0:
        # ffmpeg prints its banner + the whole ``configuration:`` flag list
        # FIRST, so a 500-char HEAD slice is pure banner and the actual cause
        # (``moov atom not found``, ``Invalid data found when processing
        # input``) — the one line that explains a clip about to be dropped as
        # degenerate — was discarded. Keep head AND tail, budgeted.
        logger.warning(
            "ffmpeg mp4 convert failed for %s: %s",
            src,
            _subprocess_error_text(b"", stderr, max_chars=2000),
        )
        return False
    # rc==0 is NOT proof of a usable file: a truncated input can make ffmpeg exit
    # clean having written only a header (or nothing). Publishing that empty MP4
    # is worse than falling back to the raw WebM, so require real bytes.
    if not dst.exists() or dst.stat().st_size == 0:
        logger.warning(
            "ffmpeg mp4 convert exited 0 but produced no usable output for %s "
            "(exists=%s); keeping raw recording",
            src,
            dst.exists(),
        )
        return False
    return True


def _endpoint_netloc(endpoint: str | None) -> str | None:
    """Reduce a configured MinIO endpoint to a bare ``host[:port]`` netloc.

    The setting is written both ways in practice (``minio.internal:8389`` and
    ``http://minio.internal:8389``), so strip any scheme, path and userinfo
    before comparing it against a URL's host.
    """
    if not endpoint:
        return None
    ep = endpoint.strip()
    if "://" in ep:
        ep = urlparse(ep).netloc or ep.split("://", 1)[1]
    ep = ep.split("/", 1)[0]
    if "@" in ep:
        ep = ep.rsplit("@", 1)[-1]
    return ep.lower() or None


def _artifact_allow_hosts() -> set[str]:
    """The MinIO endpoint(s) THIS service is configured to use.

    Presigned artifact URLs at our own object store legitimately resolve into a
    blocked internal range, so they are trusted past the SSRF guard's
    hostname/range checks. This is deliberately NOT an open operator allowlist:
    it is derived from ``STORAGE_MINIO_ENDPOINT``, so an attacker-supplied
    ``*.internal`` host that is not our artifact store is still refused.

    No license token is ever attached to this fetch — the token is bound to the
    ``allow_private=True`` bdd branch only, so trusting the host here cannot
    leak a credential (see the credential-leak guard in ``_download_zip``).
    """
    storage = get_settings().storage
    hosts: set[str] = set()
    for endpoint in (storage.minio_endpoint, getattr(storage, "minio_external_endpoint", None)):
        netloc = _endpoint_netloc(endpoint)
        if netloc:
            hosts.add(netloc)
    return hosts


def _validate_url(url: str, *, allow_private: bool = False) -> None:
    """Validate a URL is safe to fetch (SSRF protection).

    The network-range decision is delegated to the fleet-wide guard
    ``w2v_common.net.validate_public_url``. The local check this replaced had
    two holes:

    1. It only inspected IP **literals** — any hostname returned early with a
       comment claiming it would be checked "at connect time". Nothing checked
       it at connect time, so `http://internal-thing.example.com/` pointing at
       an internal address was a total bypass. The shared guard does a real
       ``socket.getaddrinfo`` and fails CLOSED when resolution fails.
    2. It relied on ``ipaddress.is_private``, which is **False** for RFC 6598
       CGNAT ``100.64.0.0/10`` — precisely this estate's Tailscale tailnet
       (postgres, registry, the k3s control plane). It also missed multicast
       (``224/4``) and reserved (``240/4``).

    ``allow_private=True`` remains the documented escape hatch for trusted
    internal peers (the ``bdd_playwright_service`` S2S package download) and
    skips the address check entirely; scheme/hostname are still enforced.

    Closing hole (1) by NAME also rejected our own artifact store: the live
    showcase/authored path sends a presigned
    ``http://minio.internal:8389/...?X-Amz-Signature=...`` as ``package_url``
    on the *untrusted* branch, which is a legitimate internal fetch. Our own
    configured MinIO endpoint is therefore passed as ``allow_hosts`` — see
    :func:`_artifact_allow_hosts` for why that is not a general widening. The
    allowed host is fetched WITHOUT IP-pinning so the presigned ``Host``/SigV4
    signature survives byte-for-byte, and the allow-list is re-consulted on
    every redirect hop by the shared guard.

    Raises:
        ValueError: the scheme is not http(s), or the URL has no hostname.
        w2v_common.net.SSRFError (a ``ValueError`` subclass): the URL targets or
            resolves to a blocked address, or DNS resolution failed.
    """
    # Before urlparse: a URL the browser and urlparse read differently would have
    # its SAFE-looking host checked and its other host navigated to.
    reject_url_parser_differentials(url, "URL")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL must use http or https scheme: {url!r}")
    if not parsed.hostname:
        raise ValueError(f"URL must have a valid hostname: {url!r}")
    if allow_private:
        return
    validate_public_url(url, allow_hosts=_artifact_allow_hosts())


def _is_within(candidate: Path, parent: Path) -> bool:
    """True iff ``candidate`` resolves to a path inside ``parent``.

    Uses ``Path.is_relative_to`` rather than prefix matching so that
    ``/tmp/outside`` is NOT considered to live inside ``/tmp/out``.
    """
    try:
        return candidate.is_relative_to(parent)
    except ValueError:
        return False


_EXTRACT_CHUNK_BYTES = 1024 * 1024  # decompress at most 1 MB per read
# A sentinel far larger than any legitimate archive, used to disable zipfile's
# file_size-based output truncation so our own counter is authoritative.
_EXTRACT_SIZE_SENTINEL = 1 << 62


def _attachment_path_if_confined(raw_path: str, base_dir: Path | None) -> Path | None:
    """Resolve an attachment ``path`` from the (untrusted) Playwright report and
    return it only if it exists AND stays confined within ``base_dir``.

    ``results.json`` is written by the user-supplied test package, so
    ``att["path"]`` is attacker-controlled: a malicious package emitting
    ``{"name": "video", "path": "/app/.env"}`` would otherwise have that file
    copied into the shared object store and served back. Confining the SOURCE
    path to the extracted package/run directory closes that exfiltration (the
    object-KEY side is already guarded by ``_safe_key``). ``base_dir=None``
    disables the check (used only by callers that supply no run directory).
    """
    candidate = Path(raw_path)
    if not candidate.exists():
        return None
    if base_dir is not None and not _is_within(candidate.resolve(), base_dir.resolve()):
        logger.warning(
            "Rejecting attachment path %r — escapes the package directory %s",
            raw_path,
            base_dir,
        )
        return None
    return candidate


def _decode_attachment_body(body: object) -> _JsonObj | None:
    """Decode an inline attachment ``body`` into a JSON object, or ``None``.

    Playwright's JSON reporter base64-encodes a ``body:`` attachment into a
    ``body`` string. Everything here is untrusted report content, so every
    failure mode — wrong type, non-base64, non-UTF-8, non-JSON, JSON that is not
    an object — degrades to ``None`` rather than raising.
    """
    if isinstance(body, dict):  # an already-parsed report
        return cast(_JsonObj, body)
    if isinstance(body, bytes | bytearray):
        blob = bytes(body)
    elif isinstance(body, str):
        try:
            blob = base64.b64decode(body, validate=True)
        except binascii.Error, ValueError:
            return None
    else:
        return None
    try:
        parsed = json.loads(blob.decode("utf-8"))
    except UnicodeDecodeError, ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _track_payload_from_attachment(att: _JsonObj, base_dir: Path | None) -> _JsonObj | None:
    """Parse a ``w2v-track`` attachment from EITHER Playwright report shape.

    ``body:`` attachments (what the injected fixture emits) arrive base64-inline
    with no ``path``; ``path:`` attachments point at an on-disk sidecar. Both are
    accepted, body first — it needs no filesystem and cannot escape the package
    dir — with a fall-through to ``path`` so a corrupt body still degrades to the
    file when one exists.
    """
    payload = _decode_attachment_body(att.get("body"))
    if payload is not None:
        return payload
    raw_path = att.get("path")
    if not raw_path:
        return None
    candidate = _attachment_path_if_confined(str(raw_path), base_dir)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    """Extract ZIP safely — reject path traversal, symlinks, and enforce size limits.

    The uncompressed cap is enforced on the bytes ACTUALLY decompressed, never on
    the ``ZipInfo.file_size`` header — that field is attacker-controlled, and a
    decompression bomb sets it to 0 (or understates it) so a pre-check that sums
    it passes while the deflate stream still expands to gigabytes. We therefore
    extract each entry ourselves, streaming through a running byte counter and
    aborting (with ``dest`` cleaned up) the moment the running total exceeds
    ``_MAX_UNCOMPRESSED_BYTES``.

    ``zipfile`` normally truncates ``open()``/``extractall()`` output to the
    declared ``file_size``; to make the counter authoritative we temporarily raise
    each entry's ``file_size`` to a huge sentinel so the read is NOT capped by the
    attacker's lie, then bound it with our own limit.
    """
    resolved_dest = dest.resolve()
    total_written = 0

    try:
        for info in zf.infolist():
            # Zip-slip: a manual per-entry extract loses ``extractall``'s implicit
            # containment, so this per-member traversal check is now the PRIMARY
            # guard (re-added), not defense-in-depth.
            target = (dest / info.filename).resolve()
            if not _is_within(target, resolved_dest):
                raise ValueError(f"ZIP path traversal attempt: {info.filename!r}")

            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            # Defeat zipfile's file_size-based output truncation so our counter,
            # not the header, decides how many bytes may be written.
            info.file_size = _EXTRACT_SIZE_SENTINEL
            with zf.open(info, "r") as src, target.open("wb") as out:
                while chunk := src.read(_EXTRACT_CHUNK_BYTES):
                    total_written += len(chunk)
                    if total_written > _MAX_UNCOMPRESSED_BYTES:
                        raise ValueError(
                            "ZIP uncompressed size exceeds "
                            f"{_MAX_UNCOMPRESSED_BYTES // (1024 * 1024)} MB limit"
                        )
                    out.write(chunk)
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise

    # Python's ``zipfile`` does NOT materialize symlinks from ZIP
    # entries — it writes the link target as a regular file — so this loop
    # is defense-in-depth, not the primary control. It still catches symlinks
    # that were in ``dest`` before extraction (pre-populated by a caller) and
    # guards against any future zipfile behaviour change that starts honouring
    # the symlink bit in external_attr.
    for path in dest.rglob("*"):
        if path.is_symlink():
            link_target = (path.parent / path.readlink()).resolve()
            if not _is_within(link_target, resolved_dest):
                shutil.rmtree(dest, ignore_errors=True)
                raise ValueError(f"ZIP contains symlink escaping sandbox: {path}")


class ExecutionEngine:
    """Runs Playwright test packages and collects results."""

    def __init__(self) -> None:
        # Lazily constructed so importing the engine (and the module-level
        # ``_engine`` singleton in routes) does not touch storage config; the
        # backend is initialized on first artifact write.
        self._storage: StorageService | None = None

    @property
    def storage(self) -> StorageService:
        """Shared object store for report + clip artifacts (honours the
        configured backend so any replica serves any job's artifacts)."""
        if self._storage is None:
            self._storage = StorageService()
        return self._storage

    async def execute(
        self,
        request: TestExecutionRequest,
        *,
        persist_dir: Path | None = None,
        public_artifact_base: str | None = None,
        work_dir: Path | None = None,
        license_token: str | None = None,
    ) -> dict[str, Any]:
        """Execute a test package; a LOGIN run is fenced so no artifact carries a secret.

        Playwright writes typed values into places this service ships: a trace
        records ``fill(password)``'s argument, the HTML report renders step titles
        (``Fill "<value>" locator(...)``), the auth-setup project can record video and
        failure screenshots, and an error message can quote a step. So when
        ``auth_config`` carries a secret: tracing is forced off, the HTML report is
        NOT uploaded, auth-setup specs are dropped before the harvest (never a clip,
        never a test row), and every string in the result, the raised error and this
        module's log lines is scrubbed of each secret value as a final backstop.
        """
        secrets = _secret_variants(login_secrets(request.auth_config))
        if not secrets:
            return await self._execute(
                request,
                persist_dir=persist_dir,
                public_artifact_base=public_artifact_base,
                work_dir=work_dir,
                license_token=license_token,
                login_run=False,
            )
        request = request.model_copy(update={"trace_on_failure": False})
        token = _RUN_SECRETS.set(secrets)
        try:
            result = await self._execute(
                request,
                persist_dir=persist_dir,
                public_artifact_base=public_artifact_base,
                work_dir=work_dir,
                license_token=license_token,
                login_run=True,
            )
        except Exception as exc:
            message = str(exc)
            if any(secret in message for secret in secrets):
                raise RuntimeError(_scrub_text(message, secrets)) from None
            raise
        finally:
            _RUN_SECRETS.reset(token)
        return _scrub_obj(result, secrets)

    async def _execute(
        self,
        request: TestExecutionRequest,
        *,
        persist_dir: Path | None,
        public_artifact_base: str | None,
        work_dir: Path | None,
        license_token: str | None,
        login_run: bool,
    ) -> _JsonObj:
        """Execute a test package and return structured results.

        Returns a dict with keys: summary, test_results, report_path, traces.

        ``license_token`` (the request-scoped caller token, opaque) authenticates
        the S2S package download from ``bdd_playwright_service`` via
        ``X-Tenant-Id: Bearer <token>``. It is applied ONLY to the trusted
        internal bdd download — never to an arbitrary external ``package_url``.
        """
        tmp_dir = work_dir or Path(tempfile.mkdtemp(prefix="test_runner_"))

        try:
            # 1. Resolve package
            package_dir = await self._resolve_package(request, tmp_dir, license_token)

            # 2. Write target URL + login env BEFORE installing: with auth it injects
            #    the `dotenv` dependency into package.json, and install must fetch it.
            #    (Installing first left a package that did not list dotenv unable to
            #    load its credentials: `Cannot find module 'dotenv/config'`.)
            dotenv_injected = self._configure_target(package_dir, request)

            # 3. Install dependencies. An injected dependency makes a shipped lockfile
            #    stale, and `npm ci` refuses a stale lockfile — so install instead.
            await self._install_dependencies(package_dir, lockfile_stale=dotenv_injected)

            # 3b. Inject the directed-footage track recorder (auto-fixture +
            #     importTestFrom) BEFORE bddgen so the generated specs inherit
            #     it. Returns the reasons it COULD NOT wire (empty == wired);
            #     they ride out on the job as `capture_warnings` so a track-less
            #     capture is diagnosable without reading pod logs.
            capture_warnings = self._ensure_track_recorder(package_dir)
            # Request fields this service accepts but cannot honour must reach the
            # CALLER, not just the pod log — a silent no-op is how a caller ends
            # up believing it turned something off.
            capture_warnings += self._unimplemented_option_warnings(request)

            # 4. Generate the BDD test files (playwright-bdd compiles .feature +
            #    steps → .features-gen, which the config's testDir points at). A
            #    plain-Playwright package without playwright-bdd skips this.
            #    WITHOUT this step a generated BDD suite has an EMPTY testDir →
            #    zero tests → zero clips. This is the core fix.
            await self._run_bddgen(package_dir)

            # 5. Run tests. `timed_out` is carried out-of-band because a
            #    wall-timeout whose salvage found nothing produces a report
            #    indistinguishable from a healthy empty suite.
            raw_output, timed_out = await self._run_tests(package_dir, request)
            if login_run and not (package_dir / "results.json").is_file():
                # FAIL CLOSED: without the reporter's results.json the report is a
                # SALVAGE synthesised from whatever .webm/.png sits on disk, and
                # nothing can tell the auth-setup recording (credentials being typed)
                # from scenario footage. A login run ships none of it.
                logger.warning(
                    "login run without results.json: withholding every clip, video "
                    "and screenshot (salvage cannot exclude the auth-setup recording)"
                )
                raw_output = {"suites": []}
            if login_run:
                # The auth-setup project typed the credentials: its steps, video and
                # screenshots are never footage and never a reported test.
                raw_output = _without_setup_specs(raw_output)

            # 5b. On the timeout path ONLY, read the denominator from the package
            #     BEFORE the report gets a vote. `_salvage_raw_from_disk`
            #     synthesizes one spec per surviving clip, so the salvaged report
            #     silently redefines "how many scenarios were expected" to equal
            #     "how many survived". Must be read while package_dir still exists
            #     — execute()'s `finally` rmtree's it.
            declared_clip_keys: set[str] = (
                await asyncio.to_thread(self._declared_clip_keys, package_dir, request.grep_pattern)
                if timed_out
                else set()
            )

            # 6. Parse results
            result = self._parse_results(raw_output, package_dir)

            # 7. Persist artifacts that need to survive temp dir cleanup.
            if persist_dir and public_artifact_base:
                result.update(
                    await self._persist_artifacts(
                        package_dir,
                        persist_dir,
                        public_artifact_base,
                        raw=result.get("raw", {}),
                        upload_report=not login_run,
                    )
                )
                # 8. Zero-footage gate — the ONE place the run is failed for
                #    producing no usable video. It runs here, after persistence,
                #    because only here is the final harvest known.
                self._guard_zero_footage(
                    result,
                    timed_out=timed_out,
                    expect_recording=request.expect_recording,
                )
                # 8b. Partial-footage gate. `_guard_zero_footage` returns as soon
                #     as ONE clip survived, so a harvest that silently dropped
                #     duds reached the caller as a clean, merely smaller capture.
                capture_warnings += self._guard_partial_footage(
                    result, expect_recording=request.expect_recording
                )
                # 8c. A clip that RECORDED a track this service then discarded is
                #     indistinguishable, in the response, from a scenario that
                #     never touched the page — and `_track_coverage_warnings`
                #     below goes silent the moment one sibling kept its track.
                capture_warnings += self._discarded_track_warnings(result)

            # Track coverage is only knowable AFTER the harvest, so the two
            # sources of "this capture has no camera movement" are joined here
            # into the one field the job carries.
            _clips = result.get("scenario_clips") or []
            _signal_warning = result.pop("_signal_kill_warning", None)
            if _signal_warning:
                capture_warnings.append(str(_signal_warning))
            capture_warnings += self._track_coverage_warnings(_clips)
            # A salvage is "N of M", never a redefined M. This is the ONE gate that
            # reads a denominator the salvaged report cannot have influenced.
            capture_warnings += self._salvage_shortfall_warnings(
                _clips,
                declared_clip_keys=declared_clip_keys,
                timed_out=timed_out,
                expect_recording=request.expect_recording,
            )
            _summary = result.get("summary") or {}
            capture_warnings += self._footage_yield_warnings(
                _clips,
                declared=max(
                    int(_summary.get("total", 0) or 0) - int(_summary.get("skipped", 0) or 0),
                    _count_specs(result.get("raw") or {}, recordable_only=True),
                ),
                expect_recording=request.expect_recording,
            )
            result["capture_warnings"] = capture_warnings

            return result

        except Exception:
            logger.exception("Test execution failed in %s", tmp_dir)
            raise
        finally:
            if work_dir is None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    async def _resolve_package(
        self,
        request: TestExecutionRequest,
        tmp_dir: Path,
        license_token: str | None = None,
    ) -> Path:
        """Download and extract the test package."""
        package_dir = tmp_dir / "tests"
        package_dir.mkdir(parents=True, exist_ok=True)

        if request.bdd_job_id:
            from w2v_common.discovery import service_url

            download_url = f"{service_url('bdd_playwright')}/api/bdd/{request.bdd_job_id}/download"
            # BDD service is internal — allow private addresses + forward the
            # caller's license token so the bdd download endpoint (fail-closed)
            # authenticates the S2S call.
            zip_path = await self._download_zip(
                download_url, tmp_dir, allow_private=True, license_token=license_token
            )

        elif request.package_url:
            # External URL — validate SSRF protection. NEVER attach the license
            # token to an arbitrary external host (credential-leak guard).
            _validate_url(request.package_url)
            zip_path = await self._download_zip(request.package_url, tmp_dir, allow_private=False)

        else:
            raise ValueError("Either bdd_job_id or package_url must be provided")

        with zipfile.ZipFile(zip_path, "r") as zf:
            _safe_extract(zf, package_dir)

        # If the zip contained a single top-level directory, descend into it
        contents = list(package_dir.iterdir())
        if len(contents) == 1 and contents[0].is_dir():
            package_dir = contents[0]

        return package_dir

    async def _download_zip(
        self,
        url: str,
        tmp_dir: Path,
        *,
        allow_private: bool = False,
        license_token: str | None = None,
    ) -> Path:
        """Download a ZIP file with size limits.

        Redirects are disabled for external URLs to prevent redirect-after-validation
        SSRF (attacker's server redirects to an internal IP).

        ``license_token`` (opaque, trusted-internal callers only) is sent as
        ``X-Tenant-Id: Bearer <token>`` so a fail-closed peer (bdd) authorizes
        the S2S download. Callers MUST NOT pass it for ``allow_private=False``
        (arbitrary external) URLs.
        """
        if not allow_private:
            _validate_url(url)

        headers: dict[str, str] = {}
        if license_token and allow_private:
            # Canonical w2v S2S license-token transport (see
            # w2v_common.auth.middleware._extract_license_token). identity_event_hook
            # preserves an existing Bearer X-Tenant-Id, so this is not overwritten.
            headers["X-Tenant-Id"] = f"Bearer {license_token}"

        zip_path = tmp_dir / "package.zip"
        async with (
            httpx.AsyncClient(
                timeout=120.0,
                follow_redirects=allow_private,  # only follow redirects for trusted internal URLs
                headers=headers,
                event_hooks={"request": [identity_event_hook]},
            ) as client,
            client.stream("GET", url) as resp,
        ):
            resp.raise_for_status()
            downloaded = 0
            with zip_path.open("wb") as f:
                async for chunk in resp.aiter_bytes(65536):
                    downloaded += len(chunk)
                    if downloaded > _MAX_ZIP_BYTES:
                        raise ValueError(
                            f"ZIP download exceeds {_MAX_ZIP_BYTES // (1024 * 1024)} MB limit"
                        )
                    f.write(chunk)

        return zip_path

    @staticmethod
    def _subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
        """Build a restricted env for subprocess execution.

        Inherits only an allowlist (PATH, HOME, LANG, TMPDIR, …) — strips secrets
        like DATABASE_URL, API_KEY, etc. that could be exfiltrated by malicious
        preinstall scripts in package.json.

        PLAYWRIGHT_BROWSERS_PATH MUST be inherited: the base image bakes the
        browsers at /ms-playwright via this var, and the npm install runs with
        --ignore-scripts (no per-package browser download). Without it, `npx
        playwright test` cannot find a browser, never launches, and the run
        produces zero footage while burning the whole wall budget. It is a path,
        not a secret, so passing it through is safe.
        """
        safe_keys = {
            "PATH",
            "HOME",
            "LANG",
            "TMPDIR",
            "USER",
            "SHELL",
            "NODE_PATH",
            "PLAYWRIGHT_BROWSERS_PATH",
        }
        env = {k: v for k, v in os.environ.items() if k in safe_keys}
        if extra:
            env.update(extra)
        return env

    async def _install_dependencies(
        self, package_dir: Path, *, lockfile_stale: bool = False
    ) -> None:
        """Install test dependencies via npm.

        Uses ``npm ci`` when a lockfile exists (fast, deterministic), or
        falls back to ``npm install`` for generated packages that only
        ship ``package.json`` — or whose lockfile FTR itself made stale by
        injecting a dependency (``lockfile_stale``: ``npm ci`` exits 1 when
        package.json and package-lock.json disagree).
        """
        if not (package_dir / "package.json").exists():
            logger.warning("No package.json found in %s, skipping install", package_dir)
            return

        has_lockfile = (package_dir / "package-lock.json").exists()
        cmd = ["npm", "ci"] if has_lockfile and not lockfile_stale else ["npm", "install"]
        cmd.extend(["--no-audit", "--no-fund", "--ignore-scripts"])

        logger.info("Installing npm dependencies in %s (%s)", package_dir, cmd[1])
        await self._run_node_tool(
            cmd, package_dir, label=f"npm {cmd[1]}", timeout=120, log_stem="npm"
        )

    async def _run_node_tool(
        self,
        cmd: list[str],
        package_dir: Path,
        *,
        label: str,
        timeout: float,
        log_stem: str,
    ) -> None:
        """Run a node build tool to completion, keeping its output on BOTH failures.

        Shared by ``_install_dependencies`` and ``_run_bddgen``. Both used to pipe
        stdio through ``communicate()`` and, on the wall timeout, DISCARD it and
        re-raise a bare ``TimeoutError`` — which ``routes`` stored as the literal
        string ``"TimeoutError: "`` while ``execute()``'s ``finally`` rmtree'd the
        package dir, deleting the only evidence. That is exactly the ``<no error>``
        class #93/#97 fixed on the Playwright path, and the fix is the same one:
        redirect stdio to FILES and read them back through ``_read_captured_stdio``
        before anything can delete them.

        Files (not PIPEs) for the same reason ``_run_tests`` uses them: a tool that
        is hung is precisely the case where a grandchild holds the pipe FDs open,
        so ``communicate()`` cannot be relied on to hand back what was already
        written — and cancelling it drops whatever it had buffered.

        The diagnostics budget is 1800 chars so the raised message still fits
        inside the 2000-char HEAD slice ``routes`` stores on the job row; a
        4000-char message would be cut mid-evidence there.
        """
        stdout_path = package_dir / f"_{log_stem}_stdout.log"
        stderr_path = package_dir / f"_{log_stem}_stderr.log"
        proc: asyncio.subprocess.Process | None = None
        pgid: int | None = None
        try:
            with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(package_dir),
                    stdout=out_f,
                    stderr=err_f,
                    env=self._subprocess_env(),
                    start_new_session=True,
                )
                pgid = self._capture_pgid(proc)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except TimeoutError:
                    # A hung registry fetch / transpiler must not leak the tree.
                    await self._terminate_process_group(proc, pgid)
                    diagnostics = await asyncio.to_thread(
                        self._read_captured_stdio, stdout_path, stderr_path, 1800
                    )
                    logger.error(
                        "%s timed out after %ss; killed process group: %s",
                        label,
                        timeout,
                        diagnostics,
                    )
                    raise TimeoutError(
                        f"{label} timed out after {timeout}s and its process group was "
                        f"killed: {diagnostics}"
                    ) from None

            if proc.returncode != 0:
                diagnostics = await asyncio.to_thread(
                    self._read_captured_stdio, stdout_path, stderr_path, 1800
                )
                logger.error("%s failed: %s", label, diagnostics)
                raise RuntimeError(
                    f"{label} failed with exit code {proc.returncode}: {diagnostics}"
                )

            logger.info("%s completed successfully", label)
        finally:
            # ALWAYS clean up the session group — same shape as _run_tests. npm/npx
            # exiting 0 does NOT mean its group is empty: a postinstall- or
            # npx-spawned esbuild/tsx daemon survives the leader and would orphan
            # under FTR's PID 1. Reaping only in the TimeoutError branch left the
            # success and non-zero-exit paths leaking. Idempotent with that branch.
            if proc is not None:
                await self._terminate_process_group(proc, pgid)

    def _configure_target(
        self,
        package_dir: Path,
        request: TestExecutionRequest,
    ) -> bool:
        """Validate target_url + auth login_url(s), write Playwright env to ``.env``.

        Returns True when the ``dotenv`` dependency was injected into package.json
        (the caller must then not ``npm ci`` against the shipped lockfile).

        BOTH tenant-supplied navigation targets run through :func:`_validate_url`
        — ``target_url`` and every login's ``login_url``. The latter is the page
        the scaffold types the tenant's credentials into, so an unguarded value
        is a credential-harvesting sink, not just an SSRF probe.

        Auth credentials (``AUTH_USERNAME``/``AUTH_PASSWORD``) are written
        to ``.env`` mode 0600 inside the 0700 sandbox tempdir.  They are
        deliberately NOT placed in the subprocess environment dict because
        ``/proc/<pid>/environ`` is readable across pods sharing a node UID
        on default K3s — see the security audit (H3, 2026-04-27).

        Several logins (``auth_config`` as a LIST, one per origin — a recording
        that signs into Keycloak-fronted marketing, Twenty and a console in one
        run) are ALSO written to ``.auth/logins.json`` (0600), keyed by origin,
        and named by ``AUTH_LOGINS_FILE``. ``AUTH_*`` carries the login for
        ``target_url``'s origin (else the first), so a single-login scaffold keeps
        working unchanged. This replaces '|'-joining several logins into one
        ``AUTH_USERNAME`` / ``AUTH_PASSWORD``.

        Newlines in the target URL are rejected to defend against ``.env``
        injection.
        """
        url = request.target_url
        # Prevent newline injection into .env file. Checked BEFORE _validate_url
        # because urlsplit silently STRIPS \r\n\t before parsing, so a smuggled
        # newline mangles the hostname the SSRF guard would resolve — the guard
        # would then reject it with a misleading DNS error instead of naming the
        # real defect.
        if "\n" in url or "\r" in url:
            raise ValueError("target_url must not contain newline characters")
        _validate_url(url)
        lines = [f"BASE_URL={url}"]

        logins = [self._login_env(e, url) for e in auth_logins(request.auth_config or None)]
        if logins:
            target_origin = login_origin({"login_url": url})
            primary = next((e for e in logins if e["origin"] == target_origin), logins[0])
            lines.append(f"AUTH_LOGIN_URL={primary['login_url']}")
            lines.append(f"AUTH_USERNAME={primary['username']}")
            lines.append(f"AUTH_PASSWORD={primary['password']}")
            if primary["sso_button"]:
                lines.append(f"AUTH_SSO_BUTTON={primary['sso_button']}")
            if primary["success_url"]:
                lines.append(f"AUTH_SUCCESS_URL={primary['success_url']}")
            # Create .auth directory for storageState output (and the logins file)
            auth_dir = package_dir / ".auth"
            auth_dir.mkdir(exist_ok=True)
            if isinstance(request.auth_config, list):
                logins_file = auth_dir / "logins.json"
                # Created 0600 BEFORE the secret is written, never chmod'ed after.
                fd = os.open(logins_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    counts: dict[str, int] = {}
                    for e in logins:
                        counts[e["origin"]] = counts.get(e["origin"], 0) + 1
                    # Keyed by origin (the origin's only login) or "<origin>#<label>"
                    # (several logins share it); each value carries its own `label`.
                    keyed = {
                        logins_file_key(e["origin"], e.get("label"), counts[e["origin"]]): e
                        for e in logins
                    }
                    json.dump(keyed, fh)
                lines.append("AUTH_LOGINS_FILE=.auth/logins.json")

        env_file = package_dir / ".env"
        env_file.write_text("\n".join(lines) + "\n")
        # Tighten perms — defence in depth on top of the 0700 tempdir.
        env_file.chmod(0o600)

        # Ensure playwright.config.ts loads .env *inside* the node process
        # so AUTH_* never appears in any subprocess's /proc/<pid>/environ.
        # The shim is idempotent — only prepended if not already present.
        if logins:
            return self._ensure_dotenv_loader(package_dir)
        return False

    @staticmethod
    def _login_env(entry: AuthLogin, target_url: str) -> dict[str, str]:
        """One login, validated: newline-free values and an SSRF-guarded login_url.

        login_url is a NAVIGATION target the scaffold's auth.setup.ts opens and
        then TYPES THE TENANT'S REAL CREDENTIALS INTO. It is therefore a
        credential-harvesting sink, strictly worse than the plain SSRF that
        `target_url` represents, so it gets exactly the same guard: http(s)
        scheme, real hostname, and the fleet-wide address check
        (`w2v_common.net.validate_public_url`, which resolves DNS, fails CLOSED,
        and blocks RFC 6598 CGNAT 100.64/10 — this estate's Tailnet). Same fix as
        nonagenticai/video_production_service#123. Checked AFTER the newline loop
        because urlsplit silently strips \r\n\t, so a smuggled newline would
        otherwise surface as a misleading DNS error.

        Not guarded: sso_button (a selector, not a URL) and success_url (a
        waitForURL *pattern*, e.g. `**/dashboard`, matched not navigated).
        SSO/IdP hints: AUTH_SSO_BUTTON clicks the IdP button (Keycloak, "Sign in
        with…") when the login form isn't inline; AUTH_SUCCESS_URL is the
        post-login URL the setup waits for. Both optional.
        """
        login_url = strict_login_url(
            str(entry.get("login_url") or entry.get("origin") or target_url), "login_url"
        )
        values = {
            "login_url": login_url,
            "username": str(entry.get("username", "")),
            "password": str(entry.get("password", "")),
            "sso_button": str(entry.get("sso_button", "")),
            "success_url": str(entry.get("success_url", "")),
        }
        env_names = {
            "login_url": "AUTH_LOGIN_URL",
            "username": "AUTH_USERNAME",
            "password": "AUTH_PASSWORD",  # pragma: allowlist secret
            "sso_button": "AUTH_SSO_BUTTON",
            "success_url": "AUTH_SUCCESS_URL",
        }
        for key, value in values.items():
            if "\n" in value or "\r" in value:
                raise ValueError(f"{env_names[key]} must not contain newline characters")
        _validate_url(login_url)
        label = login_label(entry)
        if label:
            values["label"] = label
        values["origin"] = login_origin(entry) or login_origin({"login_url": login_url}) or ""
        return values

    @staticmethod
    def _ensure_dotenv_loader(package_dir: Path) -> bool:
        """Prepend ``import 'dotenv/config';`` to ``playwright.config.ts``.

        This causes the node process running playwright to read ``.env`` at
        config-load time and populate ``process.env`` *after* exec — so the
        credentials never appear in the spawn-time environment that
        ``/proc/<pid>/environ`` exposes to neighbour pods on a shared K3s node.

        Also adds ``dotenv`` to the package's ``dependencies`` so ``npm install``
        pulls it in — which is why this runs BEFORE install. Returns True when it
        added the dependency (a shipped lockfile is then stale). Idempotent: a
        marker comment prevents double-injection.
        """
        marker = "// w2v-test-runner: dotenv injected"
        config_path = package_dir / "playwright.config.ts"
        if config_path.exists():
            existing = config_path.read_text(encoding="utf-8")
            if marker not in existing:
                config_path.write_text(
                    f"{marker}\nimport 'dotenv/config';\n{existing}",
                    encoding="utf-8",
                )

        package_json = package_dir / "package.json"
        if package_json.exists():
            try:
                pkg = json.loads(package_json.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning("package.json is not valid JSON; skipping dotenv injection")
                return False
            deps = pkg.setdefault("dependencies", {})
            if "dotenv" not in deps and "dotenv" not in pkg.get("devDependencies", {}):
                deps["dotenv"] = "^16.4.5"
                package_json.write_text(json.dumps(pkg, indent=2) + "\n", encoding="utf-8")
                return True
        return False

    @staticmethod
    def _find_define_bdd_config(package_dir: Path) -> Path | None:
        """Locate the package file that calls ``defineBddConfig({ ... })``.

        Prefers the conventional ``playwright.config.{ts,mts,cts,js}`` at the
        package root, then scans other root-level TS/JS files. ``node_modules``
        is never traversed.
        """
        preferred = [
            "playwright.config.ts",
            "playwright.config.mts",
            "playwright.config.cts",
            "playwright.config.js",
        ]
        candidates: list[Path] = []
        for name in preferred:
            p = package_dir / name
            if p.exists():
                candidates.append(p)
        # Any remaining root-level config-ish source file (bdd.config.ts etc.).
        for p in sorted(package_dir.glob("*.ts")) + sorted(package_dir.glob("*.js")):
            if p not in candidates and p.name != "fixtures.ts":
                candidates.append(p)
        for path in candidates:
            try:
                if _DEFINE_BDD_CONFIG_RE.search(path.read_text(encoding="utf-8")):
                    return path
            except OSError:
                continue
        return None

    @staticmethod
    def _chainable_test_module(package_dir: Path, spec: str) -> Path | None:
        """Resolve a package's own ``importTestFrom`` specifier to an in-package file.

        Returns the file our fixture can safely ``extend`` (it re-exports
        playwright-bdd's ``test``), or None when the specifier is one we cannot
        vouch for: a bare package specifier (``'my-fixtures'`` — not a file we
        can read or confine), a path escaping the package, or our OWN
        ``fixtures.ts`` (which would be a self-import cycle).
        """
        if not spec.startswith("."):
            return None
        root = package_dir.resolve()
        # _find_define_bdd_config only ever returns a ROOT-level config, so a
        # relative specifier is relative to the package root.
        base = (root / spec).resolve()
        candidates = (
            [base]
            if base.suffix in _TS_MODULE_SUFFIXES
            else [base.with_name(base.name + ext) for ext in _TS_MODULE_SUFFIXES]
        )
        for cand in candidates:
            if not cand.is_relative_to(root) or cand.name == "fixtures.ts":
                return None
            if cand.exists():
                return cand
        return None

    @staticmethod
    def _ensure_track_recorder(package_dir: Path) -> list[str]:
        """Wire the directed-footage track recorder into the downloaded package.

        Returns the list of capture warnings — EMPTY means the recorder is wired
        and this run can emit ``focus_track``/``cursor_track``. A non-empty list
        is the loud signal: it is logged at ERROR and carried into the job's
        ``capture_warnings`` so a track-less capture is diagnosable from the job
        output instead of only from a pod log line nobody reads.

        Why loud: with no track the orchestrator's ``footage_direction`` forces
        ``zoom="off"`` and ``footage_plans`` pops both tracks, so the ONLY code
        path that adds camera movement to a screen recording is dead. A silent
        skip here ships a static film that still reports "completed".

        Two edits, both idempotent:

        1. Write ``fixtures.ts`` into the package root — the ``base.extend``
           auto-fixture that injects the recorder as a page-init script and
           harvests ``window.__w2vTrack`` into a ``w2v-track`` attachment. The
           recorder JS is INLINED into the fixture (no runtime path to resolve).
        2. String-inject ``importTestFrom: './fixtures.ts',`` into the package's
           ``defineBddConfig({ ... })`` call so every generated spec inherits
           the fixture. Guarded by a marker comment so re-runs never double
           inject.

        A package that already sets its own ``importTestFrom`` is CHAINED, not
        skipped: ``fixtures.ts`` is rewritten to extend the package's own test
        export instead of playwright-bdd's base, and ``importTestFrom`` is
        repointed at us. The package keeps its fixtures and the recorder runs.
        Only an unresolvable/unconfinable specifier still fails — loudly.
        """
        try:
            recorder_src = (_TRACK_RECORDER_DIR / "w2v-track.recorder.js").read_text(
                encoding="utf-8"
            )
            fixtures_template = (_TRACK_RECORDER_DIR / "w2v.fixtures.ts").read_text(
                encoding="utf-8"
            )
        except OSError:
            logger.exception("Track recorder assets missing; skipping footage capture wiring")
            return [
                "Track recorder assets are missing from the runner image, so no "
                "focus/cursor track was captured: every clip renders with camera "
                "movement OFF."
            ]

        # Inline the recorder as a valid JS string literal (json.dumps == a
        # double-quoted, escaped JS string).
        fixtures_content = fixtures_template.replace(
            _RECORDER_PLACEHOLDER, json.dumps(recorder_src)
        )
        fixtures_path = package_dir / "fixtures.ts"

        def _write_fixtures(base_spec: str | None) -> None:
            """Write fixtures.ts extending ``base_spec`` (None ⇒ playwright-bdd)."""
            body = (
                fixtures_content
                if base_spec is None
                else fixtures_content.replace(_FIXTURES_BASE_IMPORT, f"from '{base_spec}'", 1)
            )
            fixtures_path.write_text(body, encoding="utf-8")

        config_path = ExecutionEngine._find_define_bdd_config(package_dir)
        if config_path is None:
            _write_fixtures(None)
            message = (
                "No defineBddConfig() call found in the test package, so the "
                "directed-footage track fixture could not be wired via "
                "importTestFrom: this capture emits no focus/cursor track and "
                "every clip renders with camera movement OFF."
            )
            logger.error("%s (package_dir=%s)", message, package_dir)
            return [message]

        existing = config_path.read_text(encoding="utf-8")
        if _TRACK_MARKER in existing:
            # Already wired — idempotent. Rewrite fixtures.ts from the SAME base
            # the marker recorded; blindly rewriting the default would drop the
            # package's own fixture module on the second pass.
            chained = _TRACK_MARKER_BASE_RE.search(existing)
            _write_fixtures(chained.group(1).strip() if chained else None)
            return []

        own = _IMPORT_TEST_FROM_RE.search(existing)
        if own is not None:
            spec = own.group(1)
            chained = ExecutionEngine._chainable_test_module(package_dir, spec)
            if chained is None:
                message = (
                    f"{config_path.name} sets importTestFrom: {spec!r}, which does not "
                    "resolve to a chainable in-package module, so the directed-footage "
                    "track fixture could not be wired: this capture emits no "
                    "focus/cursor track and every clip renders with camera movement OFF."
                )
                logger.error(message)
                _write_fixtures(None)
                return [message]
            # Chain rather than clobber: our fixture extends THEIR test export.
            _write_fixtures(spec)
            injected = _IMPORT_TEST_FROM_RE.sub(
                lambda _m: f"{_TRACK_MARKER} (base: {spec})\n    importTestFrom: './fixtures.ts'",
                existing,
                count=1,
            )
            config_path.write_text(injected, encoding="utf-8")
            logger.info(
                "Wired directed-footage track recorder into %s, chained onto the "
                "package's own fixture module %s",
                config_path.name,
                spec,
            )
            return []

        _write_fixtures(None)
        injected = _DEFINE_BDD_CONFIG_RE.sub(
            lambda m: f"{m.group(0)}\n    {_TRACK_MARKER}\n    importTestFrom: './fixtures.ts',",
            existing,
            count=1,
        )
        config_path.write_text(injected, encoding="utf-8")
        logger.info("Wired directed-footage track recorder into %s", config_path.name)
        return []

    @staticmethod
    def _declared_clip_keys(package_dir: Path, grep_pattern: str | None = None) -> set[str]:
        """The clip keys the PACKAGE declared, read from its Gherkin sources.

        This is the only honest denominator available on the wall-timeout path.
        ``_salvage_raw_from_disk`` "synthesizes one spec per clip" — its report is
        DERIVED FROM THE SURVIVING FOOTAGE, so a 12-scenario run that salvaged 9
        ``.webm`` declares exactly 9 and every report-derived gate compares 9
        against 9. The report cannot be its own denominator; the count has to come
        from something that existed BEFORE the run.

        The ``@clip-<n>`` tags stamped into the ``.feature`` files by
        ``bdd_playwright_service`` are that something, and they are strictly better
        than a count: they are the pipeline-wide join spine (``ScenarioClip.clip_key``,
        the orchestrator's narration join), so the difference against the harvest
        names WHICH beats lost their footage rather than only how many.

        Why not the alternatives:

        * ``npx playwright test --list`` would be authoritative but costs another
          subprocess against a package that has just proven it can hang, and adds a
          new failure mode on the recovery path.
        * Counting ``Scenario:``/``Scenario Outline:`` lines over-counts: an Outline
          with N examples is N tests sharing ONE ``@clip`` tag, hence one clip
          (see ``_collect_clip_specs``).

        Returns an EMPTY set when the package declares no tags at all — a
        plain-Playwright package, or a Gherkin package the generator did not stamp.
        That is "unmeasurable", not "zero declared", and the caller
        (``_salvage_shortfall_warnings``) says so out loud rather than inventing a
        denominator that would make every short package look broken.

        ``grep_pattern``, when given, is the SAME ``--grep`` value passed to
        ``npx playwright test`` for this run (``request.grep_pattern``). The
        declared set is then read PER SCENARIO — feature/scenario tags and title
        joined the way Playwright evaluates ``--grep`` — and a scenario the
        filter excluded contributes NOTHING to the denominator. Without this, a
        filtered run's excluded clips would show up as "NEVER RECORDED" even
        though they were never supposed to run, which is exactly the defect that
        made sending ``grep_pattern`` from a caller unsafe (see
        ``_MAX_DROPPED_CLIP_FRACTION``: enough filtered-out clips could look like
        a majority-missing salvage and fail a perfectly healthy job).
        """
        compiled = re.compile(grep_pattern) if grep_pattern else None
        keys: set[str] = set()
        for feature in package_dir.rglob("*.feature"):
            if _NON_SOURCE_DIRS.intersection(feature.parts):
                continue
            try:
                text = feature.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                # A source we cannot read is a hole in the denominator, so it is
                # reported rather than swallowed: a silently smaller declared set
                # is exactly the fail-open this helper exists to close.
                logger.warning("could not read declared clip tags from %s: %r", feature, exc)
                continue
            keys.update(ExecutionEngine._declared_clip_keys_in_feature(text, compiled))
        return keys

    @staticmethod
    def _declared_clip_keys_in_feature(text: str, grep: re.Pattern[str] | None) -> set[str]:
        """Walk one ``.feature`` file's Gherkin SCENARIO BY SCENARIO.

        A file-wide ``_FEATURE_CLIP_TAG_RE.finditer(text)`` (the pre-fix
        behaviour) cannot tell which ``@clip-<n>`` belongs to which scenario, so
        it cannot honour a filter at all. This walks tag lines, the ``Feature:``
        header (its tags apply to every scenario in the file, per Gherkin), and
        each ``Scenario``/``Scenario Outline`` header, and evaluates ``grep``
        against that one scenario's own tags + title — never the whole file.
        """
        keys: set[str] = set()
        feature_tags: list[str] = []
        pending_tags: list[str] = []
        seen_feature_header = False
        for line in text.splitlines():
            if _GHERKIN_TAG_LINE_RE.match(line):
                pending_tags.extend(_GHERKIN_TAG_RE.findall(line))
                continue
            header = _GHERKIN_SCENARIO_HEADER_RE.match(line)
            if header:
                scenario_tags = feature_tags + pending_tags
                pending_tags = []
                title = header.group(1).strip()
                clip_tags = {tag for tag in scenario_tags if re.fullmatch(r"clip-\d+", tag)}
                if grep is None or ExecutionEngine._grep_selects_scenario(
                    grep, title, scenario_tags
                ):
                    keys.update(clip_tags)
                continue
            stripped = line.strip()
            if not seen_feature_header and stripped.startswith("Feature:"):
                feature_tags = pending_tags
                pending_tags = []
                seen_feature_header = True
                continue
            if stripped:
                # Any other non-blank content (step text, Examples:, a table
                # row, a docstring, ...) is not a tag line — tags only ever
                # immediately precede the header they decorate, so drop
                # whatever we were accumulating rather than let it leak onto
                # a later, unrelated scenario.
                pending_tags = []
        return keys

    @staticmethod
    def _grep_selects_scenario(grep: re.Pattern[str], title: str, tags: list[str]) -> bool:
        """Would ``npx playwright test --grep <pattern>`` run this scenario?

        Playwright evaluates ``--grep`` as a regex search against the full test
        title, and playwright-bdd folds a scenario's Gherkin tags into that
        title (space-joined, ``@``-prefixed, ahead of the scenario name) — see
        the grep-semantics note carried on the orchestrator side of this change.
        Reconstructing that same joined string is what makes this an honest
        stand-in for the real filter rather than a second, divergent one.
        """
        haystack = " ".join(f"@{tag}" for tag in tags)
        if title:
            haystack = f"{haystack} {title}" if haystack else title
        return bool(grep.search(haystack))

    @staticmethod
    def _salvage_shortfall_warnings(
        clips: list[ClipDict],
        *,
        declared_clip_keys: set[str],
        timed_out: bool,
        expect_recording: bool,
    ) -> list[str]:
        """Make a PARTIAL SALVAGE expressible: "N of M", never a redefined M.

        Scope is deliberately narrow — ``timed_out`` only. On every other path
        ``results.json`` is a real report written by the reporter and the existing
        report-derived views (``summary``, ``_count_specs``) are authoritative and
        already exclude legitimately ``skipped`` scenarios. Widening this to clean
        runs would make a Gherkin ``@skip``ped scenario look like lost footage,
        which is a false SHORTFALL on runs that are healthy today.

        Two outcomes, matching ``_guard_partial_footage``'s existing policy rather
        than inventing a second one (same ``_MAX_DROPPED_CLIP_FRACTION``):

        * a minority of the declared clips missing ⇒ COMPLETE, with a warning
          naming every missing key. Hard-failing would void the clips that DID
          survive, and because ``job_manager`` hands the same row back for every
          resubmit under an ``Idempotency-Key`` that failure would be permanent for
          that key — a 12-scenario capture would lose 9 good clips over 3 misses.
          The orchestrator wants those 9: its ``demo_video_record`` re-polls this
          same idempotent job, and its own ``_footage_gap_notes`` already diffs the
          narration's clip tags against what came back, degrading the run and
          falling back to a text card per missing beat.
        * a strict MAJORITY missing ⇒ RAISE. Most of the film never recorded, which
          is the zero-footage case in all but name.

        An empty ``declared_clip_keys`` on a timed-out run is a MEASUREMENT GAP, not
        a clean bill of health, and is warned about as such: the run is a salvage of
        an unknown fraction. Silently returning ``[]`` there is the fail-open this
        function exists to close.
        """
        if not timed_out or not expect_recording:
            return []
        if not declared_clip_keys:
            message = (
                f"PARTIAL SALVAGE of UNKNOWN completeness: this run exceeded its "
                f"wall-clock budget and its report was SYNTHESIZED from the "
                f"{len(clips)} surviving clip(s), but the package declared no "
                f"@clip-<n> tags, so how many scenarios were expected is UNMEASURED. "
                f"Treat the harvest as possibly incomplete."
            )
            logger.warning(message)
            return [message]

        recorded = {str(c.get("clip_key") or "") for c in clips}
        recorded.discard("")
        missing = sorted(
            declared_clip_keys - recorded,
            key=lambda k: (int(k.rsplit("-", 1)[1]), k),
        )
        declared = len(declared_clip_keys)
        if not missing:
            return []
        what = (
            f"PARTIAL SALVAGE: this run exceeded its wall-clock budget, so it is a "
            f"salvage of {declared - len(missing)} of {declared} declared scenario "
            f"clip(s). NEVER RECORDED: {', '.join(missing)}. The report was "
            f"synthesized from the surviving footage, so its own totals cannot show "
            f"the gap; any narration joined to the missing clip_key(s) has no "
            f"footage behind it."
        )
        if len(missing) > declared * _MAX_DROPPED_CLIP_FRACTION:
            raise RuntimeError(
                f"{what} That is most of the capture — failing the job rather than "
                "reporting a mostly-missing run as a complete one (check the wall "
                "budget / the target site's responsiveness)."
            )
        logger.warning(what)
        return [what]

    @staticmethod
    def _footage_yield_warnings(
        clips: list[ClipDict], *, declared: int, expect_recording: bool
    ) -> list[str]:
        """Attribute footage that came back SHORT, not just footage that is absent.

        ``_guard_zero_footage`` fires on `if clips: return` — i.e. only when the
        harvest is EMPTY. A run that declared 6 recordable scenarios and persisted
        1 completes clean, and the orchestrator receives a sixth of the footage
        with no signal at all. So does a run where every clip persisted but each
        is a fraction of a second: the only per-clip check is ``_is_degenerate``,
        which tests ``duration_ms == 0``, so a 0.4s clip publishes as healthy.

        That is measurably the live defect. `reconcile_timeline` sizes its order
        at ~4.0s per scenario and roughly one demo_video render in three fails
        `GROSSLY UNDERFULL` — a beat is PINNED to its measured clip length, so a
        short clip cannot be grown downstream and is silently absorbed.

        This does NOT fail the run. It names the numbers on the way out, in the
        channel the orchestrator already reads, so the shortfall stops being
        invisible and the 4.0s assumption gets measured on every capture.
        """
        if not expect_recording or declared <= 0:
            return []
        warnings: list[str] = []

        if 0 < len(clips) < declared:
            warnings.append(
                f"Footage SHORTFALL: {len(clips)} clip(s) persisted against {declared} "
                f"declared recordable scenario(s). The run completes, but the "
                f"orchestrator receives only "
                f"{len(clips) / declared:.0%} of the footage it ordered."
            )

        durations = [float(c.get("duration_ms") or 0.0) / 1000.0 for c in clips]
        if durations:
            total = sum(durations)
            mean = total / len(durations)
            if mean < _EXPECTED_SECONDS_PER_SCENARIO:
                shortest = ", ".join(
                    f"{c.get('clip_key') or c.get('key') or '?'}={float(c.get('duration_ms') or 0) / 1000.0:.2f}s"
                    for c in sorted(clips, key=lambda c: float(c.get("duration_ms") or 0))[:3]
                )
                warnings.append(
                    f"Footage YIELD below assumption: {len(clips)} clip(s), sum="
                    f"{total:.2f}s, mean={mean:.2f}s/scenario against the "
                    f"{_EXPECTED_SECONDS_PER_SCENARIO:.1f}s the orchestrator assumes when "
                    f"sizing its order; shortest: {shortest}."
                )

        for message in warnings:
            logger.warning(message)
        return warnings

    @staticmethod
    def _track_coverage_warnings(clips: list[ClipDict]) -> list[str]:
        """Warn when clips were published but NONE carries a focus/cursor track.

        The wiring can be applied and the capture still land track-less — a run
        whose steps never matched a selector clicks nothing, so the recorder has
        nothing to record. That is indistinguishable, downstream, from "the
        wiring was never applied": both produce a static film. Attribute it here
        rather than leaving the orchestrator to infer it from an absence.
        """
        if not clips:
            return []
        if any(clip.get("focus_track") or clip.get("cursor_track") for clip in clips):
            return []
        message = (
            f"All {len(clips)} captured clip(s) carry an EMPTY directed-footage track "
            "(no cursor motion, no clicks, no focus). The recorder was wired but the "
            "scenarios never interacted with the page, so the compiler renders every "
            "shot with camera movement OFF — expect a static, duplicate-frame film."
        )
        logger.error(message)
        return [message]

    @staticmethod
    def _discarded_track_warnings(result: Mapping[str, Any]) -> list[str]:
        """Report clips whose RECORDED track this service threw away.

        ``_track_coverage_warnings`` answers "did ANY clip arrive with camera
        direction", which is a different question and silently fails open on this
        one: a run where the recorder captured a track for every scenario but
        ffprobe could not measure 11 of the 12 recordings publishes 11 clips with
        ``focus_track: null`` and — because the 12th kept its track — emits ZERO
        caller-visible warnings. The orchestrator then compiles 11 shots with
        camera movement OFF and has nothing to attribute it to.

        The distinction is not cosmetic. A track-less clip has exactly two causes
        and they need opposite responses:

        * the scenario never interacted with the page — nothing to fix here, and
          the existing all-empty warning already names it;
        * the scenario DID interact, the recorder DID emit a track, and this
          service dropped it (unmeasured dimensions ⇒ scaling by an unknown
          factor, or a payload ``_normalize_track`` rejected) — an ffprobe /
          transcode defect in THIS pod, invisible in the response.

        Log-only would repeat the mistake this method exists to correct:
        ``capture_warnings`` is the one non-fatal channel the orchestrator reads,
        and ``execute()``'s ``finally`` rmtree's the package dir, so the pod log
        is the only other copy. Never fails the run — a discarded track costs
        camera movement, not footage.
        """
        discarded = list(result.get("track_dropped_clips") or [])
        if not discarded:
            return []
        kept = len(result.get("scenario_clips") or [])
        message = (
            f"{len(discarded)} of {kept} published clip(s) RECORDED a directed-footage "
            f"track that this service then DISCARDED (the clip has no measured "
            f"dimensions, or its track payload was malformed): "
            f"{', '.join(sorted(discarded))}. Those shots render with camera movement "
            "OFF despite the scenario having interacted with the page — this is a "
            "capture-side defect (ffprobe/transcode), not an idle scenario."
        )
        logger.error(message)
        return [message]

    @staticmethod
    def _uses_playwright_bdd(package_dir: Path) -> bool:
        """True when the package depends on ``playwright-bdd`` (needs ``bddgen``)."""
        pkg_path = package_dir / "package.json"
        if not pkg_path.exists():
            return False
        try:
            pkg = json.loads(pkg_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
        return "playwright-bdd" in deps

    async def _run_bddgen(self, package_dir: Path) -> None:
        """Run ``npx bddgen`` so playwright-bdd compiles features → ``.features-gen``.

        Gated on the package declaring ``playwright-bdd`` so a plain-Playwright
        package (testDir pointing straight at spec files) is unaffected. A
        non-zero exit aborts the run — an empty generated suite would otherwise
        silently produce zero tests.
        """
        if not self._uses_playwright_bdd(package_dir):
            logger.info("No playwright-bdd dependency in %s, skipping bddgen", package_dir)
            return

        logger.info("Running npx bddgen in %s", package_dir)
        await self._run_node_tool(
            ["npx", "bddgen"], package_dir, label="bddgen", timeout=120, log_stem="bddgen"
        )

    async def _run_tests(
        self,
        package_dir: Path,
        request: TestExecutionRequest,
    ) -> tuple[dict[str, Any], bool]:
        """Invoke npx playwright test with JSON reporter.

        Returns ``(raw_report, timed_out)``. ``timed_out`` is an EXPLICIT
        out-of-band signal, returned as its own value rather than inferred from
        the report or stashed on the engine:

        * Inferring it from the report is exactly what fails. When the salvage
          finds neither ``results.json`` nor a single ``.webm`` it synthesizes an
          EMPTY report, which parses to ``total=0`` — indistinguishable from a
          healthy empty suite. The zero-footage gate then waves through the one
          case it exists for.
        * An instance attribute would be shared mutable state on a
          longer-lived engine; two concurrent ``execute()`` calls would race and
          one run could read the other's timeout flag.
        * A private key smuggled into the report dict would have to survive
          ``_parse_results`` and ``_persist_artifacts`` untouched, and would leak
          into the ``raw`` payload the API hands back to callers.

        A plain second return value has none of those failure modes: it is
        per-call, type-checked, and impossible to drop silently.
        """
        cmd = [
            "npx",
            "playwright",
            "test",
            "--reporter=json,html",
        ]

        if request.grep_pattern:
            cmd.extend(["--grep", request.grep_pattern])

        if request.retries > 0:
            cmd.extend(["--retries", str(request.retries)])

        if request.workers > 1:
            cmd.extend(["--workers", str(request.workers)])

        for browser in request.browsers:
            cmd.extend(["--project", browser])

        # Per-test ceiling (overrides the generated config's `timeout`). This is
        # the clean single-knob lever; the matching action/navigation/expect
        # sub-budgets ride in via PW_* env below.
        cmd.extend(["--timeout", str(request.timeout_per_test_ms)])

        # Trace mode, passed EXPLICITLY in both directions. `trace_on_failure`
        # was accepted by the schema and read nowhere in src/, so
        # `trace_on_failure=false` still recorded traces — the bdd-generated
        # config hardcodes `trace: 'on-first-retry'` — and `_store_report_tree`
        # then uploaded every trace zip into the SHARED object store. The CLI
        # flag overrides the generated config, which is the only lever this
        # service owns (the config itself belongs to bdd_playwright_service).
        # `on-first-retry` is passed rather than omitted on the true branch so
        # the field is load-bearing in BOTH directions and cannot silently rot
        # back into a no-op if the generated config's default ever changes.
        cmd.extend(["--trace", "on-first-retry" if request.trace_on_failure else "off"])

        # Fail-fast sub-budgets, each bounded by the per-test ceiling so an
        # action/navigation can never outlast its own test. These override the
        # generated config's baked defaults (10s/15s/5s) and keep every budget
        # coupled to the single per-test knob the caller controls. Bounding
        # per-action/navigation time is what makes a hung selector fail in
        # seconds instead of pinning the whole per-test timeout.
        per_test_ms = request.timeout_per_test_ms
        action_ms = min(10_000, per_test_ms)
        nav_ms = min(15_000, per_test_ms)
        expect_ms = min(5_000, per_test_ms)

        # Auth credentials are deliberately NOT placed in the subprocess env —
        # /proc/<pid>/environ is readable across pods sharing the node UID on
        # default K3s, exposing AUTH_PASSWORD to any neighbour pod.  Playwright
        # reads them from the .env file written in _configure_target instead
        # (mode 0600 inside a 0700 tempdir).  See security audit H3 (2026-04-27).
        #
        # The PW_* fail-fast budgets ride in the `extra` dict because
        # _subprocess_env strips everything outside its allowlist.
        env: dict[str, str] = {
            "BASE_URL": request.target_url,
            "PLAYWRIGHT_JSON_OUTPUT_NAME": "results.json",
            "PW_TEST_TIMEOUT_MS": str(per_test_ms),
            "PW_ACTION_TIMEOUT_MS": str(action_ms),
            "PW_NAVIGATION_TIMEOUT_MS": str(nav_ms),
            "PW_EXPECT_TIMEOUT_MS": str(expect_ms),
        }

        # Wall-clock backstop only. Per-test time is bounded by --timeout above,
        # so this just caps a pathological suite. Decoupled from the per-test cap
        # (the old `*20/1000` heuristic conflated the two) and kept under the
        # orchestrator's ~1800s re-poll total so the record stage keeps polling.
        # A per-request override (bounded by the schema) lets a caller run a
        # single `@clip` on a short budget instead of paying the full harvest.
        timeout_seconds = request.max_run_seconds or get_settings().execution.max_run_seconds

        logger.info(
            "Running: %s in %s (wall budget %ss)", " ".join(cmd), package_dir, timeout_seconds
        )
        # Redirect stdio to FILES, not pipes. A browser/ffmpeg grandchild that
        # outlives `npx` inherits the stdio FDs; with PIPEs, `communicate()`
        # blocks on pipe-EOF until that grandchild dies — pinning an
        # already-finished run to the full wall budget. Files have no EOF-wait,
        # so we can wait on the process EXIT (`proc.wait()`) and read the report
        # from disk afterwards. The JSON reporter writes ``results.json`` (via
        # PLAYWRIGHT_JSON_OUTPUT_NAME); stdout is only a fallback.
        stdout_path = package_dir / "_pw_stdout.log"
        stderr_path = package_dir / "_pw_stderr.log"
        video_drain_timeout_seconds = get_settings().execution.video_drain_timeout_seconds
        proc: asyncio.subprocess.Process | None = None
        pgid: int | None = None
        wait_outcome = _ReportWaitOutcome()
        try:
            with stdout_path.open("wb") as out_f, stderr_path.open("wb") as err_f:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    cwd=str(package_dir),
                    stdout=out_f,
                    stderr=err_f,
                    env=self._subprocess_env(env),
                    # Own session/group so the npx→node→browser→ffmpeg tree can be
                    # killed as a unit; pgid is captured now, while the leader lives.
                    start_new_session=True,
                )
                pgid = self._capture_pgid(proc)
                try:
                    wait_outcome = await self._wait_for_report_or_exit(
                        proc,
                        package_dir / "results.json",
                        timeout_seconds,
                        video_drain_timeout_seconds=video_drain_timeout_seconds,
                    )
                except TimeoutError:
                    logger.warning(
                        "Playwright exceeded %ss wall budget; killing process group "
                        "+ salvaging footage",
                        timeout_seconds,
                    )
                    await self._terminate_process_group(proc, pgid)
                    # Read the captured stdio BEFORE returning: the `finally` in
                    # execute() rmtree's package_dir, so these two files — the
                    # only record of a browser-launch crash loop — are deleted
                    # unread if we skip this. Budgeted head+tail, never a raw dump.
                    diagnostics = await asyncio.to_thread(
                        self._read_captured_stdio, stdout_path, stderr_path
                    )
                    salvaged = await asyncio.to_thread(self._salvage_raw_from_disk, package_dir)
                    # Log-only. The zero-footage FAILURE decision deliberately does
                    # NOT live here: this branch cannot see how many of the salvaged
                    # specs survive transcode/probe, so a "n_clips > 0" test here
                    # passes runs whose every clip is degenerate straight through to
                    # `completed` with `scenario_clips: []`. The single authoritative
                    # gate is `_guard_zero_footage`, in execute(), AFTER persistence,
                    # which is why the timeout is handed OUT of here as a flag.
                    self._log_salvage_outcome(salvaged, diagnostics)
                    return salvaged, True
        finally:
            # ALWAYS clean up the session group — reap the leader + SIGKILL any
            # lingering browser/ffmpeg grandchildren so they don't zombie under
            # FTR's PID 1 or hold resources between runs.
            if proc is not None:
                await self._terminate_process_group(proc, pgid)

        if proc is None:
            # Pre-existing latent crash: the cleanup above already guards on
            # ``proc is not None``, so reaching here with None meant an
            # AttributeError on ``.returncode`` — a NoneType traceback instead of
            # the real reason the process never started.
            raise RuntimeError("Playwright process was never started")
        returncode = proc.returncode or 0
        stderr_bytes = stderr_path.read_bytes()
        # Playwright exits non-zero when tests fail — that's expected.
        logger.info(
            "Playwright exited with code %d, stderr=%d bytes", returncode, len(stderr_bytes)
        )
        # ⚠️ A NEGATIVE returncode is categorically different from a non-zero one.
        # Positive = "some tests failed", which is normal and must not fail the
        # job. Negative = killed by signal N — and THIS code path is where that
        # happens: `_wait_for_report_or_exit` returns as soon as `results.json`
        # has been stable for one poll and the `finally` then SIGTERM/SIGKILLs the
        # whole process group, which its own docstring says is aimed at "a stuck
        # ffmpeg video muxer". If the muxer was mid-flush, the last clip is
        # truncated or header-less.
        #
        # `returncode` was previously interpolated into ONE error string and used
        # in no decision anywhere, so a signal-killed run whose partial
        # results.json happened to parse read as a completely clean run. This does
        # not fail the job — the report may well be fine — it makes the condition
        # VISIBLE in the channel the orchestrator already reads, next to the
        # SHORTFALL/YIELD warnings.
        signal_warning = self._build_signal_kill_warning(
            returncode, wait_outcome, video_drain_timeout_seconds
        )
        if signal_warning is not None:
            logger.warning(signal_warning)
        report = self._read_playwright_report(package_dir, stdout_path, stderr_bytes, returncode)
        if signal_warning is not None:
            # On the report itself, not on `self`: this engine instance is shared
            # across runs, so instance state would leak this warning into the NEXT
            # capture. `_signal_kill_warning` is read once, at the capture_warnings
            # assembly point, and never persisted.
            report["_signal_kill_warning"] = signal_warning
        return (report, False)

    @staticmethod
    def _build_signal_kill_warning(
        returncode: int,
        wait_outcome: _ReportWaitOutcome,
        video_drain_timeout_seconds: float,
    ) -> str | None:
        """Build the ``_signal_kill_warning`` text, or None if the run is clean.

        Two independent triggers feed the SAME warning channel — a negative
        ``returncode`` (the group was actually killed by signal) and a video
        drain that hit its deadline (the group is *about* to be killed while a
        clip may still be mid-flush) — because both mean the same thing to a
        caller: the harvested footage may be truncated.
        """
        reasons: list[str] = []
        if returncode < 0:
            reasons.append(
                f"Playwright process group was KILLED by signal {-returncode} rather than exiting"
            )
        if wait_outcome.video_drain_timed_out:
            reasons.append(
                f"the {video_drain_timeout_seconds}s video-drain deadline expired while "
                f"at least one .webm clip was still changing size"
            )
        if not reasons:
            return None
        return (
            f"{'; '.join(reasons).capitalize()}. The harvest may be truncated: the "
            f"early-exit path kills the group once results.json looks stable and the "
            f"videos (if any) have drained, which can still catch an ffmpeg muxer "
            f"mid-flush and leave the last clip short or header-less. Compare clip "
            f"durations against the scenario count before trusting this run."
        )

    @staticmethod
    def _read_playwright_report(
        package_dir: Path, stdout_path: Path, stderr_bytes: bytes, returncode: int
    ) -> dict[str, Any]:
        """Read the JSON report from disk (``results.json``), falling back to the
        captured stdout for older reporters that emit JSON to stdout."""
        results_file = package_dir / "results.json"
        if results_file.exists():
            try:
                return json.loads(results_file.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                logger.error("results.json present but unparseable: %s", exc)
        stdout_text = ""
        # A read failure is itself the diagnosis; blanking it reports the far more
        # misleading "no parseable output" instead of e.g. ENOENT/EACCES.
        stdout_note = ""
        try:
            stdout_text = stdout_path.read_text(errors="replace")
        except OSError as exc:
            stdout_note = f"<stdout unreadable: {exc!r}>"
            logger.error("could not read captured stdout %s: %r", stdout_path, exc)
        try:
            return json.loads(stdout_text)
        except json.JSONDecodeError:
            # Both streams are kept (not discarded) and each keeps its own head +
            # tail: node prints the minified code frame first, the diagnosis last.
            stdout_bytes = (stdout_note or stdout_text).encode(errors="replace")
            logger.error(
                "No JSON output from Playwright: %s",
                _subprocess_error_text(stdout_bytes, stderr_bytes, max_chars=4000),
            )
            raise RuntimeError(
                f"Playwright produced no parseable output. Exit code: {returncode}. "
                f"{_subprocess_error_text(stdout_bytes, stderr_bytes, max_chars=2000)}"
            ) from None

    @staticmethod
    async def _wait_for_report_or_exit(
        proc: asyncio.subprocess.Process,
        results_file: Path,
        timeout_seconds: float,
        *,
        poll_interval: float = 2.0,
        video_drain_timeout_seconds: float = 30.0,
    ) -> _ReportWaitOutcome:
        """Return once the process exits OR the JSON report is written, stable,
        AND any in-flight video recordings have finished flushing.

        Playwright writes ``results.json`` from the reporter's end hook — after
        ALL tests finish — but the ``npx`` process can then hang for minutes on
        teardown (a stuck ffmpeg video muxer or browser helper). Once the report
        exists we already have every clip + result, so we stop waiting for `npx`
        itself and let the caller kill the cleanup-stuck tree. This turns a
        record stage that always burned the full wall into one that finishes
        seconds after the last test.

        But ``results.json`` going stable says nothing about whether the
        per-scenario ``.webm`` clips are done being written — Playwright muxes
        video during context/browser TEARDOWN, which is exactly the phase that
        can still be running at this point. So before returning we hand off to
        :meth:`_drain_video_artifacts`, which polls the video files themselves
        (bounded by ``video_drain_timeout_seconds``) rather than trusting
        ``results.json`` as a proxy for "video is flushed".

        Raises ``TimeoutError`` if the process neither exits nor produces a
        stable report before the overall wall (``timeout_seconds``).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        last_size = -1
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=poll_interval)
                return _ReportWaitOutcome()  # process exited on its own
            if results_file.exists():
                size = results_file.stat().st_size
                if size > 0 and size == last_size:
                    logger.info(
                        "results.json written (%d bytes); not waiting for npx cleanup", size
                    )
                    return await ExecutionEngine._drain_video_artifacts(
                        proc,
                        results_file.parent,
                        video_drain_timeout_seconds,
                        poll_interval=poll_interval,
                    )
                last_size = size
            if loop.time() >= deadline:
                raise TimeoutError

    @staticmethod
    async def _drain_video_artifacts(
        proc: asyncio.subprocess.Process,
        package_dir: Path,
        drain_timeout_seconds: float,
        *,
        poll_interval: float = 2.0,
    ) -> _ReportWaitOutcome:
        """Poll ``test-results/**/*.webm`` until every clip's size is stable
        across 2 consecutive polls, the process exits on its own, or the drain
        deadline passes — whichever comes first.

        Called only after ``results.json`` already looks stable, i.e. the tests
        themselves are done; this exists solely to give Playwright's video
        muxer (which runs during context/browser teardown, not test end) a
        bounded window to finish flushing before the caller SIGTERM/SIGKILLs
        the process group. If there are no video files at all (recording
        disabled, or a spec with no scenarios), it returns immediately with no
        added latency.
        """
        video_dir = package_dir / "test-results"
        if not video_dir.exists():
            return _ReportWaitOutcome()
        if not any(video_dir.glob("**/*.webm")):
            return _ReportWaitOutcome()

        loop = asyncio.get_running_loop()
        deadline = loop.time() + drain_timeout_seconds
        last_sizes: dict[Path, int] = {}
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=poll_interval)
                return _ReportWaitOutcome()  # process exited on its own: clean path

            sizes: dict[Path, int] = {}
            for webm in video_dir.glob("**/*.webm"):
                try:
                    sizes[webm] = webm.stat().st_size
                except OSError:
                    continue  # vanished between glob and stat; ignore this poll

            if sizes and all(last_sizes.get(path) == size for path, size in sizes.items()):
                logger.info(
                    "%d video clip(s) stable; drain complete before process-group kill",
                    len(sizes),
                )
                return _ReportWaitOutcome()
            last_sizes = sizes

            if loop.time() >= deadline:
                logger.warning(
                    "Video drain deadline (%ss) exceeded with %d clip(s) still present; "
                    "killing the process group anyway — the last clip(s) may be short or "
                    "header-less",
                    drain_timeout_seconds,
                    len(sizes),
                )
                return _ReportWaitOutcome(video_drain_timed_out=True)

    @staticmethod
    def _capture_pgid(proc: asyncio.subprocess.Process) -> int | None:
        """Resolve the child's process-group id at spawn time, or None.

        Guards the mock-pid broadcast footgun: under unit tests ``proc.pid`` may
        be a mock that is not a real int (a ``MagicMock`` coerces via
        ``__index__`` to 1; a plain ``Mock`` would make ``os.getpgid`` raise
        ``TypeError``, which the bare ``ProcessLookupError`` except would not
        catch). ``os.getpgid(1)`` resolves init's group, so a later
        ``os.killpg`` would broadcast ``kill(-1)`` across the whole process tree
        (e.g. the CI runner agent). Only a real child pid yields a pgid.
        """
        pid = proc.pid
        if not isinstance(pid, int) or pid <= 1:
            return None
        try:
            return os.getpgid(pid)
        except ProcessLookupError:
            return None

    @staticmethod
    async def _terminate_process_group(proc: asyncio.subprocess.Process, pgid: int | None) -> None:
        """Cancellation-SAFE wrapper around :meth:`_reap_process_group`.

        Every call site is a ``finally``/``except`` block, and ``execute()`` runs
        inside a request task that a client disconnect or a SIGTERM drain can
        cancel. Once that happens EVERY subsequent ``await`` re-raises
        ``CancelledError`` immediately — including the grace ``proc.wait()``
        inside the reaper, whose ``contextlib.suppress(TimeoutError,
        ProcessLookupError, ChildProcessError)`` cannot catch it because
        ``CancelledError`` is a ``BaseException``. SIGTERM would land and the
        SIGKILL escalation plus the reap would be SKIPPED, leaking the whole
        npx→node→browser→ffmpeg group under FTR's PID 1 — exactly the leak the
        reaper exists to prevent, on the path it exists for.

        So the reap runs in its OWN task, awaited through :func:`asyncio.shield`:
        the task is not a child of the cancelled scope and completes regardless.
        A STRONG reference is held (the loop keeps only a weak one, and a task
        spawned during teardown can otherwise be GC'd mid-await). The
        cancellation is re-raised, never swallowed.
        """
        task = asyncio.ensure_future(ExecutionEngine._reap_process_group(proc, pgid))
        _PENDING_REAPS.add(task)
        task.add_done_callback(_PENDING_REAPS.discard)
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            logger.warning(
                "Cancelled while reaping process group %s; reap continues in a shielded task",
                pgid,
            )
            raise

    @staticmethod
    async def _reap_process_group(proc: asyncio.subprocess.Process, pgid: int | None) -> None:
        """SIGTERM→(grace)→SIGKILL the subprocess's whole process group, then reap.

        The run is spawned with ``start_new_session=True``, so the
        npx→node→browser→ffmpeg tree shares one process group. ``pgid`` is
        captured at spawn time because once the group *leader* is reaped,
        ``os.getpgid(proc.pid)`` fails — yet orphaned grandchildren can still be
        alive in that group, pinning the stdio pipe or zombie-ing under FTR's
        PID 1. Signalling the GROUP cleans them up; signalling only ``proc``
        leaves them behind. Best-effort throughout; never signals FTR's own
        group.

        Call it through :meth:`_terminate_process_group`, never directly: this
        body is NOT cancellation-safe on its own.
        """
        if pgid is not None and pgid > 1 and pgid != os.getpgid(0):
            signalled = True
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError, PermissionError:
                signalled = False
            if signalled:
                with contextlib.suppress(TimeoutError, ProcessLookupError, ChildProcessError):
                    await asyncio.wait_for(proc.wait(), timeout=3)
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(pgid, signal.SIGKILL)
        # Reap the leader so it doesn't linger as a zombie.
        with contextlib.suppress(TimeoutError, ProcessLookupError, ChildProcessError):
            await asyncio.wait_for(proc.wait(), timeout=5)

    @staticmethod
    def _salvage_raw_from_disk(package_dir: Path) -> dict[str, Any]:
        """Reconstruct a minimal Playwright-JSON dict from footage on disk.

        Used when the run is killed before the JSON reporter flushes (the reporter
        writes ``results.json`` only at run-end). Prefers a ``results.json`` the
        reporter may have written; otherwise scans the ``test-results`` output dir
        for the per-scenario ``.webm`` recordings Playwright writes as each test
        finishes, and synthesizes one spec per clip so ``_parse_results`` /
        ``_collect_clip_specs`` harvest them through the normal path. A clip whose
        on-disk dir name still carries the ``@clip-<n>`` tag joins cleanly
        (``_clip_key_for_spec`` recovers it from the title); the rest fall back to
        a slug key — a best-effort backstop, since fail-fast normally lets the
        suite complete and write a clean tagged ``results.json``.
        """
        results_file = package_dir / "results.json"
        if results_file.exists():
            try:
                parsed = json.loads(results_file.read_text())
                if parsed.get("suites"):
                    return parsed
            except json.JSONDecodeError, OSError:
                pass

        specs: list[dict[str, Any]] = []
        for webm in _salvage_order((package_dir / "test-results").glob("**/*.webm")):
            specs.append(
                {
                    "title": webm.parent.name,
                    "tags": [],
                    "tests": [
                        {
                            "results": [
                                {
                                    "status": "timedOut",
                                    "duration": 0,
                                    "attachments": [{"name": "video", "path": str(webm)}],
                                }
                            ]
                        }
                    ],
                }
            )
        return {"suites": [{"title": "salvaged", "specs": specs, "suites": []}]}

    @staticmethod
    def _read_captured_stdio(stdout_path: Path, stderr_path: Path, max_chars: int = 4000) -> str:
        """Render the run's captured ``_pw_stdout.log`` / ``_pw_stderr.log``.

        The wall-timeout branch used to return without ever opening these, and
        ``execute()``'s ``finally`` then rmtree'd the package dir — so a browser
        that crash-looped on launch was reported ONLY as the guessed "browser
        likely never launched" line, with node's actual stderr deleted unread.

        A read failure is itself the diagnosis (ENOENT means the redirect never
        happened), so it is reported rather than blanked. Size is clamped through
        the same head+tail budgeting every other call site uses: a timed-out
        Playwright run's stderr is routinely megabytes, and this string goes to a
        log line.
        """

        def _read(path: Path) -> bytes:
            try:
                return path.read_bytes()
            except OSError as exc:
                return f"<{path.name} unreadable: {exc!r}>".encode()

        return _subprocess_error_text(_read(stdout_path), _read(stderr_path), max_chars=max_chars)

    @staticmethod
    def _log_salvage_outcome(salvaged: dict[str, Any], diagnostics: str = "") -> int:
        """Log how a timed-out run's footage salvage went; return the clip count.

        Zero footage on disk means the browser almost certainly never launched —
        the FTR#37 ``PLAYWRIGHT_BROWSERS_PATH`` strip, a missing ``bddgen`` (empty
        testDir), or an install failure — NOT a merely slow run. Surface that at
        ``error`` level with the actionable cause so an empty-but-well-formed
        salvage result can't masquerade as a healthy run that just captured
        nothing; a non-empty salvage stays a ``warning``.

        The count MUST recurse. ``_salvage_raw_from_disk`` prefers a
        reporter-flushed ``results.json``, and Playwright NESTS its specs inside
        the describe-block suite ``bddgen`` emits per ``Feature:`` — a top-level
        ``sum(len(s["specs"]))`` reads a partial report holding 3 real clips as 0.
        """
        n_clips = _count_specs(salvaged)
        captured = f" Captured output: {diagnostics}" if diagnostics else ""
        if n_clips == 0:
            logger.error(
                "Salvaged 0 scenario clips from the timed-out run — the browser likely "
                "never launched (check PLAYWRIGHT_BROWSERS_PATH / bddgen / npm install). "
                "The run produced no footage.%s",
                captured,
            )
        else:
            logger.warning(
                "Salvaged %d recorded scenario clip(s) from the timed-out run.%s",
                n_clips,
                captured,
            )
        return n_clips

    def _parse_results(
        self,
        raw: dict[str, Any],
        package_dir: Path,
    ) -> dict[str, Any]:
        """Parse Playwright JSON reporter output into structured results."""
        suites = raw.get("suites", [])
        test_cases: list[TestCaseResult] = []
        total = 0
        passed = 0
        failed = 0
        skipped = 0
        flaky = 0

        for suite in suites:
            self._extract_tests_from_suite(suite, test_cases)

        for tc in test_cases:
            total += 1
            if tc.status == "passed":
                passed += 1
            elif tc.status == "failed":
                failed += 1
            elif tc.status == "skipped":
                skipped += 1
            elif tc.status == "flaky":
                flaky += 1

        pass_rate = (passed / total * 100) if total > 0 else 0.0

        summary = TestSummary(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            flaky=flaky,
            pass_rate=round(pass_rate, 1),
        )

        # Check for HTML report
        report_path = package_dir / "playwright-report" / "index.html"
        report_str = str(report_path) if report_path.exists() else ""

        return {
            "summary": summary.model_dump(),
            "test_results": [tc.model_dump() for tc in test_cases],
            "report_path": report_str,
            "raw": raw,
        }

    @staticmethod
    def _guard_zero_footage(
        result: _JsonObj,
        *,
        timed_out: bool = False,
        expect_recording: bool = True,
    ) -> None:
        """Fail a run that was supposed to record but persisted ZERO clips.

        This is the single authoritative zero-footage gate, and it deliberately
        sits DOWNSTREAM of ``_persist_artifacts`` so it composes with every way a
        run can end up with no footage:

        * the wall-timeout salvage path salvaged nothing;
        * a normal (non-timeout) run recorded nothing;
        * clips WERE salvaged/recorded but every one of them was degenerate and
          got dropped by ``_persist_scenario_clips``.

        The pre-fix guard lived inside ``_run_tests``' ``except TimeoutError``
        branch and keyed off the raw salvage count, so it missed the last two
        entirely — including the case its own PR was written to close.

        Landing such a run at ``completed`` with ``scenario_clips: []`` falsifies
        the orchestrator's ``demo_video_record`` contract: it would harvest zero
        clips, add only soft notes, and compile (and bill for) a footage-free
        narrated-text-card video. Raising lets ``_process_job``'s
        ``except Exception`` mark the job ``failed``.

        The gate fires when the harvest is empty AND either:

        * ``timed_out`` — the run blew its wall budget. This flag is load-bearing
          and CANNOT be replaced by reading the report. On the zero-footage
          timeout path ``_salvage_raw_from_disk`` finds no ``results.json`` and
          no ``.webm``, so it returns a SYNTHETIC empty report; that parses to
          ``total=0``, i.e. "declared nothing", and a report-derived gate takes
          its own exemption branch and lets the run complete with
          ``scenario_clips: []`` — the precise case this gate exists for. A
          timed-out run always intended to record; nothing survived to prove it.
        * or the run declared recordable work. "Declared work" is read from TWO
          independent views and the LARGER wins, so neither view alone can zero
          the gate out: non-skipped test cases in the parsed ``summary``, and a
          RECURSIVE ``_count_specs(raw, recordable_only=True)`` over the raw spec
          tree (Playwright nests specs inside the per-``Feature:`` describe suite
          ``bddgen`` emits — a top-level ``sum()`` reads a real report as 0).

        A suite that declared nothing (zero specs) or skipped everything
        legitimately has no footage to produce and still completes normally —
        this gate must not start 500-ing on an empty suite.

        ``expect_recording=False`` is the explicit escape for a package that does
        not record at all (Playwright ``use.video: 'off'``, or
        ``'retain-on-failure'`` with every test passing): N passing specs and
        zero clips is a HEALTHY run there, and hard-failing it would be a false
        positive that the ``Idempotency-Key`` semantics make permanent. It
        defaults to True because the orchestrator's ``demo_video_record`` stage —
        the reason this service exists — always records.

        Idempotency note (DELIBERATE, do not "fix"): ``job_manager`` returns the
        existing row on an ``Idempotency-Key`` conflict regardless of status, and
        ``routes`` returns it with ``created=False`` without re-dispatching — so a
        job this gate lands at ``failed`` is returned to every resubmit under the
        same key, permanently. That is the intended trade: the stable key exists
        to stop duplicate PAID renders (FTR #72 / orch #690), and the
        orchestrator's ``demo_video_record`` hard-fails into the poison-job refund
        path rather than silently re-billing a retry. A retry is a NEW key.
        """
        clips = result.get("scenario_clips") or []
        if clips:
            return
        if not expect_recording:
            logger.info(
                "Run persisted 0 scenario clips but the caller set "
                "expect_recording=false (non-recording package) — completing normally."
            )
            return

        summary = result.get("summary") or {}
        total = int(summary.get("total", 0) or 0)
        skipped = int(summary.get("skipped", 0) or 0)
        raw = result.get("raw") or {}
        declared = max(total - skipped, _count_specs(raw, recordable_only=True))

        if not timed_out and declared <= 0:
            logger.info(
                "Run persisted 0 scenario clips but declared no recordable tests "
                "(%d total, %d skipped) and did not time out — completing normally.",
                total,
                skipped,
            )
            return

        if timed_out and declared <= 0:
            what = (
                "Test run exceeded its wall-clock budget and salvaged no footage at all "
                "(0 recordable test(s) survived into the report)"
            )
        else:
            what = (
                f"Test run declared {declared} recordable test(s) "
                f"({total} total, {skipped} skipped)"
                + (" and exceeded its wall-clock budget" if timed_out else "")
            )
        raise RuntimeError(
            f"{what} but persisted 0 scenario clips — "
            "the run produced no usable footage (every recording was missing or "
            "degenerate: check PLAYWRIGHT_BROWSERS_PATH / bddgen / npm install, and "
            "the wall budget). Failing the job rather than reporting an empty "
            "successful run."
        )

    @staticmethod
    def _guard_partial_footage(
        result: Mapping[str, Any], *, expect_recording: bool = True
    ) -> list[str]:
        """Make a PARTIAL harvest visible, and fail it once most of it is missing.

        ``_select_publishable_artifact`` DROPS each degenerate clip and
        ``_guard_zero_footage`` returns the moment ``scenario_clips`` is
        non-empty, so a 15-spec suite with 3 duds completed as a clean job with
        12 clips and nothing on the response said 3 shots were missing
        (``capture_warnings`` only ever carried track-coverage defects). The
        orchestrator joins narration 1:1 onto ``clip_key``, so those 3 lines are
        narrated against footage that never arrives.

        Two outcomes, deliberately split (see ``_MAX_DROPPED_CLIP_FRACTION``):

        * up to and including half the clips dropped ⇒ COMPLETE, with a warning
          naming every dropped key. Hard-failing here would void the clean
          siblings, and because ``job_manager`` returns the same row for every
          resubmit under an ``Idempotency-Key`` that failure would be permanent
          for that key — a 1-dud-of-15 run would burn the whole paid capture.
        * a strict MAJORITY dropped ⇒ RAISE. The film would be missing most of
          its shots, which is the zero-footage case in all but name, and the
          message names the keys because ``execute()``'s ``finally`` rmtree's the
          package dir and the pod logs are the only other copy.

        ``expect_recording=False`` opts out entirely, for the same reason it does
        in ``_guard_zero_footage``: a non-recording package has no footage to
        lose.
        """
        dropped = list(result.get("dropped_clips") or [])
        if not dropped or not expect_recording:
            return []
        kept = len(result.get("scenario_clips") or [])
        attempted = kept + len(dropped)
        what = (
            f"{len(dropped)} of {attempted} recorded scenario clip(s) were DROPPED as "
            f"degenerate (unusable footage): {', '.join(sorted(dropped))}. "
            f"Only {kept} clip(s) shipped, so any narration joined to the dropped "
            "clip_key(s) has no footage behind it."
        )
        if len(dropped) > attempted * _MAX_DROPPED_CLIP_FRACTION:
            raise RuntimeError(
                f"{what} That is most of the capture — failing the job rather than "
                "reporting a mostly-empty successful run (check "
                "PLAYWRIGHT_BROWSERS_PATH / ffmpeg / the wall budget)."
            )
        logger.warning(what)
        return [what]

    @staticmethod
    def _unimplemented_option_warnings(request: TestExecutionRequest) -> list[str]:
        """Tell the caller which accepted request fields this service cannot honour.

        ``screenshot_on_failure`` is accepted by the schema and read NOWHERE:
        ``playwright test`` has no ``--screenshot`` CLI lever, and the config
        that would carry ``use.screenshot`` is generated by
        ``bdd_playwright_service``, not by FTR — so it genuinely cannot be wired
        from this repo. Deleting the field would be worse (pydantic drops unknown
        keys silently, so the caller would still be misled), and rejecting it
        would break an existing caller. So it is DECLARED: the notice rides out
        on ``capture_warnings``, the one non-fatal caller-visible channel, and
        not only into pod logs nobody reads.

        Its sibling ``trace_on_failure`` was the same silent no-op and IS now
        wired — ``_run_tests`` passes ``--trace`` explicitly in both directions —
        so it is deliberately absent from this list.
        """
        if request.screenshot_on_failure:
            return []
        message = (
            "screenshot_on_failure=false was IGNORED: this service cannot control "
            "Playwright's screenshot mode (no CLI flag exists and the config is "
            "generated by bdd_playwright_service). Failure screenshots were still "
            "captured and uploaded with the report."
        )
        logger.warning(message)
        return [message]

    async def _persist_artifacts(
        self,
        package_dir: Path,
        persist_dir: Path,
        public_artifact_base: str,
        *,
        raw: dict[str, Any],
        upload_report: bool = True,
    ) -> dict[str, Any]:
        """Upload generated artifacts to the SHARED object store and publish stable URLs.

        ``upload_report=False`` (a login run) withholds the HTML report: it renders
        every step title, and Playwright titles a ``fill`` with the typed value.

        Streams the HTML report tree and one per-scenario video clip (keyed by its
        ``clip-<n>`` join id) into the configured object store under the
        ``{job_id}/`` prefix, so any replica serves any job's artifacts (the
        pre-fix ``FileResponse`` off pod-local disk 404'd cross-pod). ``persist_dir``
        is used only as on-pod scratch where clips are transcoded/probed before
        being streamed up; ``persist_dir.name`` is the job_id (set by the route),
        used to build the object keys and clip URLs.

        ``artifact_paths`` now holds object **keys** (relative to the job prefix),
        never pod-local paths — the routes stream them back via the object store.
        """
        job_id = persist_dir.name
        artifact_paths: dict[str, Any] = {"storage_backend": self.storage.backend}
        report_url = ""
        report_path = ""

        # Fresh local scratch for clip transcode/probe (NOT the durable store).
        shutil.rmtree(persist_dir, ignore_errors=True)
        persist_dir.mkdir(parents=True, exist_ok=True)

        report_dir = package_dir / "playwright-report"
        if upload_report and report_dir.exists():
            report_index_key = await self._store_report_tree(job_id, report_dir)
            artifact_paths["has_report"] = True
            if report_index_key:
                artifact_paths["report_index"] = report_index_key
                report_path = report_index_key
                report_url = f"{public_artifact_base.rstrip('/')}/playwright-report/index.html"

        (
            scenario_clips,
            clip_files,
            dropped_clips,
            track_dropped_clips,
        ) = await self._persist_scenario_clips(raw, package_dir, persist_dir)
        if clip_files:
            artifact_paths["clip_files"] = clip_files

        return {
            "artifact_paths": artifact_paths,
            "report_path": report_path,
            "report_url": report_url,
            "scenario_clips": scenario_clips,
            # Keys the harvest ATTEMPTED and dropped as degenerate. Carried out
            # of here (rather than logged and forgotten) because the shortfall is
            # otherwise indistinguishable from a clean, smaller capture — see
            # `_guard_partial_footage`.
            "dropped_clips": dropped_clips,
            # Keys whose scenario DID record a `w2v-track` that this service then
            # DISCARDED (unmeasured dimensions, or a payload `_normalize_track`
            # rejected). Same reason as `dropped_clips`: the clip publishes with
            # `focus_track: null`, which is byte-identical to "this scenario never
            # interacted with the page" — see `_discarded_track_warnings`.
            "track_dropped_clips": track_dropped_clips,
        }

    async def _store_report_tree(self, job_id: str, report_dir: Path) -> str | None:
        """Stream every file under the Playwright HTML report dir into the object
        store, keyed ``{job_id}/playwright-report/<relpath>``.

        Returns the relative key of ``index.html`` if one was stored (so the
        artifact route can serve ``.../artifacts/playwright-report/index.html``),
        else ``None``. Uploads via ``store_file`` so a large trace zip embedded in
        the report never lands in memory.
        """
        index_key: str | None = None
        for path in sorted(report_dir.rglob("*")):
            if not await asyncio.to_thread(path.is_file):
                continue
            rel = path.relative_to(report_dir).as_posix()
            relpath = f"playwright-report/{rel}"
            content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            await self.storage.store_file(job_id, relpath, path, content_type=content_type)
            if rel == "index.html":
                index_key = relpath
        return index_key

    def _collect_clip_specs(
        self, raw: dict[str, Any], base_dir: Path | None = None
    ) -> list[dict[str, Any]]:
        """Walk the JSON reporter for one video clip per scenario (first wins).

        Returns dicts ``{clip_key, feature, scenario, status, video_path, track,
        lead_trim_seconds}`` — ``track`` being the PARSED ``w2v-track`` payload,
        not a path. ``lead_trim_seconds`` is the video's measured blank lead
        (``_paint_lead_seconds``, ``0.0`` when absent). For a Scenario Outline
        (1:N tests sharing a clip tag) the first test's video is kept so the
        join with narration stays 1:1.

        ``base_dir`` (the extracted package directory) confines every attachment
        SOURCE path so a package cannot name a file outside its own run.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def walk(suite: dict[str, Any]) -> None:
            feature_title = str(suite.get("title") or "")
            for spec in suite.get("specs", []):
                clip_key = _clip_key_for_spec(spec, feature_title)
                found = self._video_test_for_spec(spec, base_dir)
                if found and clip_key not in seen:
                    seen.add(clip_key)
                    # Footage AND status must describe the SAME example row. For a
                    # Scenario Outline (N examples sharing one @clip tag) the video
                    # comes from one test; deriving status from a DIFFERENT test
                    # (tests[-1]) mislabels the clip on the load-bearing @clip join
                    # spine. Take status from the test that owns the kept video.
                    source_test, video_path = found
                    status, _, _ = self._summarize_test_run(source_test.get("results", []))
                    # Same soft-fail demotion as _extract_tests_from_suite: a
                    # non-blank-but-missed clip must not be marked passed — nor
                    # `flaky`, which is what a fail-then-pass summarizes to once
                    # the caller sets retries>=1 (the schema allows 0-10). Gating
                    # on `== "passed"` let the exact same soft-failed selector
                    # escape demotion purely because a retry ladder was enabled.
                    if status in _SOFT_FAIL_DEMOTABLE and self._has_soft_fail(source_test):
                        status = "failed"
                    out.append(
                        {
                            "clip_key": clip_key,
                            "feature": feature_title,
                            "scenario": str(spec.get("title") or ""),
                            "status": status,
                            "video_path": video_path,
                            "track": self._last_track_attachment(spec, base_dir),
                            "lead_trim_seconds": self._paint_lead_seconds(source_test),
                        }
                    )
            for child in suite.get("suites", []):
                walk(child)

        for suite in raw.get("suites", []):
            walk(suite)
        return out

    @staticmethod
    def _video_test_for_spec(
        spec: _JsonObj, base_dir: Path | None = None
    ) -> tuple[_JsonObj, Path] | None:
        """Return ``(test, video_path)`` for the test whose clip we keep.

        The first test (in report order) with an on-disk ``video`` attachment
        wins, keeping the join with narration 1:1. Returning the owning test —
        not just the path — lets the caller read status from the SAME example
        row, so footage and pass/fail never come from different Scenario-Outline
        examples sharing one ``@clip-<n>`` tag.

        ``base_dir`` confines the attachment source path (see
        ``_attachment_path_if_confined``); an attachment escaping it is skipped.
        """
        for test in spec.get("tests", []):
            results = test.get("results", [])
            for result in reversed(results):
                for att in result.get("attachments", []):
                    if att.get("name") == "video" and att.get("path"):
                        candidate = _attachment_path_if_confined(str(att["path"]), base_dir)
                        if candidate is not None:
                            return test, candidate
        return None

    @staticmethod
    def _last_video_attachment(spec: _JsonObj, base_dir: Path | None = None) -> Path | None:
        """Path to the kept ``video`` attachment for a spec, if present."""
        found = ExecutionEngine._video_test_for_spec(spec, base_dir)
        return found[1] if found else None

    @staticmethod
    def _last_track_attachment(spec: _JsonObj, base_dir: Path | None = None) -> _JsonObj | None:
        """The last attempt's parsed ``w2v-track`` focus/cursor payload for a spec.

        Emitted by the injected auto-fixture (``_ensure_track_recorder``) only when
        the scenario recorded at least one click/cursor event; absent ⇒ ``None`` ⇒
        the clip carries no geometry (byte-identical to a pre-instrumentation run).

        Returns the PARSED track, not a path, because the fixture attaches with
        ``body:`` — and Playwright's JSON reporter serializes a body-attachment as
        base64 in a ``body`` field with NO ``path`` key. Only path-attachments (the
        built-in video) carry ``path``. A reader that required ``path`` therefore
        saw zero tracks on every run while video kept working; see
        ``_track_payload_from_attachment`` for the both-shapes decode.

        ``base_dir`` confines the attachment source path of the FILE shape (see
        ``_attachment_path_if_confined``)."""
        for test in spec.get("tests", []):
            results = test.get("results", [])
            for result in reversed(results):
                for att in result.get("attachments", []):
                    if att.get("name") != "w2v-track":
                        continue
                    payload = _track_payload_from_attachment(att, base_dir)
                    if payload is not None:
                        return payload
        return None

    async def _select_publishable_artifact(
        self,
        key: str,
        src: Path,
        clips_dir: Path,
        *,
        convert: bool,
        job_id: str,
        lead_trim_seconds: float = 0.0,
    ) -> tuple[Path, float, int, int, bool, int] | None:
        """Pick the artifact to publish for one clip, or None if it is unusable.

        Returns ``(path, duration_ms, width, height, has_audio, size_bytes)``.

        ``lead_trim_seconds`` is forwarded to ``_convert_to_mp4`` only — the
        raw-webm fallback (below, when conversion fails or is disabled) always
        publishes untrimmed, matching this method's existing "never drop a clip"
        posture: a best-effort trim is not worth risking the fallback path over.

        Degenerate-clip guard. A zero-byte file, or one ffprobe successfully
        reports no duration for, is a DUD — typically a recording whose ffmpeg
        muxer was SIGKILLed mid-write on the wall-timeout salvage path, leaving a
        header-less ``.webm``. Publishing it pushes the defect three stages
        downstream (the compiler renders a black gap the customer pays for), so
        it is attributed here, at capture.

        BUT degeneracy is measured on the POST-conversion artifact, and a
        transcode of a truncated input can emit a useless mp4 from a perfectly
        healthy raw ``.webm``. Dropping there would discard footage the pre-fix
        code published. So when the CONVERTED artifact is degenerate, the raw
        pre-conversion recording is RE-PROBED and published instead; only when
        BOTH are degenerate is the clip dropped. (Dropping, not raising, stays
        deliberate: one dud must not void its clean siblings — the run-level
        zero-footage failure is `_guard_zero_footage`'s job.)

        Since #79 baked ffmpeg/ffprobe into the image, ``_convert_to_mp4`` also
        requires a non-zero-size output before reporting success, so the empty
        transcode is caught one layer earlier and never reaches the fallback as
        a "converted" artifact. That makes the raw-webm fallback the RARE path
        rather than the universal one it was while ffmpeg was missing — but it
        does not make it dead: a truncated salvage recording still fails the
        convert, and this seam is what re-probes it instead of trusting it.

        ``probe_ok`` keeps "ffprobe says no duration" (a genuine dud) apart from
        "ffprobe could not run" (an environment fault); the latter must never
        empty a healthy harvest. With ffprobe now present, ``probe_ok`` is
        normally True — it is the guard for a BROKEN image, not the common case.
        """
        mp4_dst = clips_dir / f"{key}.mp4"
        converted = convert and await _convert_to_mp4(
            src, mp4_dst, lead_trim_seconds=lead_trim_seconds
        )
        if converted:
            stored = mp4_dst
        else:
            # Fall back to the raw recording (WebM); the compositor + the
            # orchestrator re-store path both read it via ffmpeg regardless.
            stored = clips_dir / f"{key}{src.suffix or '.webm'}"
            await asyncio.to_thread(shutil.copyfile, src, stored)

        duration_ms, width, height, has_audio, probe_ok = await _ffprobe_meta(stored)
        size_bytes = stored.stat().st_size
        if not _is_degenerate(size_bytes, duration_ms, probe_ok):
            _warn_if_undecodable(key, stored, size_bytes, duration_ms, width, height)
            return stored, duration_ms, width, height, has_audio, size_bytes

        if converted:
            # The mp4 is a dud — the RAW source may still be fine.
            raw_dst = clips_dir / f"{key}{src.suffix or '.webm'}"
            await asyncio.to_thread(shutil.copyfile, src, raw_dst)
            r_duration, r_width, r_height, r_audio, r_probe_ok = await _ffprobe_meta(raw_dst)
            r_size = raw_dst.stat().st_size
            if not _is_degenerate(r_size, r_duration, r_probe_ok):
                logger.warning(
                    "Converted mp4 for scenario clip %r is degenerate (%d bytes, %.1f ms) "
                    "but the raw recording is healthy (%d bytes, %.1f ms) — publishing the "
                    "RAW clip for job %s instead of dropping it",
                    key,
                    size_bytes,
                    duration_ms,
                    r_size,
                    r_duration,
                    job_id,
                )
                with contextlib.suppress(OSError):
                    mp4_dst.unlink()
                _warn_if_undecodable(key, raw_dst, r_size, r_duration, r_width, r_height)
                return raw_dst, r_duration, r_width, r_height, r_audio, r_size
            with contextlib.suppress(OSError):
                raw_dst.unlink()

        logger.warning(
            "Dropping degenerate scenario clip %r (%s: %d bytes, %.1f ms duration) "
            "for job %s — unusable footage, excluded from scenario_clips",
            key,
            stored.name,
            size_bytes,
            duration_ms,
            job_id,
        )
        return None

    async def _persist_scenario_clips(
        self,
        raw: dict[str, Any],
        package_dir: Path,
        persist_dir: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, str], list[str], list[str]]:
        """Transcode/probe each scenario clip on local scratch, then STREAM it into
        the shared object store; return
        (scenario_clips, clip_files, dropped_keys, track_dropped_keys).

        ``track_dropped_keys`` names every clip that ARRIVED with a recorded
        ``w2v-track`` sidecar which this method then discarded — the recording had
        no measured dimensions, or ``_normalize_track`` rejected the payload. Such
        a clip publishes with ``focus_track``/``cursor_track``/``viewport`` all
        ``None``, i.e. byte-identical to a scenario that never touched the page,
        so without this list the loss is only ever a pod log line.

        ``dropped_keys`` names every clip the harvest ATTEMPTED and
        ``_select_publishable_artifact`` rejected as degenerate. It used to be
        log-only: the caller saw a shorter ``scenario_clips`` list and had no way
        to tell "12 clips because 3 were duds" from "12 clips because the suite
        had 12 scenarios", so the orchestrator joined narration onto clip keys
        that never arrive.

        ``clip_files`` maps ``clip_key`` → the clip's filename; the clip route
        resolves it to object key ``{job_id}/clips/<filename>`` and streams it
        back, so any replica serves any job's clips. The transcode/probe still
        runs on local disk (``persist_dir/clips``) because ffmpeg/ffprobe need a
        real file; the durable copy is the streamed object.
        """
        # Confine every attachment SOURCE path to the extracted package dir: a
        # malicious ``results.json`` cannot make us copy a file from outside it
        # (e.g. ``/app/.env``) into the shared, caller-readable object store.
        specs = self._collect_clip_specs(raw, package_dir)
        if not specs:
            return [], {}, [], []

        clips_dir = persist_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)
        job_id = persist_dir.name
        convert = get_settings().execution.convert_clips_to_mp4

        scenario_clips: list[dict[str, Any]] = []
        clip_files: dict[str, str] = {}
        dropped_keys: list[str] = []
        track_dropped_keys: list[str] = []
        for spec in specs:
            key = spec["clip_key"]
            src = spec["video_path"]
            selected = await self._select_publishable_artifact(
                key,
                src,
                clips_dir,
                convert=convert,
                job_id=job_id,
                lead_trim_seconds=float(spec.get("lead_trim_seconds") or 0.0),
            )
            if selected is None:
                dropped_keys.append(key)
                continue
            stored, duration_ms, width, height, has_audio, size_bytes = selected
            # Directed footage (ORCH-3.2): if the injected recorder emitted a
            # `w2v-track` sidecar for this scenario, normalize its CSS-px geometry
            # into the recorded-video pixel space (matching the VP-2 producer) so
            # the compiler's cursor/auto-zoom filters land on-target. Absent or
            # malformed ⇒ all three stay None ⇒ byte-identical to today.
            focus_track: list[dict[str, Any]] | None = None
            cursor_track: list[dict[str, Any]] | None = None
            viewport: dict[str, int] | None = None
            track = spec.get("track")
            if track is not None:
                if _dims_unmeasured(width, height):
                    # No response-schema field carries a per-clip diagnostic, so
                    # the log IS the channel; the clip still publishes, geometry-less.
                    logger.warning(
                        "Clip %r carries a w2v-track but the recording has NO measured "
                        "dimensions (%sx%s — ffprobe failed or the clip is undecodable) "
                        "for job %s; omitting focus_track/cursor_track/viewport rather "
                        "than emitting all-zero geometry",
                        key,
                        width,
                        height,
                        job_id,
                    )
                    track_dropped_keys.append(key)
                else:
                    normalized = _load_and_normalize_track(track, width, height)
                    if normalized is not None:
                        focus_track, cursor_track, viewport = normalized
                    else:
                        # `_load_and_normalize_track` is total and swallows its
                        # reason, so this is the only place the rejection is
                        # knowable. Record it: the clip ships track-less and is
                        # otherwise indistinguishable from an idle scenario.
                        logger.warning(
                            "Clip %r carries a w2v-track that could not be normalized "
                            "(unreadable/malformed payload) for job %s; publishing the "
                            "clip without focus_track/cursor_track/viewport",
                            key,
                            job_id,
                        )
                        track_dropped_keys.append(key)
            # Stream the transcoded/probed clip into the shared store (never
            # buffered in memory — clips are video). clip_files keeps only the
            # filename; the route joins the `clips/` prefix + `{job_id}`.
            media_type = "video/mp4" if stored.suffix == ".mp4" else "video/webm"
            await self.storage.store_file(
                job_id, f"clips/{stored.name}", stored, content_type=media_type
            )
            clip_files[key] = stored.name
            scenario_clips.append(
                ScenarioClip(
                    clip_key=key,
                    feature=spec["feature"],
                    scenario=spec["scenario"],
                    status=spec["status"],
                    video_url=f"/api/test-runner/{job_id}/clips/{key}/video",
                    duration_ms=duration_ms,
                    size_bytes=size_bytes,
                    has_audio=has_audio,
                    width=width,
                    height=height,
                    focus_track=focus_track,
                    cursor_track=cursor_track,
                    viewport=viewport,
                ).model_dump()
            )
        logger.info(
            "Persisted %d scenario clip(s) for job %s (%d dropped as degenerate, "
            "%d published with a DISCARDED track)",
            len(scenario_clips),
            job_id,
            len(dropped_keys),
            len(track_dropped_keys),
        )
        return scenario_clips, clip_files, dropped_keys, track_dropped_keys

    @staticmethod
    def _extract_error_message(result: dict[str, Any]) -> str:
        """Pull a human-readable error message off a Playwright result entry."""
        error_obj = result.get("error", {})
        if isinstance(error_obj, dict):
            return error_obj.get("message", str(error_obj))
        return str(error_obj)

    @staticmethod
    def _soft_fail_reasons(test: Mapping[str, object]) -> list[str]:
        """The ``description`` of every ``soft-fail`` annotation on *test*.

        ``resilientClick``/``resilientFill`` record WHICH selector missed in the
        annotation description. Playwright itself reports the test ``passed``, so
        it produces no error text — meaning a demoted test would otherwise reach
        the caller as "failed" with an empty ``error``, which the orchestrator
        renders literally as ``<no error>``. The reason exists; nobody was reading it.
        """
        annotations = test.get("annotations")
        if not isinstance(annotations, list):
            return []
        reasons: list[str] = []
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            ann = cast("Mapping[str, object]", annotation)
            if ann.get("type") != "soft-fail":
                continue
            description = ann.get("description")
            reasons.append(str(description).strip() if description else "unspecified soft-fail")
        return reasons

    @staticmethod
    def _paint_lead_seconds(test: Mapping[str, object]) -> float:
        """The clip's measured blank lead, in seconds, from a ``paint-lead``
        annotation — or ``0.0`` when absent (an older package predating the
        hook, a non-recording package, or a hook that never reached the
        annotation push). Mirrors ``_soft_fail_reasons``'s annotation-reading
        shape; see bdd_playwright_service's ``_ensure_scenario_navigation``.

        Malformed input (a non-numeric or negative ``description``) is treated
        the same as absent — the clip publishes untrimmed rather than acting on
        a value that cannot be trusted.
        """
        annotations = test.get("annotations")
        if not isinstance(annotations, list):
            return 0.0
        for annotation in annotations:
            if not isinstance(annotation, dict):
                continue
            ann = cast("Mapping[str, object]", annotation)
            if ann.get("type") != "paint-lead":
                continue
            try:
                lead_ms = float(str(ann.get("description")))
            except TypeError, ValueError:
                return 0.0
            return lead_ms / 1000.0 if lead_ms > 0 else 0.0
        return 0.0

    @staticmethod
    def _has_soft_fail(test: Mapping[str, object]) -> bool:
        """True when a test carries a bdd_playwright ``soft-fail`` annotation.

        A soft-failed selector SOFT-continues the scenario (keeping the clip
        non-blank) instead of throwing, so Playwright still reports the test
        ``passed`` and records the miss only as an annotation
        ``{type: "soft-fail", ...}`` (sibling of ``results`` on the test).
        Without demoting such a test a fully soft-failed suite would report
        pass_rate=100% — a green run that proves nothing.
        """
        annotations = test.get("annotations")
        if not isinstance(annotations, list):
            return False
        for annotation in annotations:
            if (
                isinstance(annotation, dict)
                and cast("Mapping[str, object]", annotation).get("type") == "soft-fail"
            ):
                return True
        return False

    @classmethod
    def _summarize_test_run(cls, run_results: list[dict[str, Any]]) -> tuple[str, str, float]:
        """Compute final status/error/duration for a Playwright test from its attempts.

        Duration semantics: the DECIDING attempt's own duration, never the sum
        across attempts. Playwright's ``results`` are RETRIES of the same test, so
        summing them reported ~3x the real wall time at ``retries=2`` and fed an
        inflated per-scenario duration into the clip metadata. The deciding
        attempt is the one whose status won (the last failure for ``failed``, the
        terminal pass for ``passed``/``flaky``), which for a normal retry ladder
        is the final attempt.

        Status: a terminal ``passed`` CLEARS an earlier ``skipped``. Only
        failed/timedOut/interrupted used to reassign, so ``[skipped, passed]``
        reported ``skipped`` — a passing test excluded from the pass rate.
        """
        status = "passed"
        error = ""
        had_failure = False
        duration_ms = _attempt_duration_ms(run_results[0]) if run_results else 0.0

        for result in run_results:
            r_status = result.get("status", "")
            if r_status in ("failed", "timedOut", "interrupted"):
                had_failure = True
                status = "failed"
                error = cls._extract_error_message(result)
                duration_ms = _attempt_duration_ms(result)
            elif r_status == "skipped" and not had_failure:
                status = "skipped"
                duration_ms = _attempt_duration_ms(result)
            elif r_status == "passed":
                if not had_failure:
                    status = "passed"
                duration_ms = _attempt_duration_ms(result)

        # Flaky detection: multiple attempts, first failed but last passed
        if len(run_results) > 1 and had_failure:
            last_result = run_results[-1]
            if last_result.get("status", "") == "passed":
                status = "flaky"
                duration_ms = _attempt_duration_ms(last_result)

        return status, error, duration_ms

    def _extract_tests_from_suite(
        self,
        suite: dict[str, Any],
        results: list[TestCaseResult],
    ) -> None:
        """Recursively extract test results from nested Playwright suites."""
        suite_title = suite.get("title", "")

        for spec in suite.get("specs", []):
            for test in spec.get("tests", []):
                status, error, duration_ms = self._summarize_test_run(test.get("results", []))
                # A soft-failed selector still reports ``passed`` — demote it so
                # pass_rate reflects the miss (honesty gate). ``flaky`` demotes
                # too: with retries>=1 a fail-then-pass summarizes to ``flaky``,
                # which the old ``== "passed"`` gate waved through, so the SAME
                # soft-failed selector was counted under ``flaky`` instead of
                # ``failed`` purely because the caller enabled a retry ladder.
                if status in _SOFT_FAIL_DEMOTABLE and self._has_soft_fail(test):
                    status = "failed"
                    # Carry the REASON, not just the verdict. The deciding attempt
                    # PASSED in both demotable cases, so any `error` carried over
                    # from an earlier retry is not why this is being failed, and
                    # an un-annotated failure reaches the caller as "<no error>".
                    # The soft-fail annotations name the selectors that missed.
                    reasons = self._soft_fail_reasons(test)
                    error = "soft-failed selector(s): " + "; ".join(reasons)
                results.append(
                    TestCaseResult(
                        name=spec.get("title", ""),
                        feature=suite_title,
                        status=status,
                        duration_ms=duration_ms,
                        error=_clamp_error_text(error, max_chars=1000),
                    )
                )

        # Recurse into child suites
        for child in suite.get("suites", []):
            self._extract_tests_from_suite(child, results)
