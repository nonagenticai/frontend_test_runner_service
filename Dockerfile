# Multi-stage Dockerfile for frontend_test_runner_service
#
# Single-context build: w2v-common is resolved from pypiserver.test
# via the [[tool.uv.index]] entry in pyproject.toml. The build host must
# be able to reach pypiserver.test (add --add-host if necessary).
# Build with: docker buildx build -f Dockerfile .
#
# Stage 1: Python builder (resolves all deps including w2v-common)
# Stage 2: MS Playwright runtime with Python 3.14 (carries the venv)

# ============================================================================
# Stage 1: Python Builder
# ============================================================================
# Pinned to bookworm explicitly: Stage 2's noble base ships OpenSSL 3.0.x,
# and bookworm ships OpenSSL 3.0.x — same ABI. The default python:3.14-slim
# tag tracks Debian trixie which has OpenSSL 3.3.x, so the _ssl extension
# compiled against trixie's libcrypto cannot resolve OPENSSL_3.3.0 symbols
# at runtime on noble (run 25244359696 hit this).
ARG BASE_REGISTRY=docker.io
# --platform=linux/amd64: the registry.test:5000 mirror stores both arches under
# the same tag; without an explicit platform pin, buildx may resolve the arm64
# manifest if a stale arm64 entry is cached locally on the amd64 builder host
# (hp2 fleet shares one Docker daemon across 6 runners). Burned 2026-05-09 on
# run 25594214389: builder stage failed with "exec /bin/sh: exec format error"
# on apt-get update because the arm64 python layer was pulled.
FROM --platform=linux/amd64 ${BASE_REGISTRY}/library/python:3.14-slim-bookworm AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# Single-context build — no w2v_common sibling
COPY pyproject.toml uv.lock README.md ./

# Resolve and install deps; w2v-common comes from pypiserver.test
RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY alembic.ini ./alembic.ini

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Playwright Runtime with Python 3.14
# ============================================================================
FROM --platform=linux/amd64 mcr.microsoft.com/playwright:v1.50.0-noble

# Install Python 3.14 by copying it from the official python:3.14-slim image
# used by Stage 1 — same binary, same /usr/local/bin/python3.14 path, so the
# venv carried over from Stage 1 resolves cleanly. Avoids the deadsnakes PPA
# dependency that previously broke CI during Canonical Launchpad outages.
# `ffmpeg` (Debian package ships BOTH `ffmpeg` and `ffprobe`) is REQUIRED, not
# optional: _convert_to_mp4() transcodes Playwright's WebM/VP8 recordings to
# H.264 MP4 and _ffprobe_meta() probes every clip's duration/geometry. The
# Playwright base image bundles a PRIVATE `ffmpeg-linux` under /ms-playwright
# for its own recorder — it ships no `ffprobe` at all and puts neither on PATH.
# Without this package both helpers hit FileNotFoundError and degrade silently:
# every clip is published as a WebM mislabelled `video_format="mp4"` with
# duration_ms=0 / 0x0, which the video analyzer cannot decode and the
# blank-clip gate then hard-fails. Cost two days of debugging on 2026-07-30/31.
# DO NOT STRIP.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates curl libpq5 tini ffmpeg \
        libssl3 libffi8 libsqlite3-0 libbz2-1.0 liblzma5 libuuid1 \
        libreadline8t64 libtinfo6 libgdbm6t64 libgdbm-compat4t64 libnsl2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/python3.14 /usr/local/bin/python3.14
COPY --from=builder /usr/local/lib/python3.14 /usr/local/lib/python3.14
COPY --from=builder /usr/local/lib/libpython3.14.so* /usr/local/lib/

RUN ldconfig && \
    ln -sf /usr/local/bin/python3.14 /usr/local/bin/python3 && \
    ln -sf /usr/local/bin/python3.14 /usr/local/bin/python

# Create a dedicated non-root user.
# Do not pin UID 1000 here: the Playwright base image already has a non-root
# user at that UID in some variants, which causes build-time collisions.
RUN useradd -m -U testrunner && \
    mkdir -p /app /app/logs /app/data && \
    chown -R testrunner:testrunner /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=testrunner:testrunner /app/.venv /app/.venv

# Copy application code + alembic config (initContainer runs alembic upgrade head)
COPY --chown=testrunner:testrunner src/ ./src/
COPY --chown=testrunner:testrunner alembic.ini ./alembic.ini

ENV PATH="/app/.venv/bin:$PATH"

USER 1000:1000

# Port 6316 (Frontend Test Runner Service)
EXPOSE 6316

# Kubernetes livenessProbe / readinessProbe in clusters/strato/services/w2v/
# are the source of truth for health. A Dockerfile HEALTHCHECK is dead
# weight under k8s and only causes drift if the probe paths change.

# tini as PID 1 reaps zombies. `npx playwright test` spawns browser/ffmpeg
# grandchildren that we SIGKILL on cleanup; reparented to PID 1 they would
# otherwise accumulate as <defunct> entries because uvicorn does not waitpid().
ENTRYPOINT ["/usr/bin/tini", "--"]

# Use explicit python3.14 binary — the venv's python symlink points to the
# builder's /usr/local/bin/python3.14 which does not exist in Stage 2.
CMD ["/app/.venv/bin/python3.14", "-m", "src.asgi"]
