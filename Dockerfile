# syntax=docker/dockerfile:1

# Use the official UV Python base image with Python 3.13 on Debian Bookworm
# UV is a fast Python package manager that provides better performance than pip
# We use the slim variant to keep the image size smaller while still having essential tools
ARG PYTHON_VERSION=3.13
FROM ghcr.io/astral-sh/uv:python${PYTHON_VERSION}-bookworm-slim AS base

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Compile Python source to bytecode (.pyc) during install so the first import
# doesn't pay the compilation cost. This reduces agent cold-start time at the
# expense of a slightly longer build.
ENV UV_COMPILE_BYTECODE=1

# Pin the HuggingFace Hub cache path inside /app (rather than relying on the
# default $HOME/.cache/huggingface) so the turn-detector model download below
# ends up somewhere that actually survives into the final image. Without
# this: the download step runs in the "build" stage as root (HOME=/root),
# but the production stage creates a non-root "appuser" with HOME=/app and
# only COPYs /app forward (line ~75) -- so the model downloaded under
# /root/.cache never makes it into the final image, and the turn-detector
# fails at runtime with "Could not find file ... Make sure you have
# downloaded the model". Setting HF_HOME here (in the shared "base" stage,
# before it diverges into build/production) makes both the build-time
# download and the runtime lookup resolve to the same /app-relative path
# regardless of which Linux user is active, so it's included in the
# COPY --from=build --chown=appuser:appuser /app /app step below with
# correct ownership already applied.
ENV HF_HOME=/app/.cache/huggingface

# --- Build stage ---
# Install dependencies, build native extensions, and prepare the application
FROM base AS build

# Install build dependencies required for Python packages with native extensions
# gcc: C compiler needed for building Python packages with C extensions
# g++: C++ compiler needed for building Python packages with C++ extensions
# python3-dev: Python development headers needed for compilation
# We clean up the apt cache after installation to keep the image size down
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    python3-dev \
  && rm -rf /var/lib/apt/lists/*

# Create a new directory for our application code
# And set it as the working directory
WORKDIR /app

# Copy just the dependency files first, for more efficient layer caching
COPY pyproject.toml uv.lock ./
RUN mkdir -p src

# Install Python dependencies using UV's lock file
# --locked ensures we use exact versions from uv.lock for reproducible builds
# This creates a virtual environment and installs all dependencies
# Ensure your uv.lock file is checked in for consistency across environments
RUN uv sync --locked

# Pre-download any ML models or files the agent needs
# This runs before COPY . . so the download layer is cached across code-only changes.
# The module-level command discovers installed livekit-plugins-* packages without
# loading your agent code.
RUN uv run --module livekit.agents download-files

# Copy all remaining application files into the container
# This includes source code, configuration files, and dependency specifications
# (Excludes files specified in .dockerignore)
COPY . .

# --- Runtime user, shared by both stages below ---
# Build tools (gcc, g++, python3-dev) are not included in the final images.
FROM base AS runtime

# Create a non-privileged user that the app will run under.
# See https://docs.docker.com/build/building/best-practices/#user
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/app" \
    --shell "/sbin/nologin" \
    --uid "${UID}" \
    appuser

# --- API stage (src/api.py) ---
# The outbound-call trigger service -- see the "Outbound Phone Calls" and
# "Deploy to production" sections in README.md. It shares the exact same
# dependencies/venv as the agent worker below (fastapi/uvicorn are already
# installed in the "build" stage), so this just points CMD at uvicorn
# instead. Build it explicitly, since a plain `docker build .` (no
# --target) keeps building the agent worker unchanged, as it always has:
#   docker build --target api -t avery-api .
#   docker run --env-file .env.local -p 8000:8000 avery-api
FROM runtime AS api

COPY --from=build --chown=appuser:appuser /app /app

WORKDIR /app
USER appuser

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

# --- Agent worker stage (src/agent.py) ---
# This is the default target (it's the LAST stage in this file, and Docker
# builds the last stage when no --target is given), so `docker build .`
# behaves exactly as it always has.
FROM runtime AS agent

# Copy the application and virtual environment with correct ownership in a single layer
# This avoids expensive recursive chown and excludes build tools from the final image
COPY --from=build --chown=appuser:appuser /app /app

WORKDIR /app

# Switch to the non-privileged user for all subsequent operations
# This improves security by not running as root
USER appuser

# Run the AgentServer using UV
# UV will activate the virtual environment and run the agent.
# The "start" command tells the AgentServer to connect to LiveKit and begin waiting for jobs.
CMD ["uv", "run", "src/agent.py", "start"]
