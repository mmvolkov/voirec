# Builder stage: glibc-based image so uv can install manylinux wheels
# (onnxruntime and other ML packages have no musl/Alpine wheels)
FROM docker.io/python:3.12-slim-bookworm AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy only dependency files first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

# Install into /app/.venv so shebangs are valid in runtime stage
ENV UV_PROJECT_ENVIRONMENT=/app/.venv
RUN uv sync --frozen --no-dev --no-editable --extra diarize

# ─────────────────────────────────────────────────────────────
# Runtime stage: slim Debian for glibc compat with manylinux wheels
FROM docker.io/python:3.12-slim-bookworm AS runtime

# ffmpeg: required for audio conversion (VoskTranscriber, normalization)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r voirec && useradd -r -g voirec voirec

WORKDIR /app

# Copy installed venv from builder
COPY --from=builder /app/.venv /app/.venv

# Copy source
COPY src/ ./src/

# HuggingFace model cache — will be mounted as volume
ENV HF_HOME=/app/hf_cache
ENV PATH="/app/.venv/bin:$PATH"
# Numba/librosa: кэш JIT в writable директорию
ENV NUMBA_CACHE_DIR=/tmp/numba_cache

# Models are downloaded on first use; cache dir must be writable
RUN mkdir -p /app/hf_cache && chown -R voirec:voirec /app/hf_cache

USER voirec

EXPOSE 8000

CMD ["voirec-api", "--host", "0.0.0.0", "--port", "8000"]
