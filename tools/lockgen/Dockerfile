FROM python:3.12-slim

ENV UV_PYTHON_DOWNLOADS=never \
    UV_NATIVE_TLS=1 \
    UV_NO_GITHUB_FAST_PATH=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

WORKDIR /repo
COPY . .

# wandb==0.29.0 needs opentelemetry-api>=1.43.0; the dev extra pins ~=1.33.1 (conflict).
# Relax it to >=1.33.1 so the unified resolution can satisfy both.
RUN sed -i 's/"wandb"/"wandb==0.29.0"/g' pyproject.toml \
    && sed -i 's/"opentelemetry-api~=1.33.1"/"opentelemetry-api>=1.33.1"/g' pyproject.toml \
    && cp uv.lock /uv.lock.orig \
    && cat tools/lockgen/mamba_ssm.metadata.toml >> pyproject.toml \
    && uv lock \
    && cp uv.lock /uv.lock.new \
    && (diff -u /uv.lock.orig /uv.lock.new > /uv.lock.diff || true) \
    && sed '/^\[\[tool.uv.dependency-metadata\]\]$/,/^requires-dist = /d' pyproject.toml > /pyproject.toml.pruned 2>/dev/null || cp pyproject.toml /pyproject.toml.pruned \
    && echo "=== SHA256 of regenerated uv.lock ===" \
    && sha256sum /uv.lock.new \
    && echo "=== uv.lock.diff size ===" \
    && wc -c /uv.lock.diff \
    && echo "=== wandb + otel entries in new lock ===" \
    && grep -A2 'name = "wandb"' /uv.lock.new | head -4 \
    && grep -A2 'name = "opentelemetry-api"' /uv.lock.new | grep -A2 'name = "opentelemetry-api"' | head -6
