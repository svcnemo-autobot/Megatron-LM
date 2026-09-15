FROM python:3.12-slim

ENV UV_PYTHON_DOWNLOADS=never \
    UV_NATIVE_TLS=1 \
    UV_NO_GITHUB_FAST_PATH=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv setuptools wheel packaging torch --index-url https://pypi.org/simple

WORKDIR /repo
COPY . .

RUN sed -i 's/"wandb"/"wandb==0.29.0"/g' pyproject.toml \
    && cp uv.lock /uv.lock.orig \
    && UV_NO_BUILD_ISOLATION_PACKAGE="mamba-ssm transformer-engine transformer-engine-torch flash-mla fast-hadamard-transform causal-conv1d" uv lock \
    && cp uv.lock /uv.lock.new \
    && (diff -u /uv.lock.orig /uv.lock.new > /uv.lock.diff || true) \
    && cp pyproject.toml /pyproject.toml.new \
    && echo "=== SHA256 of regenerated uv.lock ===" \
    && sha256sum /uv.lock.new \
    && echo "=== uv.lock.diff size ===" \
    && wc -c /uv.lock.diff \
    && echo "=== wandb entry in new lock ===" \
    && grep -A3 'name = "wandb"' /uv.lock.new | head -8
