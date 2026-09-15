FROM python:3.12-slim

ENV UV_PYTHON_DOWNLOADS=never \
    UV_NATIVE_TLS=1 \
    UV_NO_GITHUB_FAST_PATH=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv "setuptools" "wheel" "packaging"

WORKDIR /repo
COPY . .

RUN sed -i 's/"wandb"/"wandb==0.29.0"/g' pyproject.toml \
    && sed -i 's/"opentelemetry-api~=1.33.1"/"opentelemetry-api>=1.33.1"/g' pyproject.toml \
    && cp pyproject.toml /pyproject.declared \
    && cp uv.lock /uv.lock.orig \
    && cat tools/lockgen/mamba_ssm.metadata.toml tools/lockgen/torch_triton.metadata.toml >> pyproject.toml \
    && uv lock \
    && cp uv.lock /uv.lock.new \
    && (diff -u /uv.lock.orig /uv.lock.new > /uv.lock.diff || true) \
    && echo "===== SHA256 =====" \
    && sha256sum /uv.lock.new \
    && echo "===== DIFF-BEGIN =====" \
    && cat /uv.lock.diff \
    && echo "===== DIFF-END ====="
