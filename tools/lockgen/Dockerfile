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

# Regenerate uv.lock with wandb pinned to 0.29.0. The dev extra's opentelemetry-api
# ~=1.33.1 conflicts with wandb 0.29.0 (needs >=1.43.0), so relax it to >=1.33.1.
# Override torch/triton metadata (mirroring the pyproject override-dependencies that keep
# them off) so a wandb-triggered resolution does not reactivate the triton edge.
RUN sed -i 's/"wandb"/"wandb==0.29.0"/g' pyproject.toml \
    && sed -i 's/"opentelemetry-api~=1.33.1"/"opentelemetry-api>=1.33.1"/g' pyproject.toml \
    && cp uv.lock /uv.lock.orig \
    && printf '\n[[tool.uv.dependency-metadata]]\nname = "torch"\nrequires-dist = []\n\n[[tool.uv.dependency-metadata]]\nname = "triton"\nrequires-dist = []\n' >> pyproject.toml \
    && uv lock \
    && cp uv.lock /uv.lock.new \
    && (diff -u /uv.lock.orig /uv.lock.new > /uv.lock.diff || true) \
    && echo "===== SHA256 =====" \
    && sha256sum /uv.lock.new \
    && echo "===== DIFF-BEGIN =====" \
    && cat /uv.lock.diff \
    && echo "===== DIFF-END ====="
