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

# mamba-ssm's setup.py requires CUDA/NVCC (fails on 'bare_metal_version'); supply static
# metadata mirroring the existing lock entry so uv need not run its build backend.
RUN sed -i 's/"wandb"/"wandb==0.29.0"/g' pyproject.toml \
    && cp uv.lock /uv.lock.orig \
    && printf '\n[[tool.uv.dependency-metadata]]\nname = "mamba-ssm"\nversion = "2.3.2.post1"\nrequires-dist = ["apache-tvm-ffi", "einops", "ninja", "packaging", "quack-kernels", "setuptools", "tilelang", "torch; sys_platform == \\x27never\\x27", "transformers", "triton; sys_platform == \\x27never\\x27"]\n' >> pyproject.toml \
    && uv lock \
    && cp uv.lock /uv.lock.new \
    && (diff -u /uv.lock.orig /uv.lock.new > /uv.lock.diff || true) \
    && cp pyproject.toml /pyproject.toml.new \
    && echo "=== SHA256 of regenerated uv.lock ===" \
    && sha256sum /uv.lock.new \
    && echo "=== uv.lock.diff size ===" \
    && wc -c /uv.lock.diff \
    && echo "=== wandb entry in new lock ===" \
    && grep -A3 'name = "wandb"' /uv.lock.new | head -8
