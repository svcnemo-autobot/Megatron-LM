# Scratch GB200 experiment for NVIDIA/Megatron-LM#9111 (review 5818311813, finding 2). Not for merge.
# Base: the megatron-lm-main profile image of main 45eec611, which locks flashinfer-python 0.7.2.post1.
# One variable: flashinfer-python 0.7.2.post1 (image) versus 0.6.18.post1 (dev's lock). The older
# release is overlaid with --no-deps because every other flashinfer dependency (torch,
# transformer-engine, nvidia-cutlass-dsl, apache-tvm-ffi, cuda-python) is identical in both locks.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG FI_NEW=0.7.2.post1
ARG FI_OLD=0.6.18.post1
ENV XOVER_FI_NEW=${FI_NEW} \
    XOVER_FI_OLD=${FI_OLD}
COPY . /opt/xover/src
RUN set -eux; \
    /opt/venv/bin/python -c "import importlib.metadata as m, sys; v = m.version('flashinfer-python'); print('image flashinfer-python', v); sys.exit(0 if v == '${FI_NEW}' else 1)"; \
    uv pip install --python /opt/venv/bin/python --no-deps --target "/opt/xover/fi-${FI_OLD}" "flashinfer-python==${FI_OLD}"; \
    for v in "${FI_NEW}" "${FI_OLD}"; do \
      if uv pip install --python /opt/venv/bin/python --no-deps --target "/opt/xover/cubin-$v" "flashinfer-cubin==$v"; then \
        echo "flashinfer-cubin $v installed"; \
      else \
        rm -rf "/opt/xover/cubin-$v"; \
        pp=""; \
        if [ "$v" = "${FI_OLD}" ]; then pp="/opt/xover/fi-${FI_OLD}"; fi; \
        PYTHONPATH="$pp" FLASHINFER_CUBIN_DIR="/opt/xover/cubins-$v" /opt/venv/bin/python -c "from flashinfer.artifacts import download_artifacts; download_artifacts()"; \
      fi; \
    done; \
    chmod -R a+rX /opt/xover
