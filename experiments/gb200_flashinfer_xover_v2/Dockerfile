# GB200 one-variable crossover image for NVIDIA/Megatron-LM PR #9111, review 5818311813 finding 2.
# Experiment only: not a proposed dependency change and not for merge.
#
# BASE_IMAGE        this Session's megatron-lm-dev profile arm64 dependency image, by immutable digest
#                   (dev uv.lock: flashinfer-python 0.6.18.post1, nvidia-cudnn-frontend 1.29.0).
# /opt/xover-src    the candidate source at CANDIDATE_SHA, fetched as a pinned public commit.
# /opt/xover/fi072  flashinfer-python 0.7.2.post1 plus the only other lock entries that differ in its
#                   dependency closure between dev and main 45eec611 (nvidia-cudnn-frontend 1.32.0,
#                   nccl-extensions 0.1.0), installed --no-deps beside the unchanged venv.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
ARG CANDIDATE_REPO=https://github.com/svcnemo-autobot/Megatron-LM.git
ARG CANDIDATE_SHA=5abb633162fc2865039087bcd9f443ff5c18376a
USER root
COPY report_versions.py run_xover.sh /opt/xover/
RUN set -eux; \
    git init -q /opt/xover-src; \
    git -C /opt/xover-src fetch -q --depth 1 "${CANDIDATE_REPO}" "${CANDIDATE_SHA}"; \
    git -C /opt/xover-src checkout -q --detach FETCH_HEAD; \
    test "$(git -C /opt/xover-src rev-parse HEAD)" = "${CANDIDATE_SHA}"; \
    echo "${CANDIDATE_SHA}" > /opt/xover-src/.candidate_sha; \
    rm -rf /opt/xover-src/.git; \
    uv pip install --python /opt/venv/bin/python --no-deps --target /opt/xover/fi072 \
        "flashinfer-python==0.7.2.post1" "nvidia-cudnn-frontend==1.32.0" "nccl-extensions==0.1.0"; \
    /opt/venv/bin/python /opt/xover/report_versions.py; \
    PYTHONPATH=/opt/xover/fi072 /opt/venv/bin/python /opt/xover/report_versions.py; \
    chmod -R a+rX /opt/xover /opt/xover-src; \
    chmod a+rx /opt/xover/run_xover.sh
