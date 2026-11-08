# One-variable GB200 crossover for NVIDIA/Megatron-LM PR #9111, review 5818311813 finding 2
# (test_value_pass_mismatch_within_default_recipe[mxfp8]).
# Base: the exact CI GB200 image of merge preview 45ecfd8e (dev dependency set, flashinfer-python 0.6.18.post1).
# Variable: flashinfer-python 0.7.2.post1 and its dependency pins from main 45eec611 uv.lock; nothing else changes.
# Experiment-only image; not a proposed dependency change.
FROM 766267172432.dkr.ecr.us-east-2.amazonaws.com/megatron-lm:45ecfd8e31ac67bf30b1022ffd29a275868abccc-gb-gpu
COPY constraints.txt /tmp/xover-constraints.txt
RUN uv pip install --python /opt/venv/bin/python --constraint /tmp/xover-constraints.txt \
        "flashinfer-python==0.7.2.post1" "nvidia-cudnn-frontend==1.32.0" \
    && uv pip list --python /opt/venv/bin/python | grep -i -E 'flashinfer|cudnn-frontend|nccl|cutlass|tvm'
