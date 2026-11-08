#!/usr/bin/env bash
# GB200 crossover for NVIDIA/Megatron-LM PR #9111, review 5818311813 finding 2 (experiment only).
# Arms on the same node and GPUs, alternating per trial:
#   venv   image venv (dev uv.lock: flashinfer-python 0.6.18.post1)
#   fi072  same venv with /opt/xover/fi072 first on PYTHONPATH (flashinfer-python 0.7.2.post1,
#          nvidia-cudnn-frontend 1.32.0, nccl-extensions 0.1.0 from main 45eec611 uv.lock)
set -uo pipefail
OUT=${1:-/mnt/artifacts/xover}
TRIALS=${TRIALS:-2}
mkdir -p "$OUT"
SCRATCH=${TMPDIR:-/tmp}/xover
mkdir -p "$SCRATCH/home" "$SCRATCH/cache"
export HOME="$SCRATCH/home" XDG_CACHE_HOME="$SCRATCH/cache"
cd /opt/xover-src || exit 2
tee "$OUT/candidate_sha.txt" < /opt/xover-src/.candidate_sha
nvidia-smi -L 2>&1 | tee "$OUT/gpus.txt"
TEST='tests/unit_tests/inference/test_moe_training_matches_inference.py::TestForwardBackwardMismatch::test_value_pass_mismatch_within_default_recipe'
for trial in $(seq 1 "$TRIALS"); do
  for arm in venv fi072; do
    if [ "$arm" = fi072 ]; then PP=/opt/xover/fi072:/opt/xover-src; else PP=/opt/xover-src; fi
    export FLASHINFER_WORKSPACE_BASE="$SCRATCH/fi-$arm"
    PYTHONPATH="$PP" /opt/venv/bin/python /opt/xover/report_versions.py > "$OUT/$arm-t$trial.versions" 2>&1
    PYTHONPATH="$PP" /opt/venv/bin/python -m torch.distributed.run --nproc_per_node 4 --nnodes 1 \
      --master_addr localhost --master_port 29500 \
      -m pytest -vs -rA -p no:cacheprovider "$TEST" > "$OUT/$arm-t$trial.log" 2>&1
    rc=$?
    echo "$arm t$trial rc=$rc" | tee -a "$OUT/summary.txt"
    grep -aE 'relative error vs BF16' "$OUT/$arm-t$trial.log" | sed "s/^/$arm t$trial /" | tee -a "$OUT/summary.txt"
  done
done
echo XOVER_DONE | tee -a "$OUT/summary.txt"
