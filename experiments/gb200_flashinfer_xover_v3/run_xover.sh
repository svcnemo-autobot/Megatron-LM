#!/bin/bash
# GB200 one-variable FlashInfer crossover for NVIDIA/Megatron-LM#9111 review 5818311813 finding 2.
# Scratch experiment, not for merge. Code is fixed at main 45eec611 (/opt/xover/src).
# The variable is flashinfer-python: XOVER_FI_NEW (image, main's lock) versus XOVER_FI_OLD (dev's lock).
# Trials alternate new/old twice so drift between trials is visible.
set -uo pipefail
OUT=${XOVER_OUT:-/mnt/artifacts/xover}
SRC=/opt/xover/src
PY=/opt/venv/bin/python
TEST=tests/unit_tests/inference/test_moe_training_matches_inference.py
SEL="test_value_pass_mismatch_within_default_recipe and mxfp8"
SCR=${TMPDIR:-/tmp}/xover
mkdir -p "$OUT" "$SCR/home"
export HOME="$SCR/home"
export PYTHONDONTWRITEBYTECODE=1
trap 'echo "XOVER_EXIT rc=$? $(date -u +%FT%TZ)"' EXIT
echo "XOVER_STAGE start $(date -u +%FT%TZ) new=$XOVER_FI_NEW old=$XOVER_FI_OLD"
nvidia-smi -L || true

arm_env() {
  local v=$1 pp="$SRC"
  if [ "$v" = "$XOVER_FI_OLD" ]; then pp="/opt/xover/fi-$v:$pp"; fi
  if [ -d "/opt/xover/cubin-$v" ]; then
    pp="/opt/xover/cubin-$v:$pp"
    unset FLASHINFER_CUBIN_DIR
  else
    export FLASHINFER_CUBIN_DIR="/opt/xover/cubins-$v"
  fi
  export PYTHONPATH="$pp"
  export FLASHINFER_WORKSPACE_BASE="$SCR/fi-ws-$v"
  mkdir -p "$FLASHINFER_WORKSPACE_BASE"
}

for trial in new1 old1 new2 old2; do
  case $trial in
    new*) v=$XOVER_FI_NEW ;;
    old*) v=$XOVER_FI_OLD ;;
  esac
  (
    arm_env "$v"
    cd "$SRC" || exit 90
    echo "XOVER_STAGE trial=$trial fi=$v $(date -u +%FT%TZ)"
    "$PY" experiments/gb200_flashinfer_xover_v3/report_versions.py
    timeout 2400 "$PY" -m torch.distributed.run --nproc_per_node 4 --nnodes 1 \
      --master_addr localhost --master_port 29500 -m pytest -vs -p no:cacheprovider -rA \
      "$TEST" -k "$SEL"
  ) > "$OUT/$trial.log" 2>&1
  rc=$?
  echo "TRIAL $trial fi=$v rc=$rc"
  grep -E "VER |mxfp8 seed|AssertionError|PASSED|FAILED|SKIPPED|ERROR|[0-9]+ (passed|failed|skipped)" "$OUT/$trial.log" \
    | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-300 | sort | uniq -c | sort -rn | head -40
done
echo "XOVER_DONE $(date -u +%FT%TZ)"
exit 0
