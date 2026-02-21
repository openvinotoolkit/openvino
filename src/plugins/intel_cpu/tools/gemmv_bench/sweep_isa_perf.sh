#!/usr/bin/env bash
set -euo pipefail

# Sweep GEMMV perf over shapes for ISA modes (AMX, VNNI, FP32)
# Outputs CSV: suite,mode,kernel,M,K,time_ms,gflops

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "$THIS_DIR/../../../.." && pwd)

BIN_GEMMV_DEFAULTS=(
  "$THIS_DIR/build/bin/gemmv_bench"
  "$THIS_DIR/build_rel/bin/gemmv_bench"
  "$REPO_ROOT/build/bin/Release/gemmv_bench"
  "$REPO_ROOT/build/bin/gemmv_bench"
  "$REPO_ROOT/build/gemmv_bench/bin/gemmv_bench"
)
BIN_GEMMV="${GEMMV_BIN:-}"
if [[ -z "${BIN_GEMMV}" ]]; then
  if command -v gemmv_bench >/dev/null 2>&1; then BIN_GEMMV=$(command -v gemmv_bench); fi
fi
if [[ -z "${BIN_GEMMV}" ]]; then
  for c in "${BIN_GEMMV_DEFAULTS[@]}"; do [[ -x "$c" ]] && BIN_GEMMV="$c" && break; done
fi
if [[ -z "${BIN_GEMMV}" || ! -x "${BIN_GEMMV}" ]]; then
  echo "error: gemmv_bench not found." >&2
  exit 1
fi

# Shapes: targets + tails (from compare script)
declare -a SHAPES
SHAPES+=("128 4096" "256 4096" "512 4096" "1024 4096")
SHAPES+=("144 4096" "320 4096")
SHAPES+=("128 3968" "256 3968" "512 3968" "1024 3968")
SHAPES+=("128 4032" "256 4032" "512 4032" "1024 4032")

# Run once and extract last bench_result (per-tensor i8) as CSV fields
run_one() {
  local mode="$1"; shift
  local M="$1"; local K="$2"; shift 2
  # Use a temp log to avoid mixing runs
  local tmp_log; tmp_log=$(mktemp /tmp/gemmv_log.XXXX.md)
  export GEMMV_LOG="$tmp_log"
  export OMP_NUM_THREADS=1
  "$BIN_GEMMV" "$M" "$K" >>"$tmp_log.stdout" 2>&1 || true
  local line
  line=$(grep -E '^\[GEMMV-BENCH\].*"action":"bench_result"' "$tmp_log" | grep -E '"gran":"per_tensor"' | grep -E '"w_type":"i8"' | tail -n 1)
  local t gf kn
  t=$(sed -n 's/.*"time_ms":\([0-9.][0-9.]*\).*/\1/p' <<<"$line" | tail -n 1)
  gf=$(sed -n 's/.*"gflops":\([0-9.][0-9.]*\).*/\1/p' <<<"$line" | tail -n 1)
  kn=$(sed -n 's/.*"kernel":"\([^"]*\)".*/\1/p' <<<"$line" | tail -n 1)
  if [[ -z "$t" || -z "$gf" ]]; then
    echo "gemmv,$mode,unsupported,$M,$K,na,na"
  else
    echo "gemmv,$mode,$kn,$M,$K,$t,$gf"
  fi
}

echo "suite,mode,kernel,M,K,time_ms,gflops"
# AMX
for s in "${SHAPES[@]}"; do
  read -r M K <<<"$s"
  unset GEMMV_DISABLE_AMX
  export GEMMV_DISABLE_VNNI=1
  run_one AMX "$M" "$K"
done
# VNNI
for s in "${SHAPES[@]}"; do
  read -r M K <<<"$s"
  export GEMMV_DISABLE_AMX=1
  unset GEMMV_DISABLE_VNNI
  run_one VNNI "$M" "$K"
done
# FP32 fallback
for s in "${SHAPES[@]}"; do
  read -r M K <<<"$s"
  export GEMMV_DISABLE_AMX=1
  export GEMMV_DISABLE_VNNI=1
  run_one FP32 "$M" "$K"
done
