#!/usr/bin/env bash
set -euo pipefail

# Universal, robust comparator (sequential, per-shape, per-ISA)
# Produces CSV: suite,mode,kernel,M,K,time_ms,gflops

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

BENCHDNN="${BENCHDNN_BIN:-}"
if [[ -z "${BENCHDNN}" ]]; then
  if command -v benchdnn >/dev/null 2>&1; then BENCHDNN=$(command -v benchdnn); fi
fi
if [[ -z "${BENCHDNN}" ]]; then
  CAND_LOCAL="$THIS_DIR/onednn_local/build/tests/benchdnn/benchdnn"
  [[ -x "$CAND_LOCAL" ]] && BENCHDNN="$CAND_LOCAL"
fi
if [[ -z "${BENCHDNN}" || ! -x "${BENCHDNN}" ]]; then
  echo "error: benchdnn not found." >&2
  exit 1
fi

declare -a SHAPES
SHAPES+=("128 4096" "256 4096" "512 4096" "1024 4096")
SHAPES+=("144 4096" "320 4096")
SHAPES+=("128 4032" "256 4032" "512 4032" "1024 4032")
SHAPES+=("128 3968" "256 3968" "512 3968" "1024 3968")

echo suite,mode,kernel,M,K,time_ms,gflops

run_benchdnn() {
  local isa_flag="$1"; shift
  local M=$1; local K=$2
  local dims="${M}x${K}:${K}x1"
  local out
  if [[ -n "$isa_flag" ]]; then
    out=$(ONEDNN_MAX_CPU_ISA="$isa_flag" OMP_NUM_THREADS=1 "$BENCHDNN" --mode=p --matmul --dt=u8:s8:f32 "$dims" | grep -E '^perf,') || true
  else
    out=$(OMP_NUM_THREADS=1 "$BENCHDNN" --mode=p --matmul --dt=u8:s8:f32 "$dims" | grep -E '^perf,') || true
  fi
  # perf,cpu,impl,,...,Gops,+ctime,-time,-Gflops,0time,0Gflops
  if [[ -n "$out" ]]; then
    local impl=$(echo "$out" | awk -F, '{print $3}')
    local t=$(echo "$out" | awk -F, '{print $9}')
    local gf=$(echo "$out" | awk -F, '{print $10}')
    echo benchdnn,$impl,$M,$K,1,$t,$gf
  else
    echo benchdnn,unsupported,$M,$K,1,na,na
  fi
}

run_ours() {
  local mode="$1"; shift
  local M=$1; local K=$2
  unset GEMMV_LOG
  case "$mode" in
    AMX) unset GEMMV_DISABLE_AMX; export GEMMV_DISABLE_VNNI=1;;
    VNNI) export GEMMV_DISABLE_AMX=1; unset GEMMV_DISABLE_VNNI;;
    FP32) export GEMMV_DISABLE_AMX=1; export GEMMV_DISABLE_VNNI=1;;
    AUTO|*) unset GEMMV_DISABLE_AMX GEMMV_DISABLE_VNNI;;
  esac
  local tmp_log; tmp_log=$(mktemp /tmp/gemmv_log.XXXX.md)
  export GEMMV_LOG="$tmp_log"
  OMP_NUM_THREADS=1 "$BIN_GEMMV" "$M" "$K" >/dev/null 2>&1 || true
  local line=$(grep -E '^\[GEMMV-BENCH\].*\"action\":\"bench_result\"' "$tmp_log" | tail -n 1)
  if [[ -n "$line" ]]; then
    local t=$(sed -n 's/.*\"time_ms\":\([0-9.][0-9.]*\).*/\1/p' <<<"$line" | tail -n 1)
    local gf=$(sed -n 's/.*\"gflops\":\([0-9.][0-9.]*\).*/\1/p' <<<"$line" | tail -n 1)
    local kn=$(sed -n 's/.*\"kernel\":\"\([^"]*\)\".*/\1/p' <<<"$line" | tail -n 1)
    echo ours_${mode},$kn,$M,$K,1,${t:-na},${gf:-na}
  else
    echo ours_${mode},unsupported,$M,$K,1,na,na
  fi
}

for s in "${SHAPES[@]}"; do
  read -r M K <<<"$s"
  run_benchdnn "AVX512_CORE_VNNI" $M $K
  run_benchdnn "" $M $K
  run_ours AUTO $M $K
  run_ours AMX $M $K
  run_ours VNNI $M $K
  run_ours FP32 $M $K
done

