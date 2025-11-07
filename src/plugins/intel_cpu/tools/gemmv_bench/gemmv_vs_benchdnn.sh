#!/usr/bin/env bash
set -euo pipefail

# GEMMV vs oneDNN benchdnn comparator
# - Runs benchdnn (VNNI-only and AMX) on selected shapes
# - Runs our gemmv_bench (auto-route by ISA) and aggregates median across 3 runs
# - Emits consolidated CSV to stdout and writes to /tmp/gemmv_vs_benchdnn_ext.csv

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
# repo root from src/plugins/intel_cpu/tools/gemmv_bench
REPO_ROOT=$(cd "$THIS_DIR/../../../.." && pwd)

# Locate gemmv_bench
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
  echo "hint: set GEMMV_BIN to the binary, or build from repo root:" >&2
  echo "  cmake -S . -B build -G 'Ninja Multi-Config' -DENABLE_TESTS=OFF -DENABLE_PYTHON=OFF" >&2
  echo "  cmake --build build --config Release --target gemmv_bench -j" >&2
  exit 1
fi

# Locate benchdnn
BENCHDNN="${BENCHDNN_BIN:-}"
if [[ -z "${BENCHDNN}" ]]; then
  if command -v benchdnn >/dev/null 2>&1; then BENCHDNN=$(command -v benchdnn); fi
fi
if [[ -z "${BENCHDNN}" ]]; then
  CAND_LOCAL="$THIS_DIR/onednn_local/build/tests/benchdnn/benchdnn"
  [[ -x "$CAND_LOCAL" ]] && BENCHDNN="$CAND_LOCAL"
fi
if [[ -z "${BENCHDNN}" || ! -x "${BENCHDNN}" ]]; then
  echo "error: benchdnn not found in PATH and fallback not present." >&2
  echo "hints:" >&2
  echo "  • set BENCHDNN_BIN to your benchdnn binary" >&2
  echo "  • install oneDNN system-wide (benchdnn in PATH)" >&2
  echo "  • or build inside this folder: ./build_benchdnn.sh (creates onednn_local/)" >&2
  exit 1
fi

shapes_main=("128x4096:4096x1" "256x4096:4096x1" "512x4096:4096x1" "1024x4096:4096x1")
shapes_extra=("144x4096:4096x1" "320x4096:4096x1")
shapes_tails=("128x3968:3968x1" "256x3968:3968x1" "512x3968:3968x1" "1024x3968:3968x1" \
              "128x4032:4032x1" "256x4032:4032x1" "512x4032:4032x1" "1024x4032:4032x1")

run_benchdnn_csv() {
  local isa_flag="$1"; shift
  local out_file="$1"; shift
  local lines=("$@")
  if [[ -n "$isa_flag" ]]; then
    local isa_val="${isa_flag#*=}"
    ONEDNN_MAX_CPU_ISA="$isa_val" OMP_NUM_THREADS=1 "$BENCHDNN" --mode=p --matmul --dt=u8:s8:f32 "${lines[@]}" | grep -E '^perf,' > "$out_file"
  else
    OMP_NUM_THREADS=1 "$BENCHDNN" --mode=p --matmul --dt=u8:s8:f32 "${lines[@]}" | grep -E '^perf,' > "$out_file"
  fi
}

OUT_VNNI=$(mktemp /tmp/benchdnn_vnni.XXXX.csv)
OUT_AMX=$(mktemp /tmp/benchdnn_amx.XXXX.csv)

run_benchdnn_csv "ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI" "$OUT_VNNI" "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"
run_benchdnn_csv "" "$OUT_AMX" "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"

parse_benchdnn() {
  local dims="$1"; local file="$2"
  awk -F, -vD="$dims" 'BEGIN{OFS=","} {
    name=$5; if (name ~ D) {impl=$3; t=$8; gf=$9; split(D,a,/[x:]/); m=a[1]; k=a[2]; n=a[4]; print impl,m,k,n,t,gf;}
  }' "$file"
}

run_ours_agg() {
  local M=$1; local K=$2; local reps=3
  # Modes: stable (default) vs fast (GEMMV_FAST=1)
  local fast_mode=${GEMMV_FAST:-0}
  local pin_core=${GEMMV_PIN_CORE:-0}
  if [[ "$fast_mode" == "1" ]]; then
    export GEMMV_SKIP_CALIB=1 GEMMV_SKIP_SELFTEST=1 GEMMV_PIN=1 GEMMV_PIN_CORE="$pin_core"
  else
    # Stable: skip calibration for speed, keep selftests enabled
    export GEMMV_SKIP_CALIB=1
    unset GEMMV_SKIP_SELFTEST
    export GEMMV_PIN=1 GEMMV_PIN_CORE="$pin_core"
  fi
  # Skip accuracy compare section for stability/speed; not needed for throughput CSV
  export GEMMV_SKIP_ACCURACY=1
  # Use router defaults inside bench; do not force specific kernels
  # keep per-run log local to tmp, independent of repo layout
  local tmp_log
  tmp_log=$(mktemp /tmp/gemmv_log.XXXX.md)
  export GEMMV_LOG="$tmp_log"
  local times=()
  for ((r=0;r<reps;r++)); do
    # Extra warmup in fast mode to stabilize when skipping selftests/calib
    if [[ "$fast_mode" == "1" ]]; then
      "$BIN_GEMMV" "$M" "$K" >/dev/null 2>&1 || true
    fi
    local before=$(wc -l < "$tmp_log" 2>/dev/null || echo 0)
    "$BIN_GEMMV" "$M" "$K" >/dev/null 2>&1 || true
    local after=$(wc -l < "$tmp_log" 2>/dev/null || echo 0)
    local line=$(sed -n "$((before+1)),$after p" "$tmp_log" | grep -E 'bench_result' | grep -E '"gran":"per_tensor"' | grep -E '"w_type":"i8"' | tail -n 1)
    local t=$(echo "$line" | sed -n 's/.*"time_ms":\([0-9.][0-9.]*\).*/\1/p' | tail -n 1)
    times+=("$t")
  done
  IFS=$'\n' sorted=($(printf '%s\n' "${times[@]}" | sort -n)); unset IFS
  local tuse=${sorted[1]:-${sorted[0]}}
  awk -vOFS="," -vM="$M" -vK="$K" -vN=1 -vT="$tuse" 'BEGIN{gf=2.0*M*K/(T*1e6); print "ours","auto",M,K,N,T,gf}'
}

OUT_CSV_TMP=/tmp/gemmv_vs_benchdnn_ext.csv
OUT_CSV_LOCAL="$THIS_DIR/gemmv_vs_benchdnn_ext.csv"
{
  echo suite,impl,M,K,N,time_ms,gflops
  for d in "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"; do parse_benchdnn "$d" "$OUT_VNNI" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
  for d in "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"; do parse_benchdnn "$d" "$OUT_AMX" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
  for M in 128 256 512 1024; do run_ours_agg $M 4096; done
  for M in 144 320; do run_ours_agg $M 4096; done
  for M in 128 256 512 1024; do run_ours_agg $M 3968; done
  for M in 128 256 512 1024; do run_ours_agg $M 4032; done
} | tee "$OUT_CSV_TMP"

# copy to local folder as well
cp -f "$OUT_CSV_TMP" "$OUT_CSV_LOCAL" 2>/dev/null || true
echo "CSV saved to $OUT_CSV_LOCAL and $OUT_CSV_TMP" >&2
