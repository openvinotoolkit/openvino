#!/usr/bin/env bash
set -euo pipefail

# GEMMV vs oneDNN benchdnn comparator (universal, per-ISA)
# - Runs benchdnn in two modes: VNNI-only and AMX (if available)
# - Runs our gemmv_bench in four modes: AUTO, AMX-only, VNNI-only, FP32 fallback
# - Aggregates median-of-3 timings and emits a consolidated CSV

THIS_DIR=$(cd "$(dirname "$0")" && pwd)
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

# Optional quick mode: only "main" shapes (speeds up runs on slow hosts)
QUICK=0
if [[ "${1:-}" == "--quick" ]]; then QUICK=1; shift; fi

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

if [[ $QUICK -eq 1 ]]; then
  run_benchdnn_csv "ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI" "$OUT_VNNI" "${shapes_main[@]}"
  run_benchdnn_csv "" "$OUT_AMX" "${shapes_main[@]}"
else
  run_benchdnn_csv "ONEDNN_MAX_CPU_ISA=AVX512_CORE_VNNI" "$OUT_VNNI" "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"
  run_benchdnn_csv "" "$OUT_AMX" "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"
fi

parse_benchdnn() {
  local dims="$1"; local file="$2"
  awk -F, -vD="$dims" 'BEGIN{OFS=","} {
    name=$5; if (name ~ D) {impl=$3; t=$8; gf=$9; split(D,a,/[x:]/); m=a[1]; k=a[2]; n=a[4]; print impl,m,k,n,t,gf;}
  }' "$file"
}

run_ours_agg() {
  local mode="$1"; shift
  local M=$1; local K=$2; local reps=1
  unset GEMMV_SKIP_CALIB GEMMV_SKIP_SELFTEST GEMMV_SKIP_ACCURACY GEMMV_PIN GEMMV_PIN_CORE GEMMV_LOG
  case "$mode" in
    AMX)
      unset GEMMV_DISABLE_AMX; export GEMMV_DISABLE_VNNI=1 ;;
    VNNI)
      export GEMMV_DISABLE_AMX=1; unset GEMMV_DISABLE_VNNI ;;
    FP32)
      export GEMMV_DISABLE_AMX=1; export GEMMV_DISABLE_VNNI=1 ;;
    AUTO|*)
      unset GEMMV_DISABLE_AMX GEMMV_DISABLE_VNNI ;;
  esac
  local times=(); local knames=()
  for ((r=0;r<reps;r++)); do
    local tmp_log; tmp_log=$(mktemp /tmp/gemmv_log.XXXX.md)
    export GEMMV_LOG="$tmp_log"
    # single warmup + one measured run
    "$BIN_GEMMV" "$M" "$K" >>"$tmp_log.stdout" 2>&1 || true
    "$BIN_GEMMV" "$M" "$K" >>"$tmp_log.stdout" 2>&1 || true
    local line=$(grep -E '^\[GEMMV-BENCH\].*\"action\":\"bench_result\"' "$tmp_log" | grep -E '\"gran\":\"per_tensor\"' | grep -E '\"w_type\":\"i8\"' | tail -n 1)
    local t=$(echo "$line" | sed -n 's/.*\"time_ms\":\([0-9.][0-9.]*\).*/\1/p' | tail -n 1)
    local kn=$(echo "$line" | sed -n 's/.*\"kernel\":\"\([^\"]*\)\".*/\1/p' | tail -n 1)
    times+=("$t"); knames+=("${kn:-unknown}")
  done
  IFS=$'\n' sorted=($(printf '%s\n' "${times[@]}" | sed '/^$/d' | sort -n)); unset IFS
  local tuse=""; if [[ ${#sorted[@]} -ge 2 ]]; then tuse=${sorted[1]}; else tuse=${sorted[0]:-}; fi
  if [[ -z "$tuse" ]]; then echo "ours_${mode},unsupported,$M,$K,1,na,na"; return; fi
  local kmed="unknown"; for i in "${!times[@]}"; do if ([[ "${times[$i]}" == "$tuse" ]]); then kmed="${knames[$i]}"; break; fi; done
  awk -vOFS="," -vM="$M" -vK="$K" -vN=1 -vT="$tuse" -vKNAME="$kmed" -vMODE="$mode" 'BEGIN{gf=2.0*M*K/(T*1e6); print "ours_"MODE, KNAME, M, K, N, T, gf}'
}

OUT_CSV_TMP=/tmp/gemmv_vs_benchdnn_ext.csv
OUT_CSV_LOCAL="$THIS_DIR/gemmv_vs_benchdnn_ext.csv"
{
  echo suite,impl,M,K,N,time_ms,gflops
  if [[ $QUICK -eq 1 ]]; then
    for d in "${shapes_main[@]}"; do parse_benchdnn "$d" "$OUT_VNNI" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
    for d in "${shapes_main[@]}"; do parse_benchdnn "$d" "$OUT_AMX" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
    for M in 128 256 512 1024; do run_ours_agg AUTO $M 4096; run_ours_agg AMX $M 4096; run_ours_agg VNNI $M 4096; run_ours_agg FP32 $M 4096; done
  else
    for d in "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"; do parse_benchdnn "$d" "$OUT_VNNI" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
    for d in "${shapes_main[@]}" "${shapes_extra[@]}" "${shapes_tails[@]}"; do parse_benchdnn "$d" "$OUT_AMX" | awk -F, -vOFS="," '{print "benchdnn",$0}'; done
    for M in 128 256 512 1024; do run_ours_agg AUTO $M 4096; run_ours_agg AMX $M 4096; run_ours_agg VNNI $M 4096; run_ours_agg FP32 $M 4096; done
    for M in 144 320; do run_ours_agg AUTO $M 4096; run_ours_agg AMX $M 4096; run_ours_agg VNNI $M 4096; run_ours_agg FP32 $M 4096; done
    for M in 128 256 512 1024; do run_ours_agg AUTO $M 3968; run_ours_agg AMX $M 3968; run_ours_agg VNNI $M 3968; run_ours_agg FP32 $M 3968; done
    for M in 128 256 512 1024; do run_ours_agg AUTO $M 4032; run_ours_agg AMX $M 4032; run_ours_agg VNNI $M 4032; run_ours_agg FP32 $M 4032; done
  fi
} | tee "$OUT_CSV_TMP"

# copy to local folder as well
cp -f "$OUT_CSV_TMP" "$OUT_CSV_LOCAL" 2>/dev/null || true
echo "CSV saved to $OUT_CSV_LOCAL and $OUT_CSV_TMP" >&2
