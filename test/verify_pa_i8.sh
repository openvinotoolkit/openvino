#!/bin/bash
# Regression + verification for sdpa_ocl's PA int8 BY_TOKEN compressed KV cache support.
# Derived 2026-08-07; see the sdpa-ocl-pa-i8-by-token-task memory for why each piece exists.
#
#   ./verify_pa_i8.sh unit     # 17 i8 cases x (micro | ocl | ocl+scalar-V), then the full suite
#   ./verify_pa_i8.sh which    # cliloader: which attention kernel each i8 case dispatches
#   ./verify_pa_i8.sh flags    # dump jit flags per head size (is the V block read really on?)
set -u
BIN=/home/shingyuk/work/openvino-eddy/bin/intel64/Debug/ov_gpu_unit_tests
CLI=~/work/opencl-intercept-layer/install/bin/cliloader
DEV=--device_suffix=1          # MUST be the dGPU; without it you measure the iGPU / sdpa_opt

# The ONLY i8 BY_TOKEN cases that actually reach sdpa_ocl. The other 14 of cases 95-125 dispatch
# paged_attention_opt__single_token/multi_tokens instead (ENABLE_SCORES, head 512, true GENERATE),
# so counting their pass as evidence is wrong. This set is also exactly the set that runs
# sdpa_micro, which makes micro a like-for-like A/B baseline.
OCL_CASES="95 96 98 100 101 102 103 104 106 107 108 118 119 120 121 122 123"
# Pre-existing int4 failures; 129 aborts the run.
SKIP=$(seq 126 139)

f_of() { local p=smoke_paged_attention/paged_attention_test.basic; local out=; \
         for i in $1; do out="$out:$p/$i"; done; echo "${out#:}"; }

case "${1:-unit}" in
unit)
  F=$(f_of "$OCL_CASES")
  echo "### 17 i8 BY_TOKEN cases that reach sdpa_ocl"
  for cfg in "micro:" "ocl:TEST_USE_SDPA_OCL=1" "ocl+scalarV:TEST_USE_SDPA_OCL=1 SDPA_OCL_V_PA_I8_2D=0"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    printf '%-14s ' "$name"
    env OV_GPU_PA_K_TOKEN_MAJOR=1 $envs "$BIN" --gtest_filter="$F" $DEV 2>&1 \
      | grep -oE "\[  (PASSED|FAILED)  \] [0-9]+ tests?" | tr '\n' ' '; echo
  done
  # gtest takes ONE '-' to open the negative list, then plain ':'-separated patterns. Prefixing every
  # entry with '-' (as this did) makes 127+ literal patterns that match nothing, so case 129 still ran
  # and its hard abort killed the process before any summary line -- the row printed EMPTY, which is
  # easy to misread as a pass. Keep the '-' on the first entry only.
  EX=$(for i in $SKIP; do printf ':smoke_paged_attention/paged_attention_test.basic/%s' "$i"; done)
  EX="smoke_paged_attention/paged_attention_test.basic/*:-${EX#:}"
  echo "### full basic suite (129, minus pre-existing int4 126-139)"
  for cfg in "micro tm=1:OV_GPU_PA_K_TOKEN_MAJOR=1" "ocl tm=1:OV_GPU_PA_K_TOKEN_MAJOR=1 TEST_USE_SDPA_OCL=1" \
             "micro tm=0:" "ocl tm=0:TEST_USE_SDPA_OCL=1"; do
    name=${cfg%%:*}; envs=${cfg#*:}
    printf '%-14s ' "$name"
    env $envs "$BIN" --gtest_filter="$EX" $DEV 2>&1 \
      | grep -oE "\[  (PASSED|FAILED)  \] [0-9]+ tests?" | tr '\n' ' '; echo
  done
  echo "### cache-content suites (these observe what the writer/rotate actually wrote)"
  for s in smoke_kv_cache_rotation_content smoke_adaptive_rkv pa_kv_reorder_gpu; do
    printf '%-32s ' "$s"
    OV_GPU_PA_K_TOKEN_MAJOR=1 "$BIN" --gtest_filter="*$s*" $DEV 2>&1 \
      | grep -oE "\[  (PASSED|FAILED)  \] [0-9]+ tests?" | tr '\n' ' '; echo
  done
  ;;
which)
  # A green test proves nothing until you know which kernel ran.
  for c in $(seq 95 125); do
    k=$(OV_GPU_PA_K_TOKEN_MAJOR=1 TEST_USE_SDPA_OCL=1 "$CLI" -d -dv "$BIN" \
          --gtest_filter="*paged*basic/$c" $DEV 2>&1 \
        | grep -oE "(sdpa_ocl__(generate|prefill)|paged_attention_opt__(single_token|multi_tokens))" \
        | sort -u | tr '\n' '+')
    echo "$c ${k:-NONE}"
  done
  ;;
flags)
  # V block read is gated on v_head_size % 64 == 0, so head 32 must fall back to scalar.
  for c in 95 101 102 118 119; do
    d=/tmp/vpaflags$c; rm -rf $d; mkdir -p $d
    OV_GPU_PA_K_TOKEN_MAJOR=1 TEST_USE_SDPA_OCL=1 OV_GPU_DUMP_SOURCES_PATH=$d/ \
      OV_GPU_MAX_KERNELS_PER_BATCH=1 "$BIN" --gtest_filter="*paged*basic/$c" $DEV >/dev/null 2>&1
    m=$(grep -l "IS_PREFILL 0" $d/*.cl 2>/dev/null | head -1)
    [ -z "$m" ] && { echo "case $c: no mixed kernel"; continue; }
    echo "case $c: $(grep -hoE '^#define (HEAD_SIZE|IS_PA_KV_COMPRESSED|USE_2D_BLOCK_IO_V_PA_I8) [0-9]+' $m | tr '\n' ' ')"
  done
  ;;
*) echo "usage: $0 {unit|which|flags}"; exit 1;;
esac
