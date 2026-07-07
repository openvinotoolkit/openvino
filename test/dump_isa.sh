#!/usr/bin/env bash
# Dump compiled GEN ISA (.asm) for sdpa_ocl vs sdpa_micro on the SAME int8 head=64
# prefill test, WITHOUT editing the kernel. Uses IGC ShaderDumpEnable so IGC emits
# the final .asm directly at compile time (raw .isabin from cliloader is not zebin,
# so ocloc disasm can't read it). Static => no observer effect.
#
# Output:
#   test/isa/ocl/   IGC dumps for the TEST_USE_SDPA_OCL=1 run
#   test/isa/micro/ IGC dumps for the =0 run
set -uo pipefail

ROOT=/home/gta/work/openvino-eddy
BIN=$ROOT/bin/intel64/RelWithDebInfo/ov_gpu_func_tests
OUT=$ROOT/test/isa

TEST='smoke/SDPAWithKVCacheTest.MultipleIterationStateful/with_rearrange=1_batch=1_et=f16_num_iter=10_num_groups=1_initial_batch=1_qkv_order=(0.1.2.3)_mask=1_scale=0_causal=0_compressed=1k_head=64v_head=64'

source $ROOT/build/ov_install/setupvars.sh 2>/dev/null

run_one () {
  local tag=$1 useocl=$2
  local dir=$OUT/$tag
  rm -rf "$dir"; mkdir -p "$dir"
  echo "=== [$tag] TEST_USE_SDPA_OCL=$useocl -> $dir ==="
  # OV_GPU_MAX_KERNELS_PER_BATCH=1: compile one kernel per program so each IGC dump
  # has a single entry (no entry_0002 sharing a file with siblings) and the .cl source
  # is dumped per kernel -> unambiguous identification.
  IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$dir" \
    OV_GPU_MAX_KERNELS_PER_BATCH=1 TEST_USE_SDPA_OCL=$useocl \
    "$BIN" --gtest_filter="$TEST" >"$dir/run.log" 2>&1
  echo "  exit=$? ; sdpa .asm files:"
  find "$dir" -name '*.asm' | grep -iE 'sdpa' | sed "s|$dir/||"
}

run_one ocl   1
run_one micro 0

echo
echo "=== sdpa .asm inventory (bytes) ==="
find "$OUT" -name '*.asm' | grep -iE 'sdpa' | xargs -r wc -l | sort -n
