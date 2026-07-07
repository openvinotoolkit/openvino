#!/usr/bin/env bash
# Probe which DPAS matrix-mad builtins exist on bmg (Battlemage/Xe2). Used to decide whether
# int8 can be fed to DPAS directly or must be dequantized to f16 first. Findings (2026-07-05):
#   i8_i8_matrix_mad_k32          : AVAILABLE (int8 x int8 -> int32)
#   hf_i8 / i8_hf / hf_u8 / u8_hf / bf_i8 matrix_mad : ABSENT (no mixed-precision DPAS)
# => half-score x raw-int8-V direct is HARDWARE-BLOCKED; int8 must widen to f16 before dpas.
# Also: intel_sub_group_2d_block_read_transform_8b_16r16x1c does NOT exist (only _32r16x1c);
#       plain 8b reads require col multiple of 4 (_8b_16r16x4c). See sdpa-ocl-int8-perf.
set -uo pipefail
DEV=bmg
TMP="$(cd "$(dirname "$0")" && pwd)/dump/_apiprobe"; rm -rf "$TMP"; mkdir -p "$TMP"

echo "=== int8 DPAS (i8_i8_mad_k32) ==="
cat > "$TMP/i8.cl" <<'EOF'
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(global int* acc, global int* a, global int* b){
    int lane=get_sub_group_local_id();
    int8 A=((global int8*)a)[lane], B=((global int8*)b)[lane], C=0;
    C=intel_sub_group_i8_i8_matrix_mad_k32(A,B,C);
    ((global int8*)acc)[lane]=C;
}
EOF
ocloc compile -file "$TMP/i8.cl" -device $DEV -out_dir "$TMP" >/dev/null 2>&1 \
  && echo "  i8_i8_matrix_mad_k32 : AVAILABLE" || echo "  i8_i8_matrix_mad_k32 : no"

echo "=== mixed-precision DPAS (half x int8 etc) ==="
for v in hf_i8_matrix_mad_k16 i8_hf_matrix_mad_k16 hf_u8_matrix_mad_k16 u8_hf_matrix_mad_k16 bf_i8_matrix_mad_k16; do
cat > "$TMP/$v.cl" <<EOF
#pragma OPENCL EXTENSION cl_intel_subgroup_matrix_multiply_accumulate : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(global float* acc, global short* a, global int* b){
    int lane=get_sub_group_local_id();
    short8 A=((global short8*)a)[lane]; int8 B=((global int8*)b)[lane]; float8 C=0;
    C=intel_sub_group_${v}(A,B,C);
    ((global float8*)acc)[lane]=C;
}
EOF
  ocloc compile -file "$TMP/$v.cl" -device $DEV -out_dir "$TMP" >/dev/null 2>&1 \
    && echo "  $v : AVAILABLE" || echo "  $v : no"
done

echo "=== 8b 2D block-read transform variants ==="
for rc in 32r16x1c 16r16x1c; do
cat > "$TMP/rd_$rc.cl" <<EOF
#pragma OPENCL EXTENSION cl_intel_subgroup_2d_block_io : enable
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void t(global void* V,int w,int h,int p,global uint* o){
    uint vt[8]; intel_sub_group_2d_block_read_transform_8b_${rc}((global void*)V,w,h,p,(int2)(0,0),(private uint*)&vt[0]);
    o[get_sub_group_local_id()]=vt[0];
}
EOF
  ocloc compile -file "$TMP/rd_$rc.cl" -device $DEV -out_dir "$TMP" >/dev/null 2>&1 \
    && echo "  transform_8b_$rc : AVAILABLE" || echo "  transform_8b_$rc : no"
done
