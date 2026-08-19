// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// fc_gguf_act_dq.cl — per-token INT8 dynamic quantisation of FP16 activation for
// the GGUF prefill W4A8 DP4A path.
//
// Converts FP16 activation [M, K_SIZE] -> INT8 [M, K_SIZE] + FP16 per-token
// scale [M, 1].  The scale is defined as  max_abs / 127  so that the original
// value is recovered as  scale * int8_val  (symmetric, signed INT8).
//
// Dispatch: global = [M * ACT_DQ_WG_SIZE, 1, 1],  local = [ACT_DQ_WG_SIZE, 1, 1]
// Each work-group processes exactly one activation row (one token).
//
// JIT constants supplied by FCGGUFActDQGenerator:
//   K_SIZE         — number of FP16 input elements per row (= hidden dim K)
//   ACT_DQ_WG_SIZE — work-group size; must be a power-of-two and <= K_SIZE.

#ifndef ACT_DQ_WG_SIZE
#define ACT_DQ_WG_SIZE 256
#endif

__attribute__((reqd_work_group_size(ACT_DQ_WG_SIZE, 1, 1)))
KERNEL(fc_gguf_act_dq)(
    const __global half* __restrict__ ACT,   // FP16 activation  [M, K_SIZE]
          __global char* __restrict__ ACTQ,  // INT8 output       [M, K_SIZE]
          __global half* __restrict__ ACTSC  // FP16 scale output [M, 1]
)
{
    const int m   = (int)get_group_id(0);    // token (row) index
    const int lid = (int)get_local_id(0);    // lane within the work-group

    const __global half* row = ACT + (uint)m * (uint)K_SIZE;
    // SLM for the parallel max-abs reduction: one half per lane.
    __local half _slm_max[ACT_DQ_WG_SIZE];

    // ---- Phase 1: find the max absolute value across this row ----
    // Each lane scans its strided slice [lid, lid+WG, lid+2*WG, ...].
    half lmax = 0.003h;  // lower bound to guard against all-zero rows (avoids inf scale)

#if (K_SIZE % (ACT_DQ_WG_SIZE * 8) == 0)
    // Fully vectorised path: K_SIZE is an exact multiple of WG*8 elements.
    // Load 8 halfs at a time per lane, extract per-element abs, keep running max.
    __attribute__((opencl_unroll_hint(2)))
    for (int k = lid * 8; k < K_SIZE; k += ACT_DQ_WG_SIZE * 8) {
        const half8 v = vload8(0, row + k);
        half lv = fmax(fabs(v.s0), fabs(v.s1));
        lv = fmax(lv, fmax(fabs(v.s2), fabs(v.s3)));
        lv = fmax(lv, fmax(fabs(v.s4), fabs(v.s5)));
        lv = fmax(lv, fmax(fabs(v.s6), fabs(v.s7)));
        lmax = fmax(lmax, lv);
    }
#else
    // General fallback: scalar scan (handles any K_SIZE, any WG_SIZE).
    __attribute__((opencl_unroll_hint(4)))
    for (int k = lid; k < K_SIZE; k += ACT_DQ_WG_SIZE) {
        lmax = fmax(lmax, fabs(row[k]));
    }
#endif

    _slm_max[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary tree reduction in SLM: log2(ACT_DQ_WG_SIZE) steps.
    __attribute__((opencl_unroll_hint))
    for (int s = ACT_DQ_WG_SIZE >> 1; s > 0; s >>= 1) {
        if (lid < s)
            _slm_max[lid] = fmax(_slm_max[lid], _slm_max[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const half max_val   = _slm_max[0];
    const half inv_scale = (half)(127.0f / (float)max_val);
    const half scale     = (half)((float)max_val / 127.0f);

    // ---- Phase 2: symmetric INT8 quantisation ----
    __global char* out_row = ACTQ + (uint)m * (uint)K_SIZE;

#if (K_SIZE % (ACT_DQ_WG_SIZE * 8) == 0)
    // Vectorised: quantise 8 elements per lane per iteration.
    __attribute__((opencl_unroll_hint(2)))
    for (int k = lid * 8; k < K_SIZE; k += ACT_DQ_WG_SIZE * 8) {
        const half8 v = vload8(0, row + k);
        // round-to-nearest, saturate to [-128, 127]
        char8 q;
        q.s0 = (char)clamp((int)round((float)v.s0 * (float)inv_scale), -128, 127);
        q.s1 = (char)clamp((int)round((float)v.s1 * (float)inv_scale), -128, 127);
        q.s2 = (char)clamp((int)round((float)v.s2 * (float)inv_scale), -128, 127);
        q.s3 = (char)clamp((int)round((float)v.s3 * (float)inv_scale), -128, 127);
        q.s4 = (char)clamp((int)round((float)v.s4 * (float)inv_scale), -128, 127);
        q.s5 = (char)clamp((int)round((float)v.s5 * (float)inv_scale), -128, 127);
        q.s6 = (char)clamp((int)round((float)v.s6 * (float)inv_scale), -128, 127);
        q.s7 = (char)clamp((int)round((float)v.s7 * (float)inv_scale), -128, 127);
        vstore8(q, 0, out_row + k);
    }
#else
    __attribute__((opencl_unroll_hint(4)))
    for (int k = lid; k < K_SIZE; k += ACT_DQ_WG_SIZE) {
        const int q = (int)round((float)row[k] * (float)inv_scale);
        out_row[k] = (char)clamp(q, -128, 127);
    }
#endif

    // One lane writes the per-token scale.
    if (lid == 0)
        ACTSC[m] = scale;
}
