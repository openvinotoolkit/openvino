// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// fc_gguf_act_dq.cl — per-token INT8 dynamic quantisation of FP16 activation for
// the GGUF prefill W4A8 DP4A path.
//
// Converts FP16 activation [M, K_SIZE] -> INT8/UINT8 [M, K_SIZE] + scale/ZP.
// GROUP_SIZE == K_SIZE is the MODE_PER_TOKEN case.  Otherwise each work-group
// handles one [token, group] block, matching DynamicQuantize's group contract.
//
// Dispatch: global = [ACT_DQ_WG_SIZE * NUM_GROUPS, M, 1],
// local = [ACT_DQ_WG_SIZE, 1, 1].
//
// JIT constants supplied by FCGGUFActDQGenerator:
//   K_SIZE         — number of FP16 input elements per row (= hidden dim K)
//   ACT_DQ_WG_SIZE — work-group size; must be a power-of-two and <= K_SIZE.

#ifndef ACT_DQ_WG_SIZE
#define ACT_DQ_WG_SIZE 256
#endif

#ifndef ACT_DQ_OUTPUT_VAL_MAX
#define ACT_DQ_OUTPUT_VAL_MAX 127.0f
#endif

#ifndef ACT_DQ_MIN_VAL
#define ACT_DQ_MIN_VAL 0.003h
#endif

#ifndef ACT_DQ_GROUP_SIZE
#define ACT_DQ_GROUP_SIZE K_SIZE
#endif

#ifndef ACT_DQ_NUM_GROUPS
#define ACT_DQ_NUM_GROUPS 1
#endif

#ifndef ACT_DQ_ASYMMETRIC
#define ACT_DQ_ASYMMETRIC 0
#endif

__attribute__((reqd_work_group_size(ACT_DQ_WG_SIZE, 1, 1)))
KERNEL(fc_gguf_act_dq)(
    const __global half* __restrict__ ACT,   // FP16 activation  [M, K_SIZE]
          __global char* __restrict__ ACTQ,  // INT8 output       [M, K_SIZE]
          __global half* __restrict__ ACTSC  // FP16 scale output [M, NUM_GROUPS]
    #if ACT_DQ_ASYMMETRIC
        , __global uchar* __restrict__ ACTSZP // UINT8 zero point [M, NUM_GROUPS]
    #endif
)
{
    const int group = (int)get_group_id(0);
    const int m   = (int)get_global_id(1);   // token (row) index
    const int lid = (int)get_local_id(0);    // lane within the work-group

    const uint k_begin = (uint)group * (uint)ACT_DQ_GROUP_SIZE;
    const uint k_end = min(k_begin + (uint)ACT_DQ_GROUP_SIZE, (uint)K_SIZE);
    const __global half* row = ACT + (uint)m * (uint)K_SIZE;
    // SLM for the parallel max-abs reduction: one half per lane.
    __local half _slm_max[ACT_DQ_WG_SIZE];

    // ---- Phase 1: reduce the group's dynamic range ----
    // Symmetric  : lmax = max(|v|), lower-bounded by ACT_DQ_MIN_VAL.
    // Asymmetric : signed [smin, smax]; both seed at 0 so the span ALWAYS brackets
    //              0, exactly matching dynamic_quantize_gpu_opt's asymmetric contract
    //              (grp_max = fmax(MIN_VAL, max(0, v)), grp_min = fmin(MIN_VAL, min(0, v))).
    // Each lane scans its strided slice [lid, lid+WG, lid+2*WG, ...].
    half lmax = ACT_DQ_MIN_VAL;  // same lower bound as dynamic_quantize_gpu_opt
#if ACT_DQ_ASYMMETRIC
    half smax = 0.0h;            // signed running max (seed 0 -> range includes 0)
    half smin = 0.0h;            // signed running min (seed 0 -> range includes 0)
#endif

#if (ACT_DQ_GROUP_SIZE % (ACT_DQ_WG_SIZE * 8) == 0)
    // Fully vectorised path: the quantization group is an exact multiple of WG*8 elements.
    // Load 8 halfs at a time per lane and fold them into the running reduction.
    __attribute__((opencl_unroll_hint(2)))
    for (int k = (int)k_begin + lid * 8; k < (int)k_end; k += ACT_DQ_WG_SIZE * 8) {
        const half8 v = vload8(0, row + k);
    #if ACT_DQ_ASYMMETRIC
        // Signed min/max — NOT abs: asymmetric quant maps [min, max] -> [0, 255].
        smax = fmax(smax, fmax(fmax(v.s0, v.s1), fmax(v.s2, v.s3)));
        smax = fmax(smax, fmax(fmax(v.s4, v.s5), fmax(v.s6, v.s7)));
        smin = fmin(smin, fmin(fmin(v.s0, v.s1), fmin(v.s2, v.s3)));
        smin = fmin(smin, fmin(fmin(v.s4, v.s5), fmin(v.s6, v.s7)));
    #else
        half lv = fmax(fabs(v.s0), fabs(v.s1));
        lv = fmax(lv, fmax(fabs(v.s2), fabs(v.s3)));
        lv = fmax(lv, fmax(fabs(v.s4), fabs(v.s5)));
        lv = fmax(lv, fmax(fabs(v.s6), fabs(v.s7)));
        lmax = fmax(lmax, lv);
    #endif
    }
#else
    // General fallback: scalar scan (handles any K_SIZE, any WG_SIZE).
    __attribute__((opencl_unroll_hint(4)))
    for (int k = (int)k_begin + lid; k < (int)k_end; k += ACT_DQ_WG_SIZE) {
#if ACT_DQ_ASYMMETRIC
        smax = fmax(smax, row[k]);
        smin = fmin(smin, row[k]);
#else
        lmax = fmax(lmax, fabs(row[k]));
#endif
    }
#endif

#if ACT_DQ_ASYMMETRIC
    __local half _slm_min[ACT_DQ_WG_SIZE];
    _slm_max[lid] = smax;
    _slm_min[lid] = smin;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary tree reduction in SLM (signed min AND max): log2(ACT_DQ_WG_SIZE) steps.
    for (int s = ACT_DQ_WG_SIZE >> 1; s > 0; s >>= 1) {
        if (lid < s) {
            _slm_max[lid] = fmax(_slm_max[lid], _slm_max[lid + s]);
            _slm_min[lid] = fmin(_slm_min[lid], _slm_min[lid + s]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Clamp so the span brackets 0 and is non-degenerate (max_val >= MIN_VAL > 0 >= min_val),
    // matching dynamic_quantize_gpu_opt. This guarantees max_val - min_val > 0.
    const half max_val = fmax(_slm_max[0], (half)ACT_DQ_MIN_VAL);
    const half min_val = fmin(_slm_min[0], 0.0h);
    const half quan_scale = (half)(255.0f / ((float)max_val - (float)min_val));
    const half output_scale = (half)(1.0f / (float)quan_scale);
    const uchar output_zp = convert_uchar_sat_rte(-(float)min_val * (float)quan_scale);
#else
    _slm_max[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Binary tree reduction in SLM: log2(ACT_DQ_WG_SIZE) steps.
    __attribute__((opencl_unroll_hint))
    for (int s = ACT_DQ_WG_SIZE >> 1; s > 0; s >>= 1) {
        if (lid < s)
            _slm_max[lid] = fmax(_slm_max[lid], _slm_max[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const half max_val = _slm_max[0];
    const half quan_scale = (half)(ACT_DQ_OUTPUT_VAL_MAX / (float)max_val);
    const half output_scale = (half)(1.0f / (float)quan_scale);
#endif

    // ---- Phase 2: quantise (symmetric INT8, or asymmetric UINT8 with per-group ZP) ----
    __global char* out_row = ACTQ + (uint)m * (uint)K_SIZE;

#if (ACT_DQ_GROUP_SIZE % (ACT_DQ_WG_SIZE * 8) == 0)
    // Vectorised: quantise 8 elements per lane per iteration.
    __attribute__((opencl_unroll_hint(2)))
    for (int k = (int)k_begin + lid * 8; k < (int)k_end; k += ACT_DQ_WG_SIZE * 8) {
        const half8 v = vload8(0, row + k);
        // round-to-nearest, saturate to [-128, 127]
    #if ACT_DQ_ASYMMETRIC
        uchar8 q;
    #else
        char8 q;
    #endif
    #if ACT_DQ_ASYMMETRIC
        q.s0 = convert_uchar_sat_rte((float)v.s0 * (float)quan_scale + (float)output_zp);
        q.s1 = convert_uchar_sat_rte((float)v.s1 * (float)quan_scale + (float)output_zp);
        q.s2 = convert_uchar_sat_rte((float)v.s2 * (float)quan_scale + (float)output_zp);
        q.s3 = convert_uchar_sat_rte((float)v.s3 * (float)quan_scale + (float)output_zp);
        q.s4 = convert_uchar_sat_rte((float)v.s4 * (float)quan_scale + (float)output_zp);
        q.s5 = convert_uchar_sat_rte((float)v.s5 * (float)quan_scale + (float)output_zp);
        q.s6 = convert_uchar_sat_rte((float)v.s6 * (float)quan_scale + (float)output_zp);
        q.s7 = convert_uchar_sat_rte((float)v.s7 * (float)quan_scale + (float)output_zp);
    #else
        q.s0 = convert_char_sat_rte((float)v.s0 * (float)quan_scale);
        q.s1 = convert_char_sat_rte((float)v.s1 * (float)quan_scale);
        q.s2 = convert_char_sat_rte((float)v.s2 * (float)quan_scale);
        q.s3 = convert_char_sat_rte((float)v.s3 * (float)quan_scale);
        q.s4 = convert_char_sat_rte((float)v.s4 * (float)quan_scale);
        q.s5 = convert_char_sat_rte((float)v.s5 * (float)quan_scale);
        q.s6 = convert_char_sat_rte((float)v.s6 * (float)quan_scale);
        q.s7 = convert_char_sat_rte((float)v.s7 * (float)quan_scale);
    #endif
    #if ACT_DQ_ASYMMETRIC
        vstore8(q, 0, (__global uchar*)out_row + k);  // out_row is char*; asymmetric writes uchar8
    #else
        vstore8(q, 0, out_row + k);
    #endif
    }
#else
    __attribute__((opencl_unroll_hint(4)))
    for (int k = (int)k_begin + lid; k < (int)k_end; k += ACT_DQ_WG_SIZE) {
#if ACT_DQ_ASYMMETRIC
        out_row[k] = (char)convert_uchar_sat_rte((float)row[k] * (float)quan_scale + (float)output_zp);
#else
        out_row[k] = convert_char_sat_rte((float)row[k] * (float)quan_scale);
#endif
    }
#endif

    // One lane writes the per-token scale.
    if (lid == 0) {
        ACTSC[(uint)m * ACT_DQ_NUM_GROUPS + (uint)group] = output_scale;
#if ACT_DQ_ASYMMETRIC
        ACTSZP[(uint)m * ACT_DQ_NUM_GROUPS + (uint)group] = output_zp;
#endif
    }
}
