// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "include/batch_headers/fetch_data.cl"

// MVN optimization vs `mvn_gpu_bfyx_opt_twopass.cl`:
// - The twopass kernel always rereads input in variance/output loops. This kernel adds a
//   register-cache fast path (`MVN_REREAD_INPUT=0`) so each element is fetched from global
//   memory once, then reused for centered-variance and output passes.
// - For large shapes, the selector can set `MVN_REREAD_INPUT=1` to match twopass-like
//   reread behavior and reduce register pressure.
// - The host selector chooses LWS as the largest power of two that keeps at least 8
//   normalized elements per work-item. On PTL this selects measured roofline values for
//   Qwen3-Omni C6 (`D=1152 -> LWS=128`, `D=4608 -> LWS=512`) without hardcoding shapes.
// - Variance is computed from centered values `(x - mean)^2` (stable under cancellation),
//   while preserving the same fused-ops/output flow as the twopass kernel.
// - `MVN_STACK_SIZE` is shape-dependent and provided by JIT so the selector can balance
//   cache reuse (bandwidth) against register pressure.
#ifndef MVN_STACK_SIZE
#define MVN_STACK_SIZE 16
#endif

#ifndef MVN_REREAD_INPUT
#define MVN_REREAD_INPUT 0
#endif

#ifndef LWS_IS_STATIC
#define LWS_IS_STATIC 0
#endif

#if LWS_IS_STATIC
__attribute__((reqd_work_group_size(LWS, 1, 1)))
#endif
KERNEL (mvn_gpu_bfyx_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* restrict output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     // in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          // how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  // this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       // how many elements are in one data set
    // DATA_SETS_COUNT is still emitted by the host selector for the shared MVN JIT ABI.
    const uint items_num = data_set_size / workers_per_data_set;
    const uint leftovers = data_set_size % workers_per_data_set;

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;
    uint iters_num = items_num;
    if (in_data_set_idx < leftovers)
        ++iters_num;

#if !MVN_REREAD_INPUT
    float data[MVN_STACK_SIZE];
#endif
    float my_sum = 0.f;
    for (uint i = 0; i < iters_num; ++i) {
        float v = (float)input[my_data_offset + i * workers_per_data_set];
#if !MVN_REREAD_INPUT
        data[i] = v;
#endif
        my_sum += v;
    }

    float red_sum = work_group_reduce_add(my_sum);
    float my_sum_mean = red_sum / data_set_size;

#if NORMALIZE_VARIANCE == 0
    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
#if MVN_REREAD_INPUT
        float v = (float)input[my_data_offset + iteration_in_data_set_offset];
#else
        float v = data[i];
#endif
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(v) - TO_ACTIVATION_TYPE(my_sum_mean);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#else
    // Numerically stable variance: accumulate squared centered values after mean reduction.
    // With MVN_REREAD_INPUT=0 this remains single-read from global memory (uses cached data[]).
    // With MVN_REREAD_INPUT=1 it falls back to rereading input for the centered pass.
    float my_var_acc = 0.f;
    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
#if MVN_REREAD_INPUT
        float v = (float)input[my_data_offset + iteration_in_data_set_offset];
#else
        float v = data[i];
#endif
        float d = v - my_sum_mean;
        my_var_acc = fma(d, d, my_var_acc);
    }

    float red_var_acc = work_group_reduce_add(my_var_acc);
    float my_variance = fmax(red_var_acc / data_set_size, 0.f);
#   if defined EPS_OUTSIDE_SQRT
    float my_inv = native_powr(native_sqrt(my_variance) + (float)EPSILON, -1.f);
#   elif defined EPS_INSIDE_SQRT
    float my_inv = native_rsqrt(my_variance + (float)EPSILON);
#   else
    float my_inv = native_rsqrt(my_variance);
#   endif

    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
#if MVN_REREAD_INPUT
        float v = (float)input[my_data_offset + iteration_in_data_set_offset];
#else
        float v = data[i];
#endif
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(v) - TO_ACTIVATION_TYPE(my_sum_mean)) * TO_ACTIVATION_TYPE(my_inv);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#endif
}
