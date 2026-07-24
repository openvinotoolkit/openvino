// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

// Generalized MVN optimization:
// - The host selector chooses LWS as the largest power of two that keeps at least 8
//   normalized elements per work-item. On PTL this selects the measured roofline LWS
//   values for Qwen3-Omni C6 (`D=1152 -> LWS=128`, `D=4608 -> LWS=512`) without
//   hardcoding those shapes.
// - Each work-item handles `ceil(DATA_SET_SIZE / LWS)` values. The fast path caches
//   those values in registers so input is read once, while mean and variance are both
//   accumulated in the same pass (`sum` and `sum_of_squares`).
// - `MVN_STACK_SIZE` is therefore shape-dependent and must be provided by JIT. The
//   selector keeps the register-cache path only while the stack is small enough to avoid
//   excessive register pressure; larger stacks define `MVN_REREAD_INPUT=1`, which skips
//   the register array and rereads input for the output pass.
#ifndef MVN_STACK_SIZE
#define MVN_STACK_SIZE 16
#endif

#ifndef MVN_REREAD_INPUT
#define MVN_REREAD_INPUT 0
#endif

#if !IS_DYNAMIC
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
    float my_sq = 0.f;
    for (uint i = 0; i < iters_num; ++i) {
        float v = (float)input[my_data_offset + i * workers_per_data_set];
#if !MVN_REREAD_INPUT
        data[i] = v;
#endif
        my_sum += v;
        my_sq = fma(v, v, my_sq);
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
    float red_sq = work_group_reduce_add(my_sq);
    float my_variance = red_sq / data_set_size - my_sum_mean * my_sum_mean;
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
