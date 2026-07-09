// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

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
    const uint data_sets_count = DATA_SETS_COUNT;   // how many data sets are in the processing payload
    const uint items_num = data_set_size / workers_per_data_set;
    const uint leftovers = data_set_size % workers_per_data_set;

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;
    uint iters_num = items_num;
    if (in_data_set_idx < leftovers)
        ++iters_num;

#if NORMALIZE_VARIANCE == 0
    float my_sum = 0.0f;
    for (uint i = 0; i < iters_num; ++i)
        my_sum += (float)input[my_data_offset + i * workers_per_data_set];

    my_sum = work_group_reduce_add(my_sum);
    const float mean  = my_sum / (float)data_set_size;

#elif NORMALIZE_VARIANCE == 1
    float local_mean  = 0.0f;
    float local_m2    = 0.0f;
    uint  local_count = 0;

    for (uint i = 0; i < iters_num; ++i)
    {
        const float x      = (float)input[my_data_offset + i * workers_per_data_set];
        local_count++;
        const float delta  = x - local_mean;
        local_mean        += delta / (float)local_count;
        const float delta2 = x - local_mean;
        local_m2          += delta * delta2;
    }

    __local float wg_mean[LWS];
    __local float wg_m2[LWS];
    __local uint  wg_count[LWS];

    wg_mean[in_data_set_idx]  = local_mean;
    wg_m2[in_data_set_idx]    = local_m2;
    wg_count[in_data_set_idx] = local_count;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = workers_per_data_set / 2; stride > 0; stride >>= 1) {
        if (in_data_set_idx < stride) {
            const uint  na     = wg_count[in_data_set_idx];
            const uint  nb     = wg_count[in_data_set_idx + stride];
            const float mean_a = wg_mean[in_data_set_idx];
            const float mean_b = wg_mean[in_data_set_idx + stride];
            const float m2_a   = wg_m2[in_data_set_idx];
            const float m2_b   = wg_m2[in_data_set_idx + stride];
            const uint  n_comb = na + nb;
            const float delta  = mean_b - mean_a;
            wg_mean[in_data_set_idx]  = mean_a + delta * (float)nb / (float)n_comb;
            wg_m2[in_data_set_idx]    = m2_a + m2_b + delta * delta * (float)na * (float)nb / (float)n_comb;
            wg_count[in_data_set_idx] = n_comb;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const float mean = wg_mean[0];
    float variance = wg_m2[0] / (float)data_set_size;
    variance = fmax(variance, 0.0f);


#   if defined EPS_OUTSIDE_SQRT
    float my_variance = native_powr(native_sqrt(variance) + (float)EPSILON, -1.f);
#   elif defined EPS_INSIDE_SQRT
    float my_variance = native_powr(variance + (float)EPSILON, -0.5f);
#   endif
#endif

    for (uint i = 0; i < iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
#if NORMALIZE_VARIANCE == 1        
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(mean)) * TO_ACTIVATION_TYPE(my_variance);
#else
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(mean);
#endif
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
}
