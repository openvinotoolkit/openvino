// Copyright (C) 2018-2025 Intel Corporation
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

    float my_sum = 0;
    float tmp;

    //each WI reads items_num consecutive items from batch*feature
    for (uint i=0; i<iters_num; ++i)
    {
        my_sum += (float)input[my_data_offset + i * workers_per_data_set];
    }

    my_sum = work_group_reduce_add(my_sum) / data_set_size;

#if NORMALIZE_VARIANCE == 0
    for (uint i=0; i<iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#else

    float my_variance = 0.f;
    //each WI reads items_num consecutive items from batch*feature
    for (uint i=0; i<iters_num; ++i)
    {
        tmp = (float)input[my_data_offset + i * workers_per_data_set];
        tmp -= my_sum;
        my_variance = fma(tmp, tmp, my_variance);
    }

    my_variance = work_group_reduce_add(my_variance);

    if (in_data_set_idx == 0)
    {
        my_variance /= data_set_size;

#   if defined EPS_OUTSIDE_SQRT
        my_variance = native_powr(native_sqrt(my_variance) + (float)EPSILON, -1.f);
#   elif defined EPS_INSIDE_SQRT
        my_variance = native_powr(my_variance + (float)EPSILON, -0.5f);
#   endif
    }

    my_variance = work_group_broadcast(my_variance, 0);

    for (uint i=0; i<iters_num; ++i) {
        uint iteration_in_data_set_offset = i * workers_per_data_set;
        ACTIVATION_TYPE result = (TO_ACTIVATION_TYPE(input[my_data_offset + iteration_in_data_set_offset]) - TO_ACTIVATION_TYPE(my_sum)) * TO_ACTIVATION_TYPE(my_variance);
#   if HAS_FUSED_OPS
        FUSED_OPS;
        output[my_data_offset + iteration_in_data_set_offset] = FUSED_OPS_RESULT;
#   else
        output[my_data_offset + iteration_in_data_set_offset] = TO_OUTPUT_TYPE(ACTIVATION(result, ACTIVATION_PARAMS));
#   endif
    }
#endif
}
