// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

#if IS_DYNAMIC
#define CALC_POWER(n) ({uint pos = 0; uint i = n; do { i >>= 1; ++pos; } while (i); --pos;})
#endif

#if !IS_DYNAMIC
__attribute__((reqd_work_group_size(LWS, 1, 1)))
#endif
KERNEL (softmax_gpu_continuous_bfyx)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     // in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          // how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  // this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       // how many elements are in one data set
    const uint data_sets_count = DATA_SETS_COUNT;   // how many data sets are in the processing payload
#if !IS_DYNAMIC
    const uint items_num = ITEMS_NUM;               // how many elements are processed per one WI
    const uint leftovers = LEFTOVERS;
#else
    // since workers_per_data_set is calculated by power of 2
    // items_num can be calculated by dividing data_set_size by power of 2
    const uint power = CALC_POWER(workers_per_data_set);
    const uint items_num = data_set_size>>power;
    const uint leftovers = data_set_size-(items_num<<power);
#endif

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    INPUT0_TYPE my_chunk[STACK_SIZE];
    INPUT0_TYPE my_maximum = -UNIT_VAL_MAX;
    INPUT0_TYPE my_sum = UNIT_VAL_ZERO;
    INPUT0_TYPE tmp;

    __local INPUT0_TYPE lg_storage[SLM_SIZE];

    //each WI reads items_num consecutive items from batch
    for (uint i=0; i<items_num; ++i)
    {
        tmp = input[my_data_offset + i * workers_per_data_set];
        my_maximum = max(my_maximum, tmp);
        my_chunk[i] = tmp;
    }

    if (in_data_set_idx < leftovers)
    {
        tmp = input[data_set_offset + workers_per_data_set * items_num + in_data_set_idx];
        my_maximum = max(my_maximum, tmp);
        my_chunk[items_num] = tmp;
    }

    lg_storage[in_data_set_idx] = my_maximum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_maximum = max(my_maximum, lg_storage[i]);

        lg_storage[0] = my_maximum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    //my_maximum from this point is in fact global maximum
    my_maximum = lg_storage[0];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint i=0; i<items_num; ++i)
    {
        tmp = native_exp(my_chunk[i] - my_maximum);
        my_sum += tmp;
        my_chunk[i] = tmp;
    }

    if (in_data_set_idx < leftovers)
    {
        tmp = native_exp(my_chunk[items_num] - my_maximum);
        my_sum += tmp;
        my_chunk[items_num] = tmp;
    }

    lg_storage[in_data_set_idx] = my_sum;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (in_data_set_idx == 0)
    {
        for (uint i=1; i<LWS; ++i)
            my_sum += lg_storage[i];

        lg_storage[0] = my_sum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    my_sum = lg_storage[0];

#if HAS_FUSED_OPS
    for (uint i=0; i<items_num; ++i)
    {
        ACTIVATION_TYPE dequantized = my_chunk[i] / my_sum;
        FUSED_OPS_MAIN;
        output[my_data_offset + i * workers_per_data_set] = FUSED_OPS_RESULT_MAIN;
    }
    if (in_data_set_idx < leftovers)
    {
        ACTIVATION_TYPE dequantized = my_chunk[items_num] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[data_set_offset + workers_per_data_set * items_num + in_data_set_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }
#else
    for (uint i=0; i<items_num; ++i)
        output[my_data_offset + i * workers_per_data_set] = ACTIVATION(my_chunk[i] / my_sum, ACTIVATION_PARAMS);
    if (in_data_set_idx < leftovers)
        output[data_set_offset + workers_per_data_set * items_num + in_data_set_idx] = ACTIVATION(my_chunk[items_num] / my_sum, ACTIVATION_PARAMS);
#endif
}
#ifdef CALC_POWER
#undef CALC_POWER
#endif
