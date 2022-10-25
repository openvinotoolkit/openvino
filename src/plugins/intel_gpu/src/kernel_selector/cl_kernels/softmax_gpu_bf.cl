// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/data_types.cl"


__attribute__((reqd_work_group_size(LWS, 1, 1)))
KERNEL (softmax_gpu_continuous_bfyx)(
    const __global INPUT0_TYPE* input,
    __global OUTPUT_TYPE* output
#if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#endif
) {
    const uint data_set_idx = get_global_id(1);     //in processing of which data set this WI participates?
    const uint workers_per_data_set = LWS;          //how many WI participates in processing of one data set
    const uint in_data_set_idx = get_global_id(0);  //this WI's id in group of items processing single data set
    const uint data_set_size = DATA_SET_SIZE;       //how many elements are in one data set
    const uint data_sets_count = DATA_SETS_COUNT;   //how many data sets are in the processing payload

    const uint data_set_offset = data_set_idx * data_set_size;
    const uint my_data_offset = data_set_offset + in_data_set_idx;

    INPUT0_TYPE my_chunk[ITEMS_NUM + 1];
    INPUT0_TYPE my_maximum = -UNIT_VAL_MAX;
    INPUT0_TYPE my_sum = UNIT_VAL_ZERO;
    INPUT0_TYPE tmp;

    __local INPUT0_TYPE lg_storage[LWS];

    //each WI reads ITEMS_NUM consecutive items from batch
    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        tmp = input[my_data_offset + i * workers_per_data_set];
        my_maximum = max(my_maximum, tmp);
        my_chunk[i] = tmp;
    }

    if (in_data_set_idx < LEFTOVERS)
    {
        tmp = input[data_set_offset + workers_per_data_set * ITEMS_NUM + in_data_set_idx];
        my_maximum = max(my_maximum, tmp);
        my_chunk[ITEMS_NUM] = tmp;
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

    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        tmp = native_exp(my_chunk[i] - my_maximum);
        my_sum += tmp;
        my_chunk[i] = tmp;
    }

    if (in_data_set_idx < LEFTOVERS)
    {
        tmp = native_exp(my_chunk[ITEMS_NUM] - my_maximum);
        my_sum += tmp;
        my_chunk[ITEMS_NUM] = tmp;
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
    for (uint i=0; i<ITEMS_NUM; ++i)
    {
        ACTIVATION_TYPE dequantized = my_chunk[i] / my_sum;
        FUSED_OPS_MAIN;
        output[my_data_offset + i * workers_per_data_set] = FUSED_OPS_RESULT_MAIN;
    }
    if (in_data_set_idx < LEFTOVERS)
    {
        ACTIVATION_TYPE dequantized = my_chunk[ITEMS_NUM] / my_sum;
        FUSED_OPS_LEFTOVERS;
        output[data_set_offset + workers_per_data_set * ITEMS_NUM + in_data_set_idx] = FUSED_OPS_RESULT_LEFTOVERS;
    }
#else
    for (uint i=0; i<ITEMS_NUM; ++i)
        output[my_data_offset + i * workers_per_data_set] = ACTIVATION(my_chunk[i] / my_sum, ACTIVATION_PARAMS);
    if (in_data_set_idx < LEFTOVERS)
        output[data_set_offset + workers_per_data_set * ITEMS_NUM + in_data_set_idx] = ACTIVATION(my_chunk[ITEMS_NUM] / my_sum, ACTIVATION_PARAMS);
#endif
}
