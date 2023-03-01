// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"

#define BLOCK_SIZE ELEMENTS_PER_THREAD

#if BLOCK_SIZE == 1
#define BLOCK_READ(ptr, offset) DT_INPUT_BLOCK_READ(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) DT_OUTPUT_BLOCK_WRITE(ptr, offset, val)
#define BLOCK_TYPE INPUT0_TYPE
#else
#define BLOCK_READ(ptr, offset) CAT(DT_INPUT_BLOCK_READ, BLOCK_SIZE)(ptr, offset)
#define BLOCK_WRITE(ptr, offset, val) CAT(DT_OUTPUT_BLOCK_WRITE, BLOCK_SIZE)(ptr, offset, val)
#define BLOCK_TYPE MAKE_VECTOR_TYPE(INPUT0_TYPE, BLOCK_SIZE)
#endif

#define SUBGROUP_SIZE 16
#define PARTIAL_REDUCE_SIZE (DATA_SET_SIZE / ELEMENTS_PER_THREAD / SUBGROUP_SIZE)

#if MAX_REDUCE_KERNEL

KERNEL(max_reduce)(const __global INPUT0_TYPE* input,
                   __global INPUT0_TYPE* partial_max)
{
    const uint dataset_id = get_global_id(0);
    const uint within_dataset_id = get_global_id(1);
    
    const uint dataset_offset = dataset_id * DATA_SET_SIZE;
    const uint lane = get_sub_group_local_id();
    const uint input_idx = dataset_offset + (within_dataset_id - lane) * ELEMENTS_PER_THREAD;
    
    BLOCK_TYPE vec = BLOCK_READ(input, input_idx);
    INPUT0_TYPE max_value = UNIT_VAL_MIN;
    for (int i = 0; i < BLOCK_SIZE; i++)
        max_value = max(vec[i], max_value);
    max_value = sub_group_reduce_max(max_value);
    if (lane == 0)
        partial_max[dataset_id * PARTIAL_REDUCE_SIZE + (within_dataset_id / SUBGROUP_SIZE)] = max_value;
}

#elif ADD_REDUCE_KERNEL

KERNEL(sum_reduce)(const __global INPUT0_TYPE* input,
                   const __global INPUT0_TYPE* partial_max,
                   __global INPUT0_TYPE* partial_sum,
                   __global INPUT0_TYPE* max_values)
{
    const uint dataset_id = get_global_id(0);
    const uint within_dataset_id = get_global_id(1);
    
    const uint dataset_offset = dataset_id * DATA_SET_SIZE;
    const uint lane = get_sub_group_local_id();
    const uint input_idx = dataset_offset + (within_dataset_id - lane) * ELEMENTS_PER_THREAD;
    
    INPUT0_TYPE max_value = UNIT_VAL_MIN;
    local INPUT0_TYPE final_max;
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        for (int i = 0; i < PARTIAL_REDUCE_SIZE; i++)
            max_value = max(max_value, partial_max[dataset_id * PARTIAL_REDUCE_SIZE + i]);
        final_max = max_value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    max_value = final_max;
    
    if (within_dataset_id == 0)
        max_values[dataset_id] = max_value;
    
    BLOCK_TYPE vec = BLOCK_READ(input, input_idx);
    INPUT0_TYPE sum_value = UNIT_VAL_ZERO;
    for (int i = 0; i < BLOCK_SIZE; i++)
        sum_value += native_exp(vec[i] - max_value);
    sum_value = sub_group_reduce_add(sum_value);
    if (lane == 0)
        partial_sum[dataset_id * PARTIAL_REDUCE_SIZE + (within_dataset_id / SUBGROUP_SIZE)] = sum_value;
}

#elif SOFTMAX_KERNEL

KERNEL(softmax)(
       const __global INPUT0_TYPE* input,
       __global OUTPUT_TYPE* restrict output,
#if HAS_FUSED_OPS_DECLS
       FUSED_OPS_DECLS,
#endif
      __global INPUT0_TYPE* partial_sum,
      __global INPUT0_TYPE* max_values
)
{
    const uint dataset_id = get_global_id(0);
    const uint within_dataset_id = get_global_id(1);
    
    const uint dataset_offset = dataset_id * DATA_SET_SIZE;
    const uint lane = get_sub_group_local_id();
    const uint input_idx = dataset_offset + (within_dataset_id - lane) * ELEMENTS_PER_THREAD;
    local INPUT0_TYPE max_value;
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
        max_value = max_values[dataset_id];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    INPUT0_TYPE sum_value = UNIT_VAL_ZERO;
    local INPUT0_TYPE final_sum;
    if (get_local_id(0) == 0 && get_local_id(1) == 0)
    {
        for (int i = 0; i < PARTIAL_REDUCE_SIZE; i++)
            sum_value += partial_sum[dataset_id * PARTIAL_REDUCE_SIZE + i];
        final_sum = sum_value;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_value = final_sum;
    
    BLOCK_TYPE vec = BLOCK_READ(input, input_idx);
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
#if HAS_FUSED_OPS
        ACTIVATION_TYPE dequantized = vec[i] / sum_value;
        FUSED_OPS_MAIN;
        vec[i] = FUSED_OPS_RESULT_MAIN
#else
        vec[i] = ACTIVATION(native_exp(vec[i] - max_value) / sum_value, ACTIVATION_PARAMS);
#endif
    }
    BLOCK_WRITE(output, input_idx, vec);
}

#endif
