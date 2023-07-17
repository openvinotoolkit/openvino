// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL (count_nonzero_ref)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* input,
    volatile __global OUTPUT_TYPE* output)
{
    const uint local_idx = get_local_id(0);
    const uint num_work_items = get_global_size(0);
    const uint data_size = DATA_SIZE;
    const uint items_num = data_size / num_work_items;
    const uint leftovers = data_size - (items_num * num_work_items);

    if (local_idx == 0)
        output[0] = 0;
    barrier(CLK_GLOBAL_MEM_FENCE);

    uint workitem_nonzero_count = 0;
    uint actual_items_num = items_num;
    uint input_start_idx  = (actual_items_num * local_idx) + leftovers;
    if (local_idx < leftovers) {
        actual_items_num = items_num + 1;
        input_start_idx  = actual_items_num * local_idx;
    }

    for (uint iter = 0; iter < actual_items_num; iter++) {
        const uint idx = input_start_idx + iter;
        uint count = (input[idx] == INPUT0_VAL_ZERO) ? 0 : 1;
        workitem_nonzero_count += count;
    }

    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    uint subgroup_nonzero_count = sub_group_reduce_add(workitem_nonzero_count);

    barrier(CLK_GLOBAL_MEM_FENCE);
    if (get_sub_group_local_id() == 0)
        atomic_add(&(output[0]), subgroup_nonzero_count);
}
