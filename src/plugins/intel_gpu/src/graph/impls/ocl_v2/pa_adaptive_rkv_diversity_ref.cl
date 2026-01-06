// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(SUBGROUP_SIZE, 1, 1)))
KERNEL(pa_adaptive_rkv_diversity)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_cache,              // [num_blocks, KV_HEADS_NUM, HEAD_SIZE, BLOCK_SIZE]
    __global const INPUT1_TYPE* start_sizes,            // [batch_size]
    __global const INPUT2_TYPE* evictable_sizes,        // [batch_size]
    __global const INPUT3_TYPE* block_indices,          // [total_evictable_blocks]
    __global const INPUT4_TYPE* block_indices_begins,   // [batch_size + 1]
    __global OUTPUT_TYPE* diversity_output              // [total_diversity_scores]
) {
    const uint batch_idx = get_group_id(0);
    const uint sglid = get_sub_group_local_id();

    const int start_size = start_sizes[batch_idx];
    const int evictable_size = evictable_sizes[batch_idx];
    const int num_evictable_blocks = evictable_size / PAGED_ATTENTION_BLOCK_SIZE;

    if (sglid == 0 && batch_idx == 0) {
        printf("[DEBUG] batch_idx=%d, start_size=%d, evictable_size=%d, num_evictable_blocks=%d\n",
               batch_idx, start_size, evictable_size, num_evictable_blocks);
    }

    if (num_evictable_blocks == 0)
        return;

    // TODO: Implement kernel logic here
}
