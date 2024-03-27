// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(pa_kv_cache_update)(
    OPTIONAL_SHAPE_INFO_ARG
    __global const INPUT0_TYPE* key_data,
    __global const INPUT1_TYPE* value_data,
    __global const INPUT2_TYPE* slot_mapping,
    __global OUTPUT_TYPE* key_cache_data,
    __global OUTPUT1_TYPE* value_cache_data)
{
    const uint batch_idx = (uint)get_global_id(0);
    const uint seq_idx = (uint)get_global_id(1);
    const uint hidden_idx = (uint)get_global_id(2);

    const uint in_offset = batch_idx * INPUT0_BATCH_PITCH + seq_idx * INPUT0_FEATURE_PITCH + hidden_idx;
    const uint slot_offset = batch_idx * INPUT0_FEATURE_NUM + seq_idx;

    const INPUT2_TYPE slot_idx = slot_mapping[slot_offset];
    if (hidden_idx >= INPUT0_FEATURE_PITCH || slot_idx == -1)
        return;

    const uint block_index = slot_idx / KV_CACHE_BLOCK_SIZE;
    const uint block_offset = slot_idx % KV_CACHE_BLOCK_SIZE;

#ifdef VALUE_CACHE_UPDATE
    const uint out_offset = CACHE_BLOCK_STRIDE * block_index +
                            hidden_idx * KV_CACHE_BLOCK_SIZE +
                            block_offset;

    value_cache_data[out_offset] = value_data[in_offset];
#else
    const uint head_size_outer_block = hidden_idx / X_BLOCK_SIZE;
    const uint head_size_inner_block = hidden_idx % X_BLOCK_SIZE;

    const uint out_offset = CACHE_BLOCK_STRIDE * block_index +
                            block_offset * X_BLOCK_SIZE +
                            head_size_outer_block * KV_CACHE_BLOCK_SIZE * X_BLOCK_SIZE +
                            head_size_inner_block;

    key_cache_data[out_offset] = key_data[in_offset];
#endif
}
