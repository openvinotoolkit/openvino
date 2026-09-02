// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "selective_ssm_type_utils.cl"

#define SSM_MAX_HEAD_DIM_BLOCK 4
#define SSM_STATE_ITERATION_TYPE uint
#define SSM_DT_INDEX(token) GET_DATA_INDEX(INPUT1, token, h, 0, 0)
#define SSM_B_INDEX(token, state_element) GET_DATA_INDEX(INPUT2, token, g, state_element, 0)
#define SSM_C_INDEX(token, state_element) GET_DATA_INDEX(INPUT4, token, g, state_element, 0)
#define SSM_X_INDEX(token, p) GET_DATA_INDEX(INPUT3, token, h, p, 0)
#define SSM_OUTPUT_INDEX(token, p) GET_DATA_INDEX(OUTPUT, token, h, p, 0)
#define SSM_STATE_INDEX(p_offset, state_element) ((size_t)(p_offset) * state_size + (state_element))
#define SSM_STATE_AT(state_index) local_state[state_index]

KERNEL(paged_selective_ssm_opt)(
    OPTIONAL_SHAPE_INFO_ARG
    const __global INPUT0_TYPE* A,
    const __global INPUT1_TYPE* dt,
    const __global INPUT2_TYPE* B,
    const __global INPUT3_TYPE* x,
    const __global INPUT4_TYPE* C,
    __global INPUT5_TYPE* recurrent_state_table,
    const __global INPUT6_TYPE* subsequence_begins,
    const __global INPUT7_TYPE* block_indices,
    const __global INPUT8_TYPE* block_indices_begins,
    const __global INPUT9_TYPE* num_processed_tokens,
    const __global INPUT10_TYPE* cache_interval,
    __global OUTPUT_TYPE* output,
    int head_dim_block,
    uint use_subgroup_reduction,
    __local float* work) {
    const uint lane = get_local_id(0);
    const uint runtime_lws = get_local_size(0);
    const uint subgroup_lane = get_sub_group_local_id();
    const uint subgroup_id = get_sub_group_id();
    const uint subgroup_count = get_num_sub_groups();
    const size_t h = get_global_id(1);
    const size_t seq = get_global_id(2);

    const size_t tokens = INPUT3_BATCH_NUM;
    const size_t num_heads = INPUT3_FEATURE_NUM;
    const size_t head_dim = INPUT3_SIZE_Y;
    const size_t num_groups = INPUT2_FEATURE_NUM;
    const size_t state_size = INPUT2_SIZE_Y;
    const uint lws = runtime_lws;
    const int block = head_dim_block;
    // This kernel is selected only when the full state fits in local memory,
    // so both the iteration count and state element index fit in uint.
    const uint state_iterations = (uint)(state_size / lws + (size_t)(state_size % lws != 0));
    const size_t p_base = get_group_id(0) * (size_t)block;
    const size_t num_state_blocks = INPUT5_BATCH_NUM;
    const size_t block_indices_count = INPUT7_BATCH_NUM;
    const size_t sequences = INPUT6_BATCH_NUM > 0 ? INPUT6_BATCH_NUM - 1 : 0;

    if (INPUT0_BATCH_NUM != num_heads ||
        INPUT1_BATCH_NUM != tokens || INPUT1_FEATURE_NUM != num_heads ||
        INPUT2_BATCH_NUM != tokens || INPUT4_BATCH_NUM != tokens ||
        INPUT4_FEATURE_NUM != num_groups || INPUT4_SIZE_Y != state_size ||
        INPUT5_FEATURE_NUM != num_heads || INPUT5_SIZE_Y != head_dim || INPUT5_SIZE_X != state_size ||
        INPUT8_BATCH_NUM < sequences + 1 || INPUT9_BATCH_NUM < sequences ||
        INPUT10_BATCH_NUM < sequences)
        return;

    if (seq >= sequences || h >= num_heads || p_base >= head_dim || num_groups == 0 ||
        num_heads % num_groups != 0 || block <= 0 || block > SSM_MAX_HEAD_DIM_BLOCK)
        return;

    const long token_begin = (long)subsequence_begins[GET_DATA_INDEX(INPUT6, seq, 0, 0, 0)];
    const long token_end = (long)subsequence_begins[GET_DATA_INDEX(INPUT6, seq + 1, 0, 0, 0)];
    const long block_begin = (long)block_indices_begins[GET_DATA_INDEX(INPUT8, seq, 0, 0, 0)];
    const long block_end = (long)block_indices_begins[GET_DATA_INDEX(INPUT8, seq + 1, 0, 0, 0)];

    if (token_begin < 0 || token_end < token_begin || (size_t)token_end > tokens)
        return;
    if (token_begin == token_end)
        return;

    if (block_begin < 0 || block_end <= block_begin || (size_t)block_end > block_indices_count) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
                for (int p_offset = 0; p_offset < block; ++p_offset) {
                    const size_t p = p_base + (size_t)p_offset;
                    if (p < head_dim)
                        output[GET_DATA_INDEX(OUTPUT, (size_t)token, h, p, 0)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const long first_block = (long)block_indices[GET_DATA_INDEX(INPUT7, (size_t)block_begin, 0, 0, 0)];
    if (first_block < 0 || (size_t)first_block >= num_state_blocks) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token) {
                for (int p_offset = 0; p_offset < block; ++p_offset) {
                    const size_t p = p_base + (size_t)p_offset;
                    if (p < head_dim)
                        output[GET_DATA_INDEX(OUTPUT, (size_t)token, h, p, 0)] = TO_OUTPUT_TYPE(0.0f);
                }
            }
        }
        return;
    }

    const long processed_raw = (long)num_processed_tokens[GET_DATA_INDEX(INPUT9, seq, 0, 0, 0)];
    const long interval = (long)cache_interval[GET_DATA_INDEX(INPUT10, seq, 0, 0, 0)];
    const long processed = max(processed_raw, (long)0);
    const bool cache_enabled = interval > 0;
    const ulong positive_interval = cache_enabled ? (ulong)interval : 1;
    const ulong previous_in_interval = cache_enabled ? (ulong)processed % positive_interval : 0;
    ulong tokens_until_boundary = cache_enabled ? positive_interval - previous_in_interval : 0;
    ulong write_slot = 1;
    const size_t heads_per_group = num_heads / num_groups;
    const size_t g = h / heads_per_group;
    const float A_value = ssm_to_float(A[GET_DATA_INDEX(INPUT0, h, 0, 0, 0)]);
    // Keep recurrent state in FP32 across tokens; cast only when writing output.
    __local float* local_state = work;
    __local float* reduction = work + (size_t)block * state_size;
    const int valid_head_dim_block = p_base + (size_t)block <= head_dim
                                         ? block
                                         : (int)(head_dim - p_base);

    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
        const size_t p = p_base + (size_t)p_offset;
        for (uint step = 0; step < state_iterations; ++step) {
            const uint state_element = step * lws + lane;
            if (state_element >= state_size)
                break;
            const size_t state_idx = GET_DATA_INDEX(INPUT5, (size_t)first_block, h, p, state_element);
            local_state[(size_t)p_offset * state_size + state_element] = ssm_to_float(recurrent_state_table[state_idx]);
        }
    }

    for (long token = token_begin; token < token_end; ++token) {
        const size_t token_idx = (size_t)token;
#define SSM_TOKEN_INDEX token_idx
#include "selective_ssm_recurrence.cl"
#undef SSM_TOKEN_INDEX

        const bool at_boundary = cache_enabled && --tokens_until_boundary == 0;
        const bool at_sequence_end = token + 1 == token_end;
        if (at_boundary || at_sequence_end) {
            const ulong block_position = (ulong)block_begin + write_slot++;
            if (block_position < (ulong)block_end && block_position < (ulong)block_indices_count) {
                const long block_id = (long)block_indices[GET_DATA_INDEX(INPUT7, (size_t)block_position, 0, 0, 0)];
                if (block_id >= 0 && (size_t)block_id < num_state_blocks) {
                    for (int p_offset = 0; p_offset < valid_head_dim_block; ++p_offset) {
                        const size_t p = p_base + (size_t)p_offset;
                        for (uint step = 0; step < state_iterations; ++step) {
                            const uint state_element = step * lws + lane;
                            if (state_element >= state_size)
                                break;
                            const size_t local_idx = (size_t)p_offset * state_size + state_element;
                            const size_t state_idx = GET_DATA_INDEX(INPUT5, (size_t)block_id, h, p, state_element);
                            recurrent_state_table[state_idx] = TO_INPUT5_TYPE(local_state[local_idx]);
                        }
                    }
                }
            }
            if (at_boundary)
                tokens_until_boundary = positive_interval;
        }
    }
}

#undef SSM_MAX_HEAD_DIM_BLOCK
#undef SSM_STATE_ITERATION_TYPE
#undef SSM_DT_INDEX
#undef SSM_B_INDEX
#undef SSM_C_INDEX
#undef SSM_X_INDEX
#undef SSM_OUTPUT_INDEX
#undef SSM_STATE_INDEX
#undef SSM_STATE_AT
