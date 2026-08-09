// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

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
    __local float* work) {
    const size_t lane = get_local_id(0);
    const size_t lws = get_local_size(0);
    const size_t p = get_group_id(0);
    const size_t h = get_global_id(1);
    const size_t seq = get_global_id(2);

    const size_t tokens = INPUT3_BATCH_NUM;
    const size_t num_heads = INPUT3_FEATURE_NUM;
    const size_t head_dim = INPUT3_SIZE_Y;
    const size_t num_groups = INPUT2_FEATURE_NUM;
    const size_t state_size = INPUT2_SIZE_Y;
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

    if (seq >= sequences || h >= num_heads || p >= head_dim || num_groups == 0 || num_heads % num_groups != 0)
        return;

    const long token_begin = (long)subsequence_begins[INPUT6_GET_INDEX(seq, 0, 0, 0)];
    const long token_end = (long)subsequence_begins[INPUT6_GET_INDEX(seq + 1, 0, 0, 0)];
    const long block_begin = (long)block_indices_begins[INPUT8_GET_INDEX(seq, 0, 0, 0)];
    const long block_end = (long)block_indices_begins[INPUT8_GET_INDEX(seq + 1, 0, 0, 0)];

    if (token_begin < 0 || token_end < token_begin || (size_t)token_end > tokens)
        return;
    if (token_begin == token_end)
        return;

    if (block_begin < 0 || block_end <= block_begin || (size_t)block_end > block_indices_count) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token)
                output[OUTPUT_GET_INDEX((size_t)token, h, p, 0)] = TO_OUTPUT_TYPE(0.0f);
        }
        return;
    }

    const long first_block = (long)block_indices[INPUT7_GET_INDEX((size_t)block_begin, 0, 0, 0)];
    if (first_block < 0 || (size_t)first_block >= num_state_blocks) {
        if (lane == 0) {
            for (long token = token_begin; token < token_end; ++token)
                output[OUTPUT_GET_INDEX((size_t)token, h, p, 0)] = TO_OUTPUT_TYPE(0.0f);
        }
        return;
    }

    const long processed_raw = (long)num_processed_tokens[INPUT9_GET_INDEX(seq, 0, 0, 0)];
    const long interval = (long)cache_interval[INPUT10_GET_INDEX(seq, 0, 0, 0)];
    const long processed = max(processed_raw, (long)0);
    const ulong previous_in_interval = interval > 0 ? (ulong)processed % (ulong)interval : 0;
    const long sequence_blocks = block_end - block_begin;
    const size_t heads_per_group = num_heads / num_groups;
    const size_t g = h / heads_per_group;
    __local float* local_state = work;
    __local float* reduction = work + state_size;

    for (size_t n = lane; n < state_size; n += lws) {
        const size_t state_idx = INPUT5_GET_INDEX((size_t)first_block, h, p, n);
        local_state[n] = SSM_TO_FLOAT(recurrent_state_table[state_idx]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (long token = token_begin; token < token_end; ++token) {
        const size_t token_idx = (size_t)token;
        const size_t dt_idx = INPUT1_GET_INDEX(token_idx, h, 0, 0);
        const size_t x_idx = INPUT3_GET_INDEX(token_idx, h, p, 0);
        const float dt_value = SSM_TO_FLOAT(dt[dt_idx]);
        const float dA = exp(SSM_TO_FLOAT(A[INPUT0_GET_INDEX(h, 0, 0, 0)]) * dt_value);
        const float x_value = SSM_TO_FLOAT(x[x_idx]);
        float partial = 0.0f;

        for (size_t n = lane; n < state_size; n += lws) {
            const size_t b_idx = INPUT2_GET_INDEX(token_idx, g, n, 0);
            const size_t c_idx = INPUT4_GET_INDEX(token_idx, g, n, 0);
            const float new_state = fma(local_state[n], dA,
                                        x_value * dt_value * SSM_TO_FLOAT(B[b_idx]));
            const float stored_state = SSM_ROUND_STATE(new_state);
            local_state[n] = stored_state;
            partial = fma(stored_state, SSM_TO_FLOAT(C[c_idx]), partial);
        }

        reduction[lane] = partial;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t offset = lws / 2; offset > 0; offset /= 2) {
            if (lane < offset)
                reduction[lane] += reduction[lane + offset];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lane == 0)
            output[OUTPUT_GET_INDEX(token_idx, h, p, 0)] = TO_OUTPUT_TYPE(reduction[0]);

        if (interval > 0) {
            const ulong current_tokens = (ulong)(token - token_begin + 1);
            const ulong cached_tokens = previous_in_interval + current_tokens;
            const bool at_boundary = cached_tokens % (ulong)interval == 0;
            const bool at_sequence_end = token + 1 == token_end;
            if (at_boundary || at_sequence_end) {
                const ulong slot = 1 + (cached_tokens - 1) / (ulong)interval;
                const ulong block_position = (ulong)block_begin + slot;
                if (slot < (ulong)sequence_blocks && block_position < (ulong)block_indices_count) {
                    const long block_id = (long)block_indices[INPUT7_GET_INDEX((size_t)block_position, 0, 0, 0)];
                    if (block_id >= 0 && (size_t)block_id < num_state_blocks) {
                        for (size_t n = lane; n < state_size; n += lws) {
                            const size_t state_idx = INPUT5_GET_INDEX((size_t)block_id, h, p, n);
                            recurrent_state_table[state_idx] = TO_INPUT5_TYPE(local_state[n]);
                        }
                    }
                }
            }
        }
    }
}
