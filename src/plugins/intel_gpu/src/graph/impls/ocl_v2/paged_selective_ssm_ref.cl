// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/bf16_utils.cl"

#if INPUT0_IS_FP
#define SSM_TO_FLOAT(value) convert_float(value)
#else
#define SSM_TO_FLOAT(value) _convert_as_bfloat16_float(value)
#endif

KERNEL(paged_selective_ssm_ref)
(OPTIONAL_SHAPE_INFO_ARG
 __global INPUT0_TYPE* A,
 __global INPUT1_TYPE* dt,
 __global INPUT2_TYPE* B,
 __global INPUT3_TYPE* x,
 __global INPUT4_TYPE* C,
 __global INPUT5_TYPE* recurrent_state_table,
 __global INPUT6_TYPE* subsequence_begins,
 __global INPUT7_TYPE* block_indices,
 __global INPUT8_TYPE* block_indices_begins,
 __global INPUT9_TYPE* num_processed_tokens,
 __global INPUT10_TYPE* cache_interval,
 __global OUTPUT_TYPE* output) {
    const int seq = get_global_id(0);
    const int h = get_global_id(1);
    const int p = get_global_id(2);
    const int sequences = max(INPUT6_BATCH_NUM - 1, 0);
    const int token_count = INPUT3_BATCH_NUM;
    const int num_heads = INPUT3_FEATURE_NUM;
    const int head_dim = INPUT3_Y_NUM;
    const int num_groups = INPUT2_FEATURE_NUM;
    const int state_size = INPUT2_Y_NUM;
    const int state_blocks = INPUT5_BATCH_NUM;
    if (seq >= sequences || h >= num_heads || p >= head_dim || num_groups <= 0)
        return;

    const int heads_per_group = num_heads / num_groups;
    if (heads_per_group <= 0)
        return;
    const int g = h / heads_per_group;

    if (seq + 1 >= INPUT8_BATCH_NUM || seq >= INPUT9_BATCH_NUM || seq >= INPUT10_BATCH_NUM)
        return;
    const long token_begin = convert_long(subsequence_begins[INPUT6_GET_INDEX(seq, 0, 0, 0)]);
    const long token_end = convert_long(subsequence_begins[INPUT6_GET_INDEX(seq + 1, 0, 0, 0)]);
    const long block_begin = convert_long(block_indices_begins[INPUT8_GET_INDEX(seq, 0, 0, 0)]);
    const long block_end = convert_long(block_indices_begins[INPUT8_GET_INDEX(seq + 1, 0, 0, 0)]);
    if (token_begin < 0 || token_end < token_begin || token_end > token_count || block_begin < 0 ||
        block_end <= block_begin || block_end > INPUT7_BATCH_NUM)
        return;

    const long seq_blocks = block_end - block_begin;
    const long first_block = convert_long(block_indices[INPUT7_GET_INDEX(block_begin, 0, 0, 0)]);
    if (first_block < 0 || first_block >= state_blocks)
        return;

    const long interval = convert_long(cache_interval[INPUT10_GET_INDEX(seq, 0, 0, 0)]);
    const long processed = max(convert_long(num_processed_tokens[INPUT9_GET_INDEX(seq, 0, 0, 0)]), (long)0);
    const long prev_nums = interval > 0 ? processed % interval : 0;
    for (long token = token_begin; token < token_end; token++) {
        float acc = 0.0f;

        const long step = token - token_begin + 1;
        const long cached_tokens = prev_nums + step;
        const bool reached_interval_boundary = interval > 0 && (cached_tokens % interval == 0);
        const bool reached_sequence_end = token == token_end - 1;
        const long destination_slot = interval > 0 ? 1 + (cached_tokens - 1) / interval : -1;
        const bool write_cache = interval > 0 && (reached_interval_boundary || reached_sequence_end) && destination_slot < seq_blocks;
        const long destination_block = write_cache ?
            convert_long(block_indices[INPUT7_GET_INDEX(block_begin + destination_slot, 0, 0, 0)]) : -1;

        // The recurrent state is deliberately scalar. This keeps the implementation valid for
        // any state size and preserves FP32 accumulation until an observable cache checkpoint.
        for (int n = 0; n < state_size; n++) {
            float state = SSM_TO_FLOAT(recurrent_state_table[INPUT5_GET_INDEX(first_block, h, p, n)]);
            for (long previous = token_begin; previous <= token; previous++) {
                const float previous_dt = SSM_TO_FLOAT(dt[INPUT1_GET_INDEX(previous, h, 0, 0)]);
                const float previous_x = SSM_TO_FLOAT(x[INPUT3_GET_INDEX(previous, h, p, 0)]);
                const float previous_dA = exp(SSM_TO_FLOAT(A[INPUT0_GET_INDEX(h, 0, 0, 0)]) * previous_dt);
                state = state * previous_dA + previous_x * previous_dt * SSM_TO_FLOAT(B[INPUT2_GET_INDEX(previous, g, n, 0)]);
            }
            acc = fma(state, SSM_TO_FLOAT(C[INPUT4_GET_INDEX(token, g, n, 0)]), acc);
            if (write_cache && destination_block >= 0 && destination_block < state_blocks) {
                recurrent_state_table[INPUT5_GET_INDEX(destination_block, h, p, n)] = TO_INPUT5_TYPE(state);
            }
        }
        output[OUTPUT_GET_INDEX(token, h, p, 0)] = TO_OUTPUT_TYPE(acc);
    }
}
