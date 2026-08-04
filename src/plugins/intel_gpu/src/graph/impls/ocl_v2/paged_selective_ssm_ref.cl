// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(paged_selective_ssm_ref)
(__global INPUT0_TYPE* A,
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
 __global OUTPUT_TYPE* output,
 int num_heads,
 int num_groups,
 int head_dim,
 int state_size) {
    const int seq = get_global_id(0);
    const int h = get_global_id(1);
    const int p = get_global_id(2);
    const int heads_per_group = num_heads / num_groups;
    const int g = h / heads_per_group;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int block_end = block_indices_begins[seq + 1];
    const int seq_blocks = max(block_end - block_begin, 0);
    const int processed = num_processed_tokens[seq];
    const int interval = cache_interval[seq];
    const int prev_nums = interval > 0 ? processed % interval : 0;
    const int first_block = block_indices[block_begin];
    const int state_stride = head_dim * state_size;

    float state[512];
    for (int n = 0; n < state_size; n++) {
        state[n] = convert_float(recurrent_state_table[((first_block * num_heads + h) * head_dim + p) * state_size + n]);
    }

    for (int token = token_begin; token < token_end; token++) {
        const int dt_offset = token * num_heads + h;
        const int bc_base = (token * num_groups + g) * state_size;
        const int x_offset = (token * num_heads + h) * head_dim + p;
        const float dt_value = convert_float(dt[dt_offset]);
        const float dA = exp(convert_float(A[h]) * dt_value);
        const float x_value = convert_float(x[x_offset]);

        float acc = 0.0f;
        for (int n = 0; n < state_size; n++) {
            state[n] = state[n] * dA + x_value * dt_value * convert_float(B[bc_base + n]);
            acc = fma(state[n], convert_float(C[bc_base + n]), acc);
        }
        output[x_offset] = TO_OUTPUT_TYPE(acc);

        const int processed_tokens = (token - token_begin) + 1;
        const int cached_tokens = prev_nums + processed_tokens;
        const bool reached_interval_boundary = interval > 0 && ((cached_tokens % interval) == 0);
        const bool reached_sequence_end = token == token_end - 1;
        if (reached_interval_boundary || reached_sequence_end) {
            const int slot = interval > 0 ? (1 + (cached_tokens - 1) / interval) : 1;
            if (slot < seq_blocks) {
                const int block_id = block_indices[block_begin + slot];
                for (int n = 0; n < state_size; n++) {
                    recurrent_state_table[((block_id * num_heads + h) * head_dim + p) * state_size + n] = (INPUT5_TYPE)(state[n]);
                }
            }
        }
    }
}
