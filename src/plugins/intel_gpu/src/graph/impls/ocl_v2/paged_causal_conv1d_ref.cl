// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

KERNEL(paged_causal_conv1d_ref)
(__global INPUT0_TYPE* input_embeds,
 __global INPUT1_TYPE* conv_state_table,
 __global INPUT2_TYPE* conv_weight,
 __global INPUT3_TYPE* conv_bias,
 __global INPUT4_TYPE* subsequence_begins,
 __global INPUT5_TYPE* block_indices,
 __global INPUT6_TYPE* block_indices_begins,
 __global INPUT7_TYPE* past_lens,
 __global INPUT8_TYPE* cache_interval,
 __global OUTPUT_TYPE* output_embeds,
 int seq_count,
 int hidden_size,
 int input_token_stride,
 int input_hidden_stride,
 int state_block_stride,
 int state_hidden_stride,
 int state_kernel_stride,
 int weight_hidden_stride,
 int weight_kernel_stride,
 int bias_hidden_stride,
 int output_token_stride,
 int output_hidden_stride,
 int num_blocks) {
    const int seq = (int)get_global_id(0);
    const int h = (int)get_global_id(1);

    if (seq >= seq_count || h >= hidden_size)
        return;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int blk_begin = block_indices_begins[seq];
    const int blk_end = block_indices_begins[seq + 1];

    if (token_begin < 0 || token_end < token_begin || blk_end <= blk_begin)
        return;

    const int block_span = blk_end - blk_begin;
    if (block_span <= 1)
        return;

    const int read_physical_block = block_indices[blk_begin];
    if (read_physical_block < 0 || read_physical_block >= num_blocks)
        return;

    float state[KERNEL_SIZE];

    const int read_state_base = read_physical_block * state_block_stride + h * state_hidden_stride;
    for (int k = 0; k < KERNEL_SIZE; k++) {
        state[k] = convert_float(conv_state_table[read_state_base + k * state_kernel_stride]);
    }

    float bias_val = 0.0f;
#if HAS_BIAS
    bias_val = convert_float(conv_bias[h * bias_hidden_stride]);
#endif

    for (int t = 0; t < token_end - token_begin; t++) {
        const int token_idx = token_begin + t;

        for (int k = 0; k + 1 < KERNEL_SIZE; k++) {
            state[k] = state[k + 1];
        }

        const int in_off = token_idx * input_token_stride + h * input_hidden_stride;
        state[KERNEL_SIZE - 1] = convert_float(input_embeds[in_off]);

        float sum = bias_val;
        const int w_base = h * weight_hidden_stride;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            sum = fma(state[k], convert_float(conv_weight[w_base + k * weight_kernel_stride]), sum);
        }

        const int out_off = token_idx * output_token_stride + h * output_hidden_stride;
        output_embeds[out_off] = TO_OUTPUT_TYPE(sum);

        const int interval = cache_interval[seq];
        if (interval > 0) {
            const int processed_tokens = t + 1;
            if ((processed_tokens % interval) == 0) {
                const int logical_block = (processed_tokens + interval - 1) / interval;
                if (logical_block >= 1 && logical_block < block_span) {
                    const int physical_block = block_indices[blk_begin + logical_block];
                    if (physical_block >= 0 && physical_block < num_blocks) {
                        const int state_base = physical_block * state_block_stride + h * state_hidden_stride;
                        for (int k = 0; k < KERNEL_SIZE; k++) {
                            conv_state_table[state_base + k * state_kernel_stride] = TO_INPUT1_TYPE(state[k]);
                        }
                    }
                }
            }
        }
    }

    const int seq_tokens = token_end - token_begin;
    const int interval = cache_interval[seq];
    int final_logical_block = 1;
    if (interval > 0) {
        final_logical_block = (seq_tokens + interval - 1) / interval;
    }
    if (final_logical_block >= block_span) {
        final_logical_block = block_span - 1;
    }

    const int final_physical_block = block_indices[blk_begin + final_logical_block];
    if (final_physical_block >= 0 && final_physical_block < num_blocks) {
        const int final_state_base = final_physical_block * state_block_stride + h * state_hidden_stride;
        for (int k = 0; k < KERNEL_SIZE; k++) {
            conv_state_table[final_state_base + k * state_kernel_stride] = TO_INPUT1_TYPE(state[k]);
        }
    }
}
