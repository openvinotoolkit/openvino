// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// Token-parallel variant of paged_causal_conv1d_ref.
// Each work-item handles one hidden channel over a range of tokens_per_item flattened tokens and
// rebuilds its convolution window from the input, so the sequential chain is bounded by the range
// instead of the whole sequence.
// The work-item that owns the first token of a sequence (head item) also covers its first
// KERNEL_SIZE - 1 tokens and is the only one that reads the cached state (slot 0) or writes state
// slots. Slot 0 may alias a write slot (e.g. block_indices = {row, row}), so all state accesses of
// a sequence stay ordered within one work-item, as in the reference kernel.
KERNEL(paged_causal_conv1d_opt)
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
 int num_blocks,
 int token_count,
 int tokens_per_item,
 int input_offset,
 int state_offset,
 int weight_offset,
 int bias_offset,
 int subsequence_offset,
 int block_indices_offset,
 int block_begins_offset,
 int past_lens_offset,
 int cache_interval_offset,
 int output_offset) {
    input_embeds += input_offset;
    conv_state_table += state_offset;
    conv_weight += weight_offset;
    conv_bias += bias_offset;
    subsequence_begins += subsequence_offset;
    block_indices += block_indices_offset;
    block_indices_begins += block_begins_offset;
    past_lens += past_lens_offset;
    cache_interval += cache_interval_offset;
    output_embeds += output_offset;

    const int chunk = (int)get_global_id(0);
    const int h = (int)get_global_id(1);

    if (h >= hidden_size || seq_count <= 0)
        return;

    const int chunk_begin = chunk * tokens_per_item;
    if (chunk_begin >= token_count)
        return;
    const int chunk_end = min(chunk_begin + tokens_per_item, token_count);

    // Last sequence that starts at or before chunk_begin (sequence 0 if none does).
    int first_seq = 0;
    int last_seq = seq_count - 1;
    while (first_seq < last_seq) {
        const int mid = (first_seq + last_seq + 1) >> 1;
        if (subsequence_begins[mid] <= chunk_begin)
            first_seq = mid;
        else
            last_seq = mid - 1;
    }

    float weight[KERNEL_SIZE];
    const int w_base = h * weight_hidden_stride;
    for (int k = 0; k < KERNEL_SIZE; k++) {
        weight[k] = convert_float(conv_weight[w_base + k * weight_kernel_stride]);
    }

    float bias_val = 0.0f;
#if HAS_BIAS
    bias_val = convert_float(conv_bias[h * bias_hidden_stride]);
#endif

    const int in_h = h * input_hidden_stride;
    const int out_h = h * output_hidden_stride;

    for (int seq = first_seq; seq < seq_count; seq++) {
        const int token_begin = subsequence_begins[seq];
        if (token_begin >= chunk_end)
            break;

        const int token_end = subsequence_begins[seq + 1];
        if (token_begin < 0 || token_end <= token_begin)
            continue;

        const int seq_tokens = token_end - token_begin;
        const bool is_head = token_begin >= chunk_begin;
        const int range_end = min(chunk_end, token_end) - token_begin;
        const int lt_begin = is_head ? 0 : max(chunk_begin - token_begin, KERNEL_SIZE - 1);
        const int lt_end = is_head ? max(range_end, min(KERNEL_SIZE - 1, seq_tokens)) : range_end;
        if (lt_begin >= lt_end)
            continue;

        const int blk_begin = block_indices_begins[seq];
        const int block_span = block_indices_begins[seq + 1] - blk_begin;
        const int read_physical_block = (block_span > 1) ? block_indices[blk_begin] : -1;
        if (read_physical_block < 0 || read_physical_block >= num_blocks) {
            for (int lt = lt_begin; lt < lt_end; lt++) {
                output_embeds[(token_begin + lt) * output_token_stride + out_h] = TO_OUTPUT_TYPE(0.0f);
            }
            continue;
        }

        const int seq_interval = cache_interval[seq];
        const int prev_nums = (seq_interval > 0) ? (past_lens[seq] % seq_interval) : 0;

        // state[0] is shifted out before its first use.
        float state[KERNEL_SIZE];
        state[0] = 0.0f;
        if (is_head) {
            const int read_state_base = read_physical_block * state_block_stride + h * state_hidden_stride;
            for (int k = 1; k < KERNEL_SIZE; k++) {
                state[k] = convert_float(conv_state_table[read_state_base + k * state_kernel_stride]);
            }
        } else {
            for (int k = 1; k < KERNEL_SIZE; k++) {
                state[k] = convert_float(input_embeds[(token_begin + lt_begin - KERNEL_SIZE + k) * input_token_stride + in_h]);
            }
        }

        for (int lt = lt_begin; lt < lt_end; lt++) {
            const int token_idx = token_begin + lt;

            for (int k = 0; k + 1 < KERNEL_SIZE; k++) {
                state[k] = state[k + 1];
            }
            state[KERNEL_SIZE - 1] = convert_float(input_embeds[token_idx * input_token_stride + in_h]);

            float sum = bias_val;
            for (int k = 0; k < KERNEL_SIZE; k++) {
                sum = fma(state[k], weight[k], sum);
            }
            output_embeds[token_idx * output_token_stride + out_h] = TO_OUTPUT_TYPE(sum);

            if (is_head) {
                const int cached_tokens = prev_nums + (lt + 1);
                const int interval_hit = (seq_interval > 0) && ((cached_tokens % seq_interval) == 0);
                if (interval_hit || lt == seq_tokens - 1) {
                    const int slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                    if (slot >= 1 && slot < block_span) {
                        const int physical_block = block_indices[blk_begin + slot];
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

        // State checkpoints after the head range. Here lt_end >= KERNEL_SIZE - 1, so every
        // checkpoint window lies entirely in the input.
        if (is_head && lt_end < seq_tokens) {
            const int last = seq_tokens - 1;
            int lt_event = last;
            if (seq_interval > 0) {
                int first_hit = seq_interval - prev_nums - 1;
                if (first_hit < lt_end) {
                    first_hit += ((lt_end - first_hit + seq_interval - 1) / seq_interval) * seq_interval;
                }
                lt_event = min(first_hit, last);
            }

            for (;;) {
                const int cached_tokens = prev_nums + (lt_event + 1);
                const int slot = (seq_interval > 0) ? (1 + (cached_tokens - 1) / seq_interval) : 1;
                if (slot >= 1 && slot < block_span) {
                    const int physical_block = block_indices[blk_begin + slot];
                    if (physical_block >= 0 && physical_block < num_blocks) {
                        const int state_base = physical_block * state_block_stride + h * state_hidden_stride;
                        const int window_begin = token_begin + lt_event - KERNEL_SIZE + 1;
                        for (int k = 0; k < KERNEL_SIZE; k++) {
                            const float v = convert_float(input_embeds[(window_begin + k) * input_token_stride + in_h]);
                            conv_state_table[state_base + k * state_kernel_stride] = TO_INPUT1_TYPE(v);
                        }
                    }
                }
                if (lt_event >= last)
                    break;
                lt_event = min(lt_event + seq_interval, last);
            }
        }
    }
}
