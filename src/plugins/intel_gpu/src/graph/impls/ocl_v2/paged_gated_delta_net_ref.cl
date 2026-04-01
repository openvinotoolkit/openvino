// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

#define K_SLICE_SIZE ((K_HEAD_DIM + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(paged_gated_delta_net_ref)
(__global INPUT0_TYPE* query,
 __global INPUT1_TYPE* key,
 __global INPUT2_TYPE* value,
 __global INPUT3_TYPE* recurrent_state_table,
 __global INPUT4_TYPE* gate,
 __global INPUT5_TYPE* beta,
 __global INPUT6_TYPE* subsequence_begins,
 __global INPUT7_TYPE* block_indices,
 __global INPUT8_TYPE* block_indices_begins,
 __global INPUT9_TYPE* past_lens,
 __global INPUT10_TYPE* cache_interval,
 __global OUTPUT_TYPE* output,
 int num_sequences,
 int query_head_offset,
 int key_head_offset,
 int value_head_offset,
 int q_token_stride,
 int q_head_stride,
 int k_token_stride,
 int k_head_stride,
 int v_token_stride,
 int v_head_stride) {
    const int seq = get_global_id(0);
    const int h = get_global_id(1);
    const int v_block = get_group_id(2);
    const int lid = get_sub_group_local_id();

    if (seq >= num_sequences || h >= V_HEAD_NUM)
        return;

    const int start_iv = v_block * V_BLOCK_SIZE;
    if (start_iv >= V_HEAD_DIM)
        return;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int block_end = block_indices_begins[seq + 1];
    const int seq_blocks = max(block_end - block_begin, 0);
    const int past_len = past_lens[seq];
    const int interval = cache_interval[seq];

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;

    float state[V_BLOCK_SIZE][K_SLICE_SIZE];
    float q_norm[K_SLICE_SIZE];
    float k_norm[K_SLICE_SIZE];

    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
            state[v_idx][ks] = 0.0f;
        }

        if (curr_iv >= V_HEAD_DIM)
            continue;

        if (interval > 0 && seq_blocks > 0 && past_len > 0) {
            const int read_slot = 0;
            if (read_slot < seq_blocks) {
                int block_id = block_indices[block_begin + read_slot];
                int base = (block_id * V_HEAD_NUM + h) * (K_HEAD_DIM * V_HEAD_DIM) + curr_iv;
                for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                    const int k_idx = lid + ks * SUBGROUP_SIZE;
                    if (k_idx < K_HEAD_DIM) {
                        state[v_idx][ks] = convert_float(recurrent_state_table[base + k_idx * V_HEAD_DIM]);
                    }
                }
            }
        }
    }

    for (int token = token_begin; token < token_end; token++) {
        const int q_base = token * q_token_stride + (hk + query_head_offset) * q_head_stride;
        const int k_base = token * k_token_stride + (hk + key_head_offset) * k_head_stride;

        float q_sum_local = 0.0f;
        float k_sum_local = 0.0f;
        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
            const int k_idx = lid + ks * SUBGROUP_SIZE;
            if (k_idx < K_HEAD_DIM) {
                const float q_val = convert_float(query[q_base + k_idx]);
                const float k_val = convert_float(key[k_base + k_idx]);
                q_norm[ks] = q_val;
                k_norm[ks] = k_val;
                q_sum_local = fma(q_val, q_val, q_sum_local);
                k_sum_local = fma(k_val, k_val, k_sum_local);
            } else {
                q_norm[ks] = 0.0f;
                k_norm[ks] = 0.0f;
            }
        }

        float q_sum = sub_group_reduce_add(q_sum_local);
        q_sum = sub_group_broadcast(q_sum, 0);
        float k_sum = sub_group_reduce_add(k_sum_local);
        k_sum = sub_group_broadcast(k_sum, 0);

        const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, 1e-6f);
        const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, 1e-6f);
        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
            q_norm[ks] *= q_scale;
            k_norm[ks] *= k_scale;
        }

        const int g_idx = token * V_HEAD_NUM + h;
        const int beta_idx = token * V_HEAD_NUM + h;
        const float b_g = exp(convert_float(gate[g_idx]));
        const float b_beta = convert_float(beta[beta_idx]);

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            if (curr_iv >= V_HEAD_DIM)
                continue;

            const int v_idx_offset = token * v_token_stride + (h + value_head_offset) * v_head_stride + curr_iv;
            const float b_v = convert_float(value[v_idx_offset]);

            float h_k_local = 0.0f;
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                state[v_idx][ks] *= b_g;
                h_k_local = fma(state[v_idx][ks], k_norm[ks], h_k_local);
            }
            float h_k = sub_group_reduce_add(h_k_local);
            h_k = sub_group_broadcast(h_k, 0);

            const float update_val = (b_v - h_k) * b_beta;

            float out_val_local = 0.0f;
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                state[v_idx][ks] = fma(k_norm[ks], update_val, state[v_idx][ks]);
                out_val_local = fma(state[v_idx][ks], q_norm[ks], out_val_local);
            }
            float out_val = sub_group_reduce_add(out_val_local);
            out_val = sub_group_broadcast(out_val, 0);

            if (lid == 0) {
                const int out_offset = (token * V_HEAD_NUM + h) * V_HEAD_DIM + curr_iv;
                output[out_offset] = TO_OUTPUT_TYPE(out_val);
            }
        }

        if (interval > 0 && seq_blocks > 0) {
            const int local_token_idx = token - token_begin;
            const int processed_tokens = local_token_idx + 1;
            const bool should_store = ((processed_tokens % interval) == 0) || (token == token_end - 1);
            if (should_store) {
                const int slot = (processed_tokens + interval - 1) / interval;
                if (slot < seq_blocks) {
                    const int block_id = block_indices[block_begin + slot];
                    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                        int curr_iv = start_iv + v_idx;
                        if (curr_iv >= V_HEAD_DIM)
                            continue;

                        int base = (block_id * V_HEAD_NUM + h) * (K_HEAD_DIM * V_HEAD_DIM) + curr_iv;
                        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                            const int k_idx = lid + ks * SUBGROUP_SIZE;
                            if (k_idx < K_HEAD_DIM) {
                                recurrent_state_table[base + k_idx * V_HEAD_DIM] = (INPUT3_TYPE)(state[v_idx][ks]);
                            }
                        }
                    }
                }
            }
        }
    }
}
