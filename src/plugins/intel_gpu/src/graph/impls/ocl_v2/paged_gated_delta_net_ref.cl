// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

#define K_SLICE_SIZE ((K_HEAD_DIM + SUBGROUP_SIZE - 1) / SUBGROUP_SIZE)

#ifndef K_VEC_SIZE
#define K_VEC_SIZE 1
#endif

#if ((K_HEAD_DIM % 16) == 0) && ((V_HEAD_DIM % 16) == 0) && (K_VEC_SIZE > 1) && ((K_HEAD_DIM % (SUBGROUP_SIZE * K_VEC_SIZE)) == 0)
#define KV_OPT_VEC_PATH 1
#define K_LANE_ELEMS (K_HEAD_DIM / SUBGROUP_SIZE)

typedef MAKE_VECTOR_TYPE(float, K_VEC_SIZE) K_VEC_TYPE;
#define K_VEC_ZERO ((K_VEC_TYPE)(0.0f))
#define K_VEC_LOAD_Q(ptr, idx) CAT(convert_float, K_VEC_SIZE)(CAT(vload, K_VEC_SIZE)(0, (ptr) + (idx)))
#define K_VEC_LOAD_K(ptr, idx) CAT(convert_float, K_VEC_SIZE)(CAT(vload, K_VEC_SIZE)(0, (ptr) + (idx)))
#define K_VEC_FROM_TMP(tmp) CAT(vload, K_VEC_SIZE)(0, (tmp))
#define K_VEC_TO_TMP(vec, tmp) CAT(vstore, K_VEC_SIZE)((vec), 0, (tmp))

#if (K_VEC_SIZE == 8)
#define K_VEC_DOT(a, b) (dot((a).lo, (b).lo) + dot((a).hi, (b).hi))
#else
#define K_VEC_DOT(a, b) dot((a), (b))
#endif

#define K_VEC_COUNT (K_LANE_ELEMS / K_VEC_SIZE)
#else
#define KV_OPT_VEC_PATH 0
#endif

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
    const int start_iv = v_block * V_BLOCK_SIZE;

    const int token_begin = subsequence_begins[seq];
    const int token_end = subsequence_begins[seq + 1];
    const int block_begin = block_indices_begins[seq];
    const int block_end = block_indices_begins[seq + 1];
    const int interval = cache_interval[seq];

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;

#if KV_OPT_VEC_PATH
    K_VEC_TYPE state[V_BLOCK_SIZE][K_VEC_COUNT];
    K_VEC_TYPE q_norm[K_VEC_COUNT];
    K_VEC_TYPE k_norm[K_VEC_COUNT];
#else
    float state[V_BLOCK_SIZE][K_SLICE_SIZE];
    float q_norm[K_SLICE_SIZE];
    float k_norm[K_SLICE_SIZE];
#endif

    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
        if (curr_iv >= V_HEAD_DIM)
            continue;
#endif

        {
            int block_id = block_indices[block_begin];
            int base = (block_id * V_HEAD_NUM + h) * (K_HEAD_DIM * V_HEAD_DIM) + curr_iv;
#if KV_OPT_VEC_PATH
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                float tmp[K_VEC_SIZE];
                for (int ke = 0; ke < K_VEC_SIZE; ke++) {
                    const int k_idx = lid * K_LANE_ELEMS + kc * K_VEC_SIZE + ke;
                    tmp[ke] = convert_float(recurrent_state_table[base + k_idx * V_HEAD_DIM]);
                }
                state[v_idx][kc] = K_VEC_FROM_TMP(tmp);
            }
#else
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                const int k_idx = lid + ks * SUBGROUP_SIZE;
                if (k_idx < K_HEAD_DIM) {
                    state[v_idx][ks] = convert_float(recurrent_state_table[base + k_idx * V_HEAD_DIM]);
                }
            }
#endif
        }
    }

    for (int token = token_begin; token < token_end; token++) {
        const int q_base = token * q_token_stride + (hk + query_head_offset) * q_head_stride;
        const int k_base = token * k_token_stride + (hk + key_head_offset) * k_head_stride;

        float q_sum_local = 0.0f;
        float k_sum_local = 0.0f;
#if KV_OPT_VEC_PATH
        const int q_lane_base = q_base + lid * K_LANE_ELEMS;
        const int k_lane_base = k_base + lid * K_LANE_ELEMS;
        for (int kc = 0; kc < K_VEC_COUNT; kc++) {
            const int offset = kc * K_VEC_SIZE;
            q_norm[kc] = K_VEC_LOAD_Q(query, q_lane_base + offset);
            k_norm[kc] = K_VEC_LOAD_K(key, k_lane_base + offset);
            q_sum_local += K_VEC_DOT(q_norm[kc], q_norm[kc]);
            k_sum_local += K_VEC_DOT(k_norm[kc], k_norm[kc]);
        }
#else
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
#endif

        float q_sum = sub_group_reduce_add(q_sum_local);
        q_sum = sub_group_broadcast(q_sum, 0);
        float k_sum = sub_group_reduce_add(k_sum_local);
        k_sum = sub_group_broadcast(k_sum, 0);

        const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, 1e-6f);
        const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, 1e-6f);
#if KV_OPT_VEC_PATH
        for (int kc = 0; kc < K_VEC_COUNT; kc++) {
            q_norm[kc] *= q_scale;
            k_norm[kc] *= k_scale;
        }
#else
        for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
            q_norm[ks] *= q_scale;
            k_norm[ks] *= k_scale;
        }
#endif

        const int g_idx = token * V_HEAD_NUM + h;
        const int beta_idx = token * V_HEAD_NUM + h;
        const float b_g = exp(convert_float(gate[g_idx]));
        const float b_beta = convert_float(beta[beta_idx]);

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
            if (curr_iv >= V_HEAD_DIM)
                continue;
#endif

            const int v_idx_offset = token * v_token_stride + (h + value_head_offset) * v_head_stride + curr_iv;
            const float b_v = convert_float(value[v_idx_offset]);

            float h_k_local = 0.0f;
#if KV_OPT_VEC_PATH
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                state[v_idx][kc] *= b_g;
                h_k_local += K_VEC_DOT(state[v_idx][kc], k_norm[kc]);
            }
#else
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                state[v_idx][ks] *= b_g;
                h_k_local = fma(state[v_idx][ks], k_norm[ks], h_k_local);
            }
#endif
            float h_k = sub_group_reduce_add(h_k_local);
            h_k = sub_group_broadcast(h_k, 0);

            const float update_val = (b_v - h_k) * b_beta;

            float out_val_local = 0.0f;
#if KV_OPT_VEC_PATH
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                state[v_idx][kc] = fma(k_norm[kc], update_val, state[v_idx][kc]);
                out_val_local += K_VEC_DOT(state[v_idx][kc], q_norm[kc]);
            }
#else
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                state[v_idx][ks] = fma(k_norm[ks], update_val, state[v_idx][ks]);
                out_val_local = fma(state[v_idx][ks], q_norm[ks], out_val_local);
            }
#endif
            float out_val = sub_group_reduce_add(out_val_local);
            out_val = sub_group_broadcast(out_val, 0);

            if (lid == 0) {
                const int out_offset = (token * V_HEAD_NUM + h) * V_HEAD_DIM + curr_iv;
                output[out_offset] = TO_OUTPUT_TYPE(out_val);
            }
        }

        if (interval > 0) {
            const int local_token_idx = token - token_begin;
            const int processed_tokens = local_token_idx + 1;
            const bool should_store = ((processed_tokens % interval) == 0) || (token == token_end - 1);
            if (should_store) {
                const int slot = (processed_tokens + interval - 1) / interval;
                const int block_id = block_indices[block_begin + slot];
                for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                    int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
                    if (curr_iv >= V_HEAD_DIM)
                        continue;
#endif

                    int base = (block_id * V_HEAD_NUM + h) * (K_HEAD_DIM * V_HEAD_DIM) + curr_iv;
#if KV_OPT_VEC_PATH
                    for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                        float tmp[K_VEC_SIZE];
                        K_VEC_TO_TMP(state[v_idx][kc], tmp);
                        for (int ke = 0; ke < K_VEC_SIZE; ke++) {
                            const int k_idx = lid * K_LANE_ELEMS + kc * K_VEC_SIZE + ke;
                            recurrent_state_table[base + k_idx * V_HEAD_DIM] = (INPUT3_TYPE)(tmp[ke]);
                        }
                    }
#else
                    for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                        const int k_idx = lid + ks * SUBGROUP_SIZE;
                        if (k_idx < K_HEAD_DIM) {
                            recurrent_state_table[base + k_idx * V_HEAD_DIM] = (INPUT3_TYPE)(state[v_idx][ks]);
                        }
                    }
#endif
                }
            }
        }
    }
}
