// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(sum8)(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

inline void FUNC(normalize_kq_128)(float8* b_k, float8* b_q) {
    float k_sum = FUNC(sum8)((*b_k) * (*b_k));
    k_sum = sub_group_reduce_add(k_sum);
    k_sum = sub_group_broadcast(k_sum, 0);
    const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, 1e-6f);
    *b_k *= k_scale;

    float q_sum = FUNC(sum8)((*b_q) * (*b_q));
    q_sum = sub_group_reduce_add(q_sum);
    q_sum = sub_group_broadcast(q_sum, 0);
    const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, 1e-6f);
    *b_q *= q_scale;
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
#define K_VEC_LOAD_Q(ptr, idx) CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT0_TYPE, K_VEC_SIZE, (ptr), (idx)))
#define K_VEC_LOAD_K(ptr, idx) CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT1_TYPE, K_VEC_SIZE, (ptr), (idx)))
#define K_VEC_LOAD_STATE(ptr, idx) CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT3_TYPE, K_VEC_SIZE, (ptr), (idx)))
#define K_VEC_TO_STATE(vec) CAT(convert_, CAT(INPUT3_TYPE, K_VEC_SIZE))(vec)
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
    const int interval = cache_interval[seq];

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const int q_head_base = (hk + query_head_offset) * q_head_stride;
    const int k_head_base = (hk + key_head_offset) * k_head_stride;
    const int v_head_base = (h + value_head_offset) * v_head_stride;
    const int state_stride = K_HEAD_DIM * V_HEAD_DIM;

#if KV_OPT_VEC_PATH
    K_VEC_TYPE state[V_BLOCK_SIZE][K_VEC_COUNT];
    K_VEC_TYPE q_norm[K_VEC_COUNT];
    K_VEC_TYPE k_norm[K_VEC_COUNT];
#else
    float state[V_BLOCK_SIZE][K_SLICE_SIZE];
    float q_norm[K_SLICE_SIZE];
    float k_norm[K_SLICE_SIZE];
#endif

    const int initial_block_id = block_indices[block_begin];
    const int initial_block_base = (initial_block_id * V_HEAD_NUM + h) * state_stride;

#pragma unroll
    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
        if (curr_iv >= V_HEAD_DIM)
            continue;
#endif

        {
            int base = initial_block_base + curr_iv * K_HEAD_DIM;
#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            state[v_idx][0] = K_VEC_LOAD_STATE(recurrent_state_table, base);
#elif KV_OPT_VEC_PATH
#pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                state[v_idx][kc] = K_VEC_LOAD_STATE(recurrent_state_table, base + k_base);
            }
#else
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                const int k_idx = lid + ks * SUBGROUP_SIZE;
                if (k_idx < K_HEAD_DIM) {
                    state[v_idx][ks] = convert_float(recurrent_state_table[base + k_idx]);
                }
            }
#endif
        }
    }

    int token = token_begin;
    int slot = 1;
    while (token < token_end) {
        const int chunk_end = interval > 0 ? min(token + interval, token_end) : token_end;

        int q_base = token * q_token_stride + q_head_base;
        int k_base = token * k_token_stride + k_head_base;
        int v_base = token * v_token_stride + v_head_base;
        int g_idx = token * V_HEAD_NUM + h;

        for (; token < chunk_end; token++, q_base += q_token_stride, k_base += k_token_stride, v_base += v_token_stride, g_idx += V_HEAD_NUM) {

            float q_sum_local = 0.0f;
            float k_sum_local = 0.0f;
#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            q_norm[0] = K_VEC_LOAD_Q(query, q_base);
            k_norm[0] = K_VEC_LOAD_K(key, k_base);
            FUNC(normalize_kq_128)(&k_norm[0], &q_norm[0]);
#elif KV_OPT_VEC_PATH
#pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int offset = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                q_norm[kc] = K_VEC_LOAD_Q(query, q_base + offset);
                k_norm[kc] = K_VEC_LOAD_K(key, k_base + offset);
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

#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            // already normalized by FUNC(normalize_kq_128)
#else
            float q_sum = sub_group_reduce_add(q_sum_local);
            q_sum = sub_group_broadcast(q_sum, 0);
            float k_sum = sub_group_reduce_add(k_sum_local);
            k_sum = sub_group_broadcast(k_sum, 0);

            const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, 1e-6f);
            const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, 1e-6f);
#if KV_OPT_VEC_PATH
#pragma unroll
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
#endif

            const float b_g = exp(convert_float(gate[g_idx]));
            const float b_beta = convert_float(beta[g_idx]);

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
                if (curr_iv >= V_HEAD_DIM)
                    continue;
#endif
                float h_k_local = 0.0f;
#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] *= b_g;
                h_k_local = FUNC(sum8)(state[v_idx][0] * k_norm[0]);
#elif KV_OPT_VEC_PATH
#pragma unroll
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

#if KV_OPT_VEC_PATH
                const int v_base_aligned = v_base + (curr_iv & ~(SUBGROUP_SIZE - 1));
                const int v_lane = curr_iv & (SUBGROUP_SIZE - 1);
                const float v_val = convert_float(BLOCK_READN(INPUT2_TYPE, 1, value, v_base_aligned));
                const float b_v = sub_group_broadcast(v_val, v_lane);
#else
                const int v_idx_offset = v_base + curr_iv;
                const float b_v = convert_float(value[v_idx_offset]);
#endif
                const float update_val = (b_v - h_k) * b_beta;

                float out_val_local = 0.0f;
#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] = fma(k_norm[0], update_val, state[v_idx][0]);
                out_val_local = FUNC(sum8)(state[v_idx][0] * q_norm[0]);
#elif KV_OPT_VEC_PATH
#pragma unroll
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
        }

        const int block_id = block_indices[block_begin + slot];
        slot++;
        const int block_base = (block_id * V_HEAD_NUM + h) * state_stride;
#pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
#if !KV_OPT_VEC_PATH
            if (curr_iv >= V_HEAD_DIM)
                continue;
#endif

            int base = block_base + curr_iv * K_HEAD_DIM;
#if KV_OPT_VEC_PATH && (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base, K_VEC_TO_STATE(state[v_idx][0]));
#elif KV_OPT_VEC_PATH
#pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base + k_base, K_VEC_TO_STATE(state[v_idx][kc]));
            }
#else
            for (int ks = 0; ks < K_SLICE_SIZE; ks++) {
                const int k_idx = lid + ks * SUBGROUP_SIZE;
                if (k_idx < K_HEAD_DIM) {
                    recurrent_state_table[base + k_idx] = (INPUT3_TYPE)(state[v_idx][ks]);
                }
            }
#endif
        }
    }
}
