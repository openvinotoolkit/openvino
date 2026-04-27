// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#ifndef FUSE_QK_L2NORM
#    define FUSE_QK_L2NORM 0
#endif

#ifndef Q_L2_NORM_EPS
#    define Q_L2_NORM_EPS 1e-6f
#endif

#ifndef K_L2_NORM_EPS
#    define K_L2_NORM_EPS 1e-6f
#endif

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(sum8)(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

inline void FUNC(normalize_kq_128)(float8* b_k, float8* b_q) {
#if FUSE_QK_L2NORM
    float k_sum = FUNC(sum8)((*b_k) * (*b_k));
    k_sum = sub_group_reduce_add(k_sum);
    const float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
    *b_k *= k_scale;

    float q_sum = FUNC(sum8)((*b_q) * (*b_q));
    q_sum = sub_group_reduce_add(q_sum);
    const float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
    *b_q *= q_scale;
#else
    *b_q *= SCALE_FACTOR;
#endif
}

#ifndef K_VEC_SIZE
#    define K_VEC_SIZE 1
#endif

#if ((K_HEAD_DIM % 16) != 0) || ((V_HEAD_DIM % 16) != 0)
#    error "paged_gated_delta_net_opt requires K_HEAD_DIM and V_HEAD_DIM divisible by 16"
#endif

#define K_LANE_ELEMS (K_HEAD_DIM / SUBGROUP_SIZE)

typedef MAKE_VECTOR_TYPE(float, K_VEC_SIZE) K_VEC_TYPE;
#if (K_VEC_SIZE == 1)
#    define K_VEC_LOAD_Q(ptr, idx)     convert_float(BLOCK_READN(INPUT0_TYPE, 1, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     convert_float(BLOCK_READN(INPUT1_TYPE, 1, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) convert_float(BLOCK_READN(INPUT3_TYPE, 1, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        ((INPUT3_TYPE)(vec))
#    define K_VEC_DOT(a, b)            ((a) * (b))
#    define K_VEC_SUM_SQ(a)            ((a) * (a))
#elif (K_VEC_SIZE == 8)
#    define K_VEC_LOAD_Q(ptr, idx)     convert_float8(BLOCK_READN(INPUT0_TYPE, 8, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     convert_float8(BLOCK_READN(INPUT1_TYPE, 8, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) convert_float8(BLOCK_READN(INPUT3_TYPE, 8, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        CAT(convert_, CAT(INPUT3_TYPE, 8))(vec)
#    define K_VEC_DOT(a, b)            FUNC(sum8)((a) * (b))
#    define K_VEC_SUM_SQ(a)            FUNC(sum8)((a) * (a))
#else
#    define K_VEC_LOAD_Q(ptr, idx)     CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT0_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_LOAD_K(ptr, idx)     CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT1_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_LOAD_STATE(ptr, idx) CAT(convert_float, K_VEC_SIZE)(BLOCK_READN(INPUT3_TYPE, K_VEC_SIZE, (ptr), (idx)))
#    define K_VEC_TO_STATE(vec)        CAT(convert_, CAT(INPUT3_TYPE, K_VEC_SIZE))(vec)
#    define K_VEC_DOT(a, b)            dot((a), (b))
#    define K_VEC_SUM_SQ(a)            dot((a), (a))
#endif

#define K_VEC_COUNT (K_LANE_ELEMS / K_VEC_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(paged_gated_delta_net_opt)
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
    const int past_len = past_lens[seq];
    const int interval = cache_interval[seq];
    const int prev_nums = interval > 0 ? past_len % interval : 0;

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;
    const int q_head_base = (hk + query_head_offset) * q_head_stride;
    const int k_head_base = (hk + key_head_offset) * k_head_stride;
    const int v_head_base = (h + value_head_offset) * v_head_stride;
    const int state_stride = K_HEAD_DIM * V_HEAD_DIM;

    K_VEC_TYPE state[V_BLOCK_SIZE][K_VEC_COUNT];
    K_VEC_TYPE q_norm[K_VEC_COUNT];
    K_VEC_TYPE k_norm[K_VEC_COUNT];

    const int initial_block_id = block_indices[block_begin];
    const int initial_block_base = (initial_block_id * V_HEAD_NUM + h) * state_stride;

#pragma unroll
    for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
        int curr_iv = start_iv + v_idx;
        int base = initial_block_base + curr_iv * K_HEAD_DIM;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
        state[v_idx][0] = K_VEC_LOAD_STATE(recurrent_state_table, base);
#else
#    pragma unroll
        for (int kc = 0; kc < K_VEC_COUNT; kc++) {
            const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
            state[v_idx][kc] = K_VEC_LOAD_STATE(recurrent_state_table, base + k_base);
        }
#endif
    }

    int token = token_begin;
    int slot = 1;
    int tokens_to_next_boundary = interval > 0 ? (prev_nums > 0 ? (interval - prev_nums) : interval) : (token_end - token_begin);
    while (token < token_end) {
        const int chunk_end = min(token + tokens_to_next_boundary, token_end);

        int q_base = token * q_token_stride + q_head_base;
        int k_base = token * k_token_stride + k_head_base;
        int v_base = token * v_token_stride + v_head_base;
        int g_idx = token * V_HEAD_NUM + h;

        for (; token < chunk_end; token++, q_base += q_token_stride, k_base += k_token_stride, v_base += v_token_stride, g_idx += V_HEAD_NUM) {
            float q_sum_local = 0.0f;
            float k_sum_local = 0.0f;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            q_norm[0] = K_VEC_LOAD_Q(query, q_base);
            k_norm[0] = K_VEC_LOAD_K(key, k_base);
            FUNC(normalize_kq_128)(&k_norm[0], &q_norm[0]);
#else
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int offset = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                q_norm[kc] = K_VEC_LOAD_Q(query, q_base + offset);
                k_norm[kc] = K_VEC_LOAD_K(key, k_base + offset);
                q_sum_local += K_VEC_SUM_SQ(q_norm[kc]);
                k_sum_local += K_VEC_SUM_SQ(k_norm[kc]);
            }
#endif

#if !((K_VEC_SIZE == 8) && (K_VEC_COUNT == 1))
            float q_scale = SCALE_FACTOR;
            float k_scale = 1.0f;
#    if FUSE_QK_L2NORM
            const float q_sum = sub_group_reduce_add(q_sum_local);
            const float k_sum = sub_group_reduce_add(k_sum_local);

            q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
            k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
#    endif
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                q_norm[kc] *= q_scale;
                k_norm[kc] *= k_scale;
            }
#endif

            const float b_g = exp(convert_float(gate[g_idx]));
            const float b_beta = convert_float(beta[g_idx]);

            float b_v_block[V_BLOCK_SIZE];
            float h_k_block[V_BLOCK_SIZE];
            float update_block[V_BLOCK_SIZE];
            float out_block[V_BLOCK_SIZE];
#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                const int curr_iv = start_iv + v_idx;
                const int v_base_aligned = v_base + (curr_iv & ~(SUBGROUP_SIZE - 1));
                const int v_lane = curr_iv & (SUBGROUP_SIZE - 1);
                const float v_val = convert_float(BLOCK_READN(INPUT2_TYPE, 1, value, v_base_aligned));
                b_v_block[v_idx] = sub_group_broadcast(v_val, v_lane);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                float h_k_local = 0.0f;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] *= b_g;
                h_k_local = FUNC(sum8)(state[v_idx][0] * k_norm[0]);
#else
#    pragma unroll
                for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                    state[v_idx][kc] *= b_g;
                    h_k_local += K_VEC_DOT(state[v_idx][kc], k_norm[kc]);
                }
#endif
                h_k_block[v_idx] = sub_group_reduce_add(h_k_local);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                update_block[v_idx] = (b_v_block[v_idx] - h_k_block[v_idx]) * b_beta;
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                float out_val_local = 0.0f;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
                state[v_idx][0] = fma(k_norm[0], update_block[v_idx], state[v_idx][0]);
                out_val_local = FUNC(sum8)(state[v_idx][0] * q_norm[0]);
#else
#    pragma unroll
                for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                    state[v_idx][kc] = fma(k_norm[kc], update_block[v_idx], state[v_idx][kc]);
                    out_val_local += K_VEC_DOT(state[v_idx][kc], q_norm[kc]);
                }
#endif
                out_block[v_idx] = sub_group_reduce_add(out_val_local);
            }

#pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                const float out_val = out_block[v_idx];

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
            int base = block_base + curr_iv * K_HEAD_DIM;
#if (K_VEC_SIZE == 8) && (K_VEC_COUNT == 1)
            BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base, K_VEC_TO_STATE(state[v_idx][0]));
#else
#    pragma unroll
            for (int kc = 0; kc < K_VEC_COUNT; kc++) {
                const int k_base = kc * K_VEC_SIZE * SUBGROUP_SIZE;
                BLOCK_WRITEN(INPUT3_TYPE, K_VEC_SIZE, recurrent_state_table, base + k_base, K_VEC_TO_STATE(state[v_idx][kc]));
            }
#endif
        }

        if (interval > 0)
            tokens_to_next_boundary = interval;
    }
}
