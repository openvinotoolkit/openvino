// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#define V_BLOCK_SIZE 4

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(sum8)(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}

#if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
inline void FUNC(prepare_qk)(__global INPUT0_TYPE* q, __global INPUT1_TYPE* k, int q_offset, int k_offset, int lid, float8* b_q, float8* b_k) {
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);

    // Memory coalescing illustration (SUBGROUP_SIZE = 16, SUBGROUP_SIZE * 8 = 128 elements per chunk):
    // +-------+---------------+---------------+     +---------------+
    // | Loop j|  Thread0 (T0) |  Thread1 (T1) | ... | Thread15(T15) | <-- GPU Hardware Coalesced Read Transaction
    // +-------+---------------+---------------+     +---------------+
    // |   0   |  k[0]         |  k[1]         | ... |  k[15]        | == Block Read of size 16 starting at 0
    // |   1   |  k[16]        |  k[17]        | ... |  k[31]        | == Block Read of size 16 starting at 16
    // |   2   |  k[32]        |  k[33]        | ... |  k[47]        |
    // |  ...  |   ...         |   ...         | ... |   ...         |
    // |   7   |  k[112]       |  k[113]       | ... |  k[127]       | == Block Read of size 16 starting at 112
    // +-------+---------------+---------------+     +---------------+

#    if FUSE_QK_L2NORM
    // normalize k and q (l2norm + q scale)
    float k_sum = 0.0f;
#        pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        float8 lane_k = (float8)(0.0f);
#        pragma unroll
        for (int j = 0; j < 8; j++) {
            // Map row elements across the subgroup threads (stride-1) for perfectly coalesced block reads
            int k_idx = c * (SUBGROUP_SIZE * 8) + j * SUBGROUP_SIZE + lid;
            lane_k[j] = convert_float(k[k_offset + k_idx]);
            k_sum += lane_k[j] * lane_k[j];
        }
        b_k[c] = lane_k;
    }
    float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
    for (int c = 0; c < K_CHUNKS; c++)
        b_k[c] *= (float8)(k_scale);

    float q_sum = 0.0f;
#        pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        float8 lane_q = (float8)(0.0f);
#        pragma unroll
        for (int j = 0; j < 8; j++) {
            // Map row elements across the subgroup threads (stride-1) for perfectly coalesced block reads
            int q_idx = c * (SUBGROUP_SIZE * 8) + j * SUBGROUP_SIZE + lid;
            lane_q[j] = convert_float(q[q_offset + q_idx]);
            q_sum += lane_q[j] * lane_q[j];
        }
        b_q[c] = lane_q;
    }
    float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);
    for (int c = 0; c < K_CHUNKS; c++)
        b_q[c] *= (float8)(q_scale);
#    else
#        pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        float8 lane_k = (float8)(0.0f);
        float8 lane_q = (float8)(0.0f);
#        pragma unroll
        for (int j = 0; j < 8; j++) {
            // Map row elements across the subgroup threads (stride-1) for perfectly coalesced block reads
            int idx = c * (SUBGROUP_SIZE * 8) + j * SUBGROUP_SIZE + lid;
            lane_k[j] = convert_float(k[k_offset + idx]);
            lane_q[j] = convert_float(q[q_offset + idx]) * SCALE_FACTOR;
        }
        b_k[c] = lane_k;
        b_q[c] = lane_q;
    }
#    endif
}
#else
inline void FUNC(prepare_qk)(__global INPUT0_TYPE* q, __global INPUT1_TYPE* k, int q_offset, int k_offset, float* k_scale, float* q_scale) {
    *k_scale = 1.0f;
    *q_scale = SCALE_FACTOR;
#    if FUSE_QK_L2NORM
    float k_sum = 0.0f;
    float q_sum = 0.0f;
    for (int k_idx = 0; k_idx < K_HEAD_DIM; k_idx++) {
        float k_val = convert_float(k[k_offset + k_idx]);
        float q_val = convert_float(q[q_offset + k_idx]);
        k_sum += k_val * k_val;
        q_sum += q_val * q_val;
    }
    *k_scale = rsqrt(k_sum + K_L2_NORM_EPS);
    *q_scale = rsqrt(q_sum + Q_L2_NORM_EPS) * SCALE_FACTOR;
#    endif
}
#endif

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gated_delta_net_ref)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* initial_state,
 __global INPUT4_TYPE* g,
 __global INPUT5_TYPE* beta,
 __global OUTPUT_TYPE* output,
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,
#endif
 int seq_len,
 int key_offset,
 int value_offset) {
    const int T_len = seq_len;
    const int H_len = V_HEAD_NUM;
    const int HK_len = K_HEAD_NUM;
    const int K_len = K_HEAD_DIM;
    const int V_len = V_HEAD_DIM;

    const int start_iv = get_group_id(2) * V_BLOCK_SIZE;
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();

    const int Q_T_STRIDE = HK_len * K_len;
    const int K_T_STRIDE = (HK_len + key_offset) * K_len;
    const int V_T_STRIDE = (H_len + value_offset) * V_len;
    const int Q_B_STRIDE = T_len * Q_T_STRIDE;
    const int K_B_STRIDE = T_len * K_T_STRIDE;
    const int V_B_STRIDE = T_len * V_T_STRIDE;
    const int STATE_BASE = (b * H_len + h) * (K_len * V_len);
    const int group_size = H_len / HK_len;
    const int hk = h / group_size;
    const int out_bh_base = b * T_len * H_len * V_len + h * V_len;

#if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
    float8 h_state[V_BLOCK_SIZE][K_CHUNKS];
#else
    float h_state_f[V_BLOCK_SIZE][K_HEAD_DIM];
#endif

#if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
// 1. LOAD STATE BLOCK: 4 V-columns, 8 K-rows per lane
// Memory layout mapping across subgroup (e.g. SUBGROUP_SIZE=16, j=0..7):
// Thread 0 reads indices: 0, 16, 32, ..., 112
// Thread 1 reads indices: 1, 17, 33, ..., 113
// ...
// Result: Each tick `j`, the subgroup performs a perfect sequential read [0..15], [16..31], etc.
#    pragma unroll
    for (int row_chunk = 0; row_chunk < K_CHUNKS; row_chunk++) {
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            float8 lane_state = (float8)(0.0f);
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = row_chunk * (SUBGROUP_SIZE * 8) + j * SUBGROUP_SIZE + lid;
                lane_state[j] = convert_float(initial_state[STATE_BASE + row_idx * V_len + curr_iv]);
            }
            h_state[v_idx][row_chunk] = lane_state;
        }
    }
#else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                h_state_f[v_idx][k_idx] =
                    (curr_iv < V_len) ? convert_float(initial_state[STATE_BASE + k_idx * V_len + curr_iv]) : 0.0f;
            }
        }
    }
#endif

    for (int t = 0; t < T_len; t++) {
#if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
        // 2. LOAD COMMON TIMESTEP DATA
        int g_idx = (b * T_len + t) * H_len + h;
        float b_g = exp(convert_float(g[g_idx]));
        float b_beta = convert_float(beta[g_idx]);

        int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + hk * K_len;
        int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
        int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
        int out_offset = out_bh_base + t * H_len * V_len;
        float8 b_k[K_CHUNKS];
        float8 b_q[K_CHUNKS];

        FUNC(prepare_qk)(q, k, q_offset, k_offset, lid, b_q, b_k);

        // V load
        float4 b_v_vec = (float4)(0.0f);
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
        }

        // 3. RECURRENT UPDATE
        float4 dot_part_k_vec = (float4)(0.0f);
        float4 dot_part_q_vec = (float4)(0.0f);

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            float dot_part_k = 0.0f;
#    pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                // Loop fusion: Combine applying the gating scalar and computing the k dot product
                h_state[v_idx][c] *= (float8)(b_g);
                dot_part_k += FUNC(sum8)(h_state[v_idx][c] * b_k[c]);
            }
            dot_part_k_vec[v_idx] = dot_part_k;
        }

        float4 h_k_vec = (float4)(sub_group_reduce_add(dot_part_k_vec.s0),
                      sub_group_reduce_add(dot_part_k_vec.s1),
                      sub_group_reduce_add(dot_part_k_vec.s2),
                      sub_group_reduce_add(dot_part_k_vec.s3));

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            float update_val = (b_v_vec[v_idx] - h_k_vec[v_idx]) * b_beta;
            float dot_part_q = 0.0f;
#    pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                // Loop fusion: use FMA to compute recurrent state update while simultaneously computing q dot product
                h_state[v_idx][c] = fma(b_k[c], (float8)(update_val), h_state[v_idx][c]);
                dot_part_q += FUNC(sum8)(h_state[v_idx][c] * b_q[c]);
            }
            dot_part_q_vec[v_idx] = dot_part_q;
        }

        float4 b_output_vec = (float4)(sub_group_reduce_add(dot_part_q_vec.s0),
                           sub_group_reduce_add(dot_part_q_vec.s1),
                           sub_group_reduce_add(dot_part_q_vec.s2),
                           sub_group_reduce_add(dot_part_q_vec.s3));

        if (lid == 0) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                output[out_offset + curr_iv] = TO_OUTPUT_TYPE(b_output_vec[v_idx]);
            }
        }
#else
        if (lid == 0) {
            int g_idx = (b * T_len + t) * H_len + h;
            float b_g = exp(convert_float(g[g_idx]));
            float b_beta = convert_float(beta[g_idx]);

            int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + hk * K_len;
            int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
            int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
            int out_offset = out_bh_base + t * H_len * V_len;

            float k_scale = 1.0f;
            float q_scale = SCALE_FACTOR;
            FUNC(prepare_qk)(q, k, q_offset, k_offset, &k_scale, &q_scale);

            float k_norm[K_HEAD_DIM];
            float q_norm[K_HEAD_DIM];
            for (int k_idx = 0; k_idx < K_len; k_idx++) {
                k_norm[k_idx] = convert_float(k[k_offset + k_idx]) * k_scale;
                q_norm[k_idx] = convert_float(q[q_offset + k_idx]) * q_scale;
            }

            float4 b_v_vec = (float4)(0.0f);
#    pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                if (curr_iv < V_len)
                    b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
            }

            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_len)
                    continue;

                float h_k = 0.0f;
                for (int k_idx = 0; k_idx < K_len; k_idx++) {
                    h_state_f[v_idx][k_idx] *= b_g;
                    h_k = fma(h_state_f[v_idx][k_idx], k_norm[k_idx], h_k);
                }

                float update_val = (b_v_vec[v_idx] - h_k) * b_beta;

                float b_output = 0.0f;
                for (int k_idx = 0; k_idx < K_len; k_idx++) {
                    h_state_f[v_idx][k_idx] = fma(k_norm[k_idx], update_val, h_state_f[v_idx][k_idx]);
                    b_output = fma(h_state_f[v_idx][k_idx], q_norm[k_idx], b_output);
                }

                output[out_offset + curr_iv] = TO_OUTPUT_TYPE(b_output);
            }
        }
#endif
    }

#if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
// 4. WRITE BACK STATE BLOCK
// Writes back using the exact same coalesced mapped layout as LOAD STATE BLOCK
#    pragma unroll
    for (int row_chunk = 0; row_chunk < K_CHUNKS; row_chunk++) {
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                // Map state mapping to match coalesced block layout (stride-1 access)
                int row_idx = row_chunk * (SUBGROUP_SIZE * 8) + j * SUBGROUP_SIZE + lid;
#    if OUTPUT_STATE
                output_state[STATE_BASE + row_idx * V_len + curr_iv] = (OUTPUT1_TYPE)(h_state[v_idx][row_chunk][j]);
#    else
                initial_state[STATE_BASE + row_idx * V_len + curr_iv] = (INPUT3_TYPE)(h_state[v_idx][row_chunk][j]);
#    endif
            }
        }
    }
#else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                if (curr_iv >= V_len)
                    continue;
#    if OUTPUT_STATE
                output_state[STATE_BASE + k_idx * V_len + curr_iv] = (OUTPUT1_TYPE)(h_state_f[v_idx][k_idx]);
#    else
                initial_state[STATE_BASE + k_idx * V_len + curr_iv] = (INPUT3_TYPE)(h_state_f[v_idx][k_idx]);
#    endif
            }
        }
    }
#endif
}