// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"

// GatedDeltaNet (stateful, non-paged) kernel for single-token decode (seq_len == 1, chosen on the host),
// recurrent state layout [B, H, K, V] (V innermost). Longer sequences use gated_delta_net_ref.cl, which keeps the
// state in registers across tokens: this kernel streams the state twice per token, so it only wins for one token.
//
// One sub-group (= one work-group) per (batch, v-head, tile of SUBGROUP_SIZE V-columns); lane -> one V column.
// The state is streamed, not kept in registers:
//   pass 1: h_k = exp(g) * sum_k S[k][v] * k[k]           (coalesced row reads, no cross-lane reduction needed)
//   pass 2: S'[k][v] = exp(g) * S[k][v] + k[k] * u[v],  u = (v - h_k) * beta
//           o[v] = sum_k S'[k][v] * q[k]                  (S' written back, second read hits the cache)
// No SLM, no barriers, and every sub-group is independent, so a decode step is not limited by the sequential latency
// of the reference kernel (scalar loads strided by V, 2 sub-group reductions per V column per token).
// q/k are read once per token, one element per lane per 16-row chunk, and broadcast with sub_group_broadcast.
// Math is identical to gated_delta_net_ref.cl (fp32 accumulation); one partial sum per 16-row chunk keeps the
// FMA dependency chains short.

#define K_CHUNKS (K_HEAD_DIM / SUBGROUP_SIZE)

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
__attribute__((reqd_work_group_size(1, 1, SUBGROUP_SIZE)))
KERNEL(gated_delta_net_opt)
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
 int query_offset,
 int key_offset,
 int value_offset,
 int q_t_stride,
 int k_t_stride,
 int v_t_stride) {
    const int T_len = seq_len;
    const int H_len = V_HEAD_NUM;
    const int b = get_group_id(0) / V_HEAD_NUM;
    const int h = get_group_id(0) % V_HEAD_NUM;
    const int lane = get_sub_group_local_id();
    const int iv = get_group_id(1) * SUBGROUP_SIZE + lane;

    const int group_size = V_HEAD_NUM / K_HEAD_NUM;
    const int hk = h / group_size;

    const int Q_B_STRIDE = T_len * q_t_stride;
    const int K_B_STRIDE = T_len * k_t_stride;
    const int V_B_STRIDE = T_len * v_t_stride;
    const int STATE_BASE = (b * H_len + h) * (K_HEAD_DIM * V_HEAD_DIM) + iv;
    const int out_bh_base = b * T_len * H_len * V_HEAD_DIM + h * V_HEAD_DIM + iv;

#if OUTPUT_STATE
    __global OUTPUT1_TYPE* dst = output_state + STATE_BASE;
    const __global INPUT3_TYPE* src0 = initial_state + STATE_BASE;
    // Token t > 0 continues from the state written by token t - 1 (same work-item, so the writes are visible).
#    define LOAD_STATE(row) (t == 0 ? convert_float(src0[(row) * V_HEAD_DIM]) : convert_float(dst[(row) * V_HEAD_DIM]))
#    define STORE_STATE(row, val) dst[(row) * V_HEAD_DIM] = (OUTPUT1_TYPE)(val)
#else
    __global INPUT3_TYPE* dst = initial_state + STATE_BASE;
#    define LOAD_STATE(row) convert_float(dst[(row) * V_HEAD_DIM])
#    define STORE_STATE(row, val) dst[(row) * V_HEAD_DIM] = (INPUT3_TYPE)(val)
#endif

    for (int t = 0; t < T_len; t++) {
        const int g_idx = (b * T_len + t) * H_len + h;
        const float b_g = exp(convert_float(g[g_idx]));
        const float b_beta = convert_float(beta[g_idx]);

        const int q_offset = b * Q_B_STRIDE + t * q_t_stride + (hk + query_offset) * K_HEAD_DIM;
        const int k_offset = b * K_B_STRIDE + t * k_t_stride + (hk + key_offset) * K_HEAD_DIM;
        const int v_offset = b * V_B_STRIDE + t * v_t_stride + (h + value_offset) * V_HEAD_DIM;

        // q/k: element (c * SUBGROUP_SIZE + lane) is held by this lane.
        float k_l[K_CHUNKS];
        float q_l[K_CHUNKS];
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            k_l[c] = convert_float(k[k_offset + c * SUBGROUP_SIZE + lane]);
            q_l[c] = convert_float(q[q_offset + c * SUBGROUP_SIZE + lane]);
            k_sum = fma(k_l[c], k_l[c], k_sum);
            q_sum = fma(q_l[c], q_l[c], q_sum);
        }
#if FUSE_QK_L2NORM
        const float k_scale = rsqrt(sub_group_reduce_add(k_sum) + K_L2_NORM_EPS);
        const float q_scale = rsqrt(sub_group_reduce_add(q_sum) + Q_L2_NORM_EPS) * SCALE_FACTOR;
#else
        const float k_scale = 1.0f;
        const float q_scale = SCALE_FACTOR;
#endif
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            k_l[c] *= k_scale;
            q_l[c] *= q_scale;
        }
        const float b_v = convert_float(v[v_offset + iv]);

        // pass 1: h_k = exp(g) * S^T k
        float hk_c[K_CHUNKS];
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            float acc = 0.0f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < SUBGROUP_SIZE; i++) {
                acc = fma(LOAD_STATE(c * SUBGROUP_SIZE + i), sub_group_broadcast(k_l[c], i), acc);
            }
            hk_c[c] = acc;
        }
        float h_k = 0.0f;
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            h_k += hk_c[c];
        }
        h_k *= b_g;
        const float update_val = (b_v - h_k) * b_beta;

        // pass 2: S' = exp(g) * S + k u^T,  o = S'^T q
        float o_c[K_CHUNKS];
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            float acc = 0.0f;
            __attribute__((opencl_unroll_hint))
            for (int i = 0; i < SUBGROUP_SIZE; i++) {
                const int row = c * SUBGROUP_SIZE + i;
                const float s = fma(sub_group_broadcast(k_l[c], i), update_val, LOAD_STATE(row) * b_g);
                STORE_STATE(row, s);
                acc = fma(s, sub_group_broadcast(q_l[c], i), acc);
            }
            o_c[c] = acc;
        }
        float o = 0.0f;
        __attribute__((opencl_unroll_hint))
        for (int c = 0; c < K_CHUNKS; c++) {
            o += o_c[c];
        }
        output[out_bh_base + t * H_len * V_HEAD_DIM] = TO_OUTPUT_TYPE(o);
    }
#undef LOAD_STATE
#undef STORE_STATE
}

#undef K_CHUNKS
