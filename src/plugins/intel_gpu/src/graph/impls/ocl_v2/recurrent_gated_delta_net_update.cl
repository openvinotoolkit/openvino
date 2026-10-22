// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"

// Qwen llama.cpp staged gather: capture every original source needed by this op
// before any in-place cache writes. Metadata stays on device and varies each call.
#if INPLACE_STAGE == 0
KERNEL(recurrent_cache_update_stage)(OPTIONAL_SHAPE_INFO_ARG const __global INPUT0_TYPE* q,
                                     const __global INPUT1_TYPE* k,
                                     const __global INPUT2_TYPE* v,
                                     const __global INPUT3_TYPE* old_cache,
                                     const __global INPUT4_TYPE* gate,
                                     const __global INPUT5_TYPE* beta,
                                     const __global INPUT6_TYPE* active_rows,
                                     const __global INPUT7_TYPE* extra_rows,
                                     const __global INPUT8_TYPE* reset_begin,
                                     const __global INPUT9_TYPE* reset_length,
                                     const __global INPUT10_TYPE* offsets,
                                     const __global INPUT11_TYPE* control,
                                     __global OUTPUT_TYPE* output,
                                     __global OUTPUT1_TYPE* cache,
                                     __global float* staged,
                                     int inplace) {
    const size_t i = get_global_id(0);
    const int F = INPUT3_SIZE_X, S = INPUT3_SIZE_Y, B = INPUT0_BATCH_NUM;
    const int rb = reset_begin[INPUT8_OFFSET], re = rb + reset_length[INPUT9_OFFSET];
    const int es = offsets[INPUT10_GET_INDEX(0, 0, 0, 1)], eb = offsets[INPUT10_OFFSET], E = INPUT7_SIZE_X - es;
#    define OLD(r, f) old_cache[INPUT3_GET_INDEX(0, 0, r, f)]
#    define ROWA(b)   active_rows[INPUT6_GET_INDEX(0, 0, 0, b)]
#    define ROWE(e)   extra_rows[INPUT7_GET_INDEX(0, 0, 0, e + es)]
#    define ROUND(v)  convert_float((INPUT0_TYPE)(v))
    if (i >= (size_t)(B + E) * F)
        return;
    const int row = i / F, f = i % F, src = row < B ? ROWA(row) : ROWE(row - B);
    // Identity suffix rows are read by their destination owner in the compute
    // stage. They need no staging; all nonidentity sources are captured first.
    if (row >= B && src == eb + row - B)
        return;
    staged[i] = src >= rb && src < re ? 0.0f : ROUND(OLD(src, f));
#    undef OLD
#    undef ROWA
#    undef ROWE
#    undef ROUND
}
#else
#    include "include/batch_headers/sub_group_block_read.cl"
#    include "include/batch_headers/sub_group_block_write.cl"
#    include "include/batch_headers/sub_group_shuffle.cl"

inline float FUNC(l2norm_scale)(float sum, float extra_scale, float eps) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + eps) * extra_scale;
}

inline float FUNC(dot8_fma)(float8 a, float8 b) {
    float acc = a.s0 * b.s0;
    acc = fma(a.s1, b.s1, acc);
    acc = fma(a.s2, b.s2, acc);
    acc = fma(a.s3, b.s3, acc);
    acc = fma(a.s4, b.s4, acc);
    acc = fma(a.s5, b.s5, acc);
    acc = fma(a.s6, b.s6, acc);
    acc = fma(a.s7, b.s7, acc);
    return acc;
}

#    if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
inline float8 FUNC(load_q8_as_float8)(const __global INPUT0_TYPE* p) {
#        if INPUT0_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#        elif INPUT0_TYPE_SIZE == 4
    return convert_float8(vload8(0, (const __global float*)p));
#        else
    float8 v = (float8)(0.0f);
#            pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline float8 FUNC(load_k8_as_float8)(const __global INPUT1_TYPE* p) {
#        if INPUT1_TYPE_SIZE == 2
    return convert_float8(vload8(0, (const __global half*)p));
#        elif INPUT1_TYPE_SIZE == 4
    return convert_float8(vload8(0, (const __global float*)p));
#        else
    float8 v = (float8)(0.0f);
#            pragma unroll
    for (int i = 0; i < 8; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline float4 FUNC(load_v4_as_float4)(const __global INPUT2_TYPE* p) {
#        if INPUT2_TYPE_SIZE == 2
    return convert_float4(vload4(0, (const __global half*)p));
#        elif INPUT2_TYPE_SIZE == 4
    return convert_float4(vload4(0, (const __global float*)p));
#        else
    float4 v = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#            pragma unroll
    for (int i = 0; i < 4; i++)
        v[i] = convert_float(p[i]);
    return v;
#        endif
}

inline void FUNC(
    prepare_qk)(const __global INPUT0_TYPE* q, const __global INPUT1_TYPE* k, int q_offset, int k_offset, int lane_k_base, float8* b_q, float8* b_k) {
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
#        if FUSE_QK_L2NORM
    // normalize k and q (l2norm + q scale)
    float k_sum = 0.0f;
    float q_sum = 0.0f;
#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        int idx = lane_k_base + c * 8;
        float8 lane_k = FUNC(load_k8_as_float8)(k + k_offset + idx);
        float8 lane_q = FUNC(load_q8_as_float8)(q + q_offset + idx);
        b_k[c] = lane_k;
        b_q[c] = lane_q;
        k_sum += FUNC(dot8_fma)(lane_k, lane_k);
        q_sum += FUNC(dot8_fma)(lane_q, lane_q);
    }
    float k_scale = FUNC(l2norm_scale)(k_sum, 1.0f, K_L2_NORM_EPS);
    float q_scale = FUNC(l2norm_scale)(q_sum, SCALE_FACTOR, Q_L2_NORM_EPS);

#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        b_k[c] *= (float8)(k_scale);
        b_q[c] *= (float8)(q_scale);
    }
#        else
#            pragma unroll
    for (int c = 0; c < K_CHUNKS; c++) {
        int idx = lane_k_base + c * 8;
        float8 lane_k = FUNC(load_k8_as_float8)(k + k_offset + idx);
        float8 lane_q = FUNC(load_q8_as_float8)(q + q_offset + idx);
        b_k[c] = lane_k;
        b_q[c] = lane_q * (float8)(SCALE_FACTOR);
    }
#        endif
}
#    else
inline void FUNC(prepare_qk)(__global INPUT0_TYPE* q, __global INPUT1_TYPE* k, int q_offset, int k_offset, float* k_scale, float* q_scale) {
    *k_scale = 1.0f;
    *q_scale = SCALE_FACTOR;
#        if FUSE_QK_L2NORM
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
#        endif
}
#    endif

// A single work item owns each cache element. Later source planes win overlapping writes.
inline void FUNC(write_cache_state)(OPTIONAL_SHAPE_INFO_ARG __global OUTPUT1_TYPE* cache,
                                    const __global INPUT11_TYPE* control,
                                    int b,
                                    int slot,
                                    int B,
                                    int valid,
                                    int h,
                                    int k,
                                    int v,
                                    float value) {
    const int begin = control[INPUT11_GET_INDEX(0, 0, 0, 0)];
    const int sequences = control[INPUT11_GET_INDEX(0, 0, 0, 1)];
    const int stride = control[INPUT11_GET_INDEX(0, 0, 0, 2)];
    const int source = control[INPUT11_GET_INDEX(0, 0, 0, 3)];
    const int r = slot * B + b - source, N = valid * B - source;
    if (r < 0 || r >= N)
        return;
    const int dst = begin + (r / sequences) * stride + r % sequences;
    for (int plane = r / sequences + 1; plane * sequences < N; ++plane) {
        const int offset = dst - begin - plane * stride;
        if (offset >= 0 && offset < sequences && plane * sequences + offset < N)
            return;
    }
    // Match the former materialized state tensor's rounding, followed by cache precision.
    cache[OUTPUT1_GET_INDEX(0, 0, dst, (h * V_HEAD_DIM + v) * K_HEAD_DIM + k)] = (OUTPUT1_TYPE)((INPUT0_TYPE)value);
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(recurrent_gated_delta_net_update)
(OPTIONAL_SHAPE_INFO_ARG __global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* initial_state,
 __global INPUT4_TYPE* g,
 __global INPUT5_TYPE* beta,
 const __global INPUT6_TYPE* active_rows,
 const __global INPUT7_TYPE* extra_rows,
 const __global INPUT8_TYPE* reset_begin,
 const __global INPUT9_TYPE* reset_length,
 const __global INPUT10_TYPE* extra_offsets,
 const __global INPUT11_TYPE* write_control,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* updated_cache,
 int seq_len,
 int query_offset,
 int key_offset,
 int value_offset,
 int q_t_stride,
 int k_t_stride,
 int v_t_stride,
 __global float* staged,
 int inplace) {
    const int T_len = seq_len;
    const int H_len = V_HEAD_NUM;
    const int HK_len = K_HEAD_NUM;
    const int K_len = K_HEAD_DIM;
    const int V_len = V_HEAD_DIM;

    const int start_iv = get_group_id(2) * V_BLOCK_SIZE;
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();

    const int Q_T_STRIDE = q_t_stride;
    const int K_T_STRIDE = k_t_stride;
    const int V_T_STRIDE = v_t_stride;
    const int Q_B_STRIDE = T_len * Q_T_STRIDE;
    const int K_B_STRIDE = T_len * K_T_STRIDE;
    const int V_B_STRIDE = T_len * V_T_STRIDE;
    const int B = INPUT0_BATCH_NUM;
    const int valid = min((int)SNAPSHOT_COUNT, T_len);
    const int rb = reset_begin[INPUT8_OFFSET], re = rb + reset_length[INPUT9_OFFSET];
    if (b >= B) {
        const int dst = b - B;
        const int begin = write_control[INPUT11_GET_INDEX(0, 0, 0, 0)];
        const int sequences = write_control[INPUT11_GET_INDEX(0, 0, 0, 1)];
        const int stride = write_control[INPUT11_GET_INDEX(0, 0, 0, 2)];
        const int source = write_control[INPUT11_GET_INDEX(0, 0, 0, 3)];
        const int N = valid * B - source;
        // Updated rows are written directly by their recurrence owner below.
        for (int plane = 0; plane * sequences < N; ++plane) {
            const int offset = dst - begin - plane * stride;
            if (offset >= 0 && offset < sequences && plane * sequences + offset < N)
                return;
        }
        const int eb = extra_offsets[INPUT10_GET_INDEX(0, 0, 0, 0)];
        const int es = extra_offsets[INPUT10_GET_INDEX(0, 0, 0, 1)];
        int row = dst;
        if (dst >= eb && dst < eb + INPUT7_SIZE_X - es)
            row = extra_rows[INPUT7_GET_INDEX(0, 0, 0, dst - eb + es)];
        for (int k_idx = lid; k_idx < K_len; k_idx += SUBGROUP_SIZE)
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; ++v_idx) {
                const int iv = start_iv + v_idx;
                if (iv < V_len) {
                    const int f = (h * V_len + iv) * K_len + k_idx;
                    float value = inplace && row != dst ? staged[(B + dst - eb) * INPUT3_SIZE_X + f]
                                                        : (row >= rb && row < re ? 0.0f : convert_float(initial_state[INPUT3_GET_INDEX(0, 0, row, f)]));
                    value = convert_float((INPUT0_TYPE)value);
                    if (!inplace || as_uint(value) != as_uint(convert_float(initial_state[INPUT3_GET_INDEX(0, 0, dst, f)])))
                        updated_cache[OUTPUT1_GET_INDEX(0, 0, dst, f)] = (OUTPUT1_TYPE)value;
                }
            }
        return;
    }
    const int source_row = active_rows[INPUT6_GET_INDEX(0, 0, 0, b)];
#    define LOAD_STATE(k, v)                                                          \
        (source_row >= rb && source_row < re                                          \
             ? 0.0f                                                                   \
             : (inplace ? staged[b * INPUT3_SIZE_X + (h * V_len + (v)) * K_len + (k)] \
                        : convert_float((INPUT0_TYPE)initial_state[INPUT3_GET_INDEX(0, 0, source_row, (h * V_len + (v)) * K_len + (k))])))
    const int STATE_BASE = (b * H_len + h) * (K_len * V_len);
    const int group_size = H_len / HK_len;
    const int hk = h / group_size;
    const int out_bh_base = b * T_len * H_len * V_len + h * V_len;

#    if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
    const int K_CHUNKS = K_HEAD_DIM / (SUBGROUP_SIZE * 8);
    float8 h_state[V_BLOCK_SIZE][K_CHUNKS];
#    else
    float h_state_f[V_BLOCK_SIZE][K_HEAD_DIM];
#    endif

#    if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
// 1. LOAD STATE BLOCK: 4 V-columns, 8 K-rows per lane
#        pragma unroll
    for (int row_chunk = 0; row_chunk < K_CHUNKS; row_chunk++) {
#        pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            float8 lane_state = (float8)(0.0f);
#        pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = lid * (K_HEAD_DIM / SUBGROUP_SIZE) + row_chunk * 8 + j;
                lane_state[j] = LOAD_STATE(row_idx, curr_iv);
            }
            h_state[v_idx][row_chunk] = lane_state;
        }
    }
#    else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                h_state_f[v_idx][k_idx] = (curr_iv < V_len) ? LOAD_STATE(k_idx, curr_iv) : 0.0f;
            }
        }
    }
#    endif

    for (int t = 0; t < T_len; t++) {
#    if (K_HEAD_DIM == 128 && (V_HEAD_DIM % V_BLOCK_SIZE == 0))
        // 2. LOAD COMMON TIMESTEP DATA
        int g_idx = (b * T_len + t) * H_len + h;
        float b_g = exp(convert_float(g[g_idx]));
        float b_beta = convert_float(beta[g_idx]);
        const int lane_k_base = lid * (K_HEAD_DIM / SUBGROUP_SIZE);

        int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + (hk + query_offset) * K_len;
        int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
        int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
        int out_offset = out_bh_base + t * H_len * V_len;
        float8 b_k[K_CHUNKS];
        float8 b_q[K_CHUNKS];

        FUNC(prepare_qk)(q, k, q_offset, k_offset, lane_k_base, b_q, b_k);

        // V load
        float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#        if (V_BLOCK_SIZE == 4)
        b_v_vec = FUNC(load_v4_as_float4)(v + v_offset + start_iv);
#        else
#            pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
        }
#        endif

        // 3. RECURRENT UPDATE
        float4 dot_part_k_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
        float4 dot_part_q_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            float dot_part_k = 0.0f;
#        pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                float8 s = h_state[v_idx][c] * (float8)(b_g);
                h_state[v_idx][c] = s;
                dot_part_k += FUNC(dot8_fma)(s, b_k[c]);
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
#        pragma unroll
            for (int c = 0; c < K_CHUNKS; c++) {
                float8 s = fma(b_k[c], (float8)(update_val), h_state[v_idx][c]);
                h_state[v_idx][c] = s;
                if (T_len - 1 - t < SNAPSHOT_COUNT) {
                    for (int j = 0; j < 8; ++j)
                        FUNC(write_cache_state)(OPTIONAL_SHAPE_INFO_TENSOR updated_cache,
                                                write_control,
                                                b,
                                                T_len - 1 - t,
                                                B,
                                                valid,
                                                h,
                                                lane_k_base + c * 8 + j,
                                                start_iv + v_idx,
                                                s[j]);
                }
                dot_part_q += FUNC(dot8_fma)(s, b_q[c]);
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
#    else
        if (lid == 0) {
            int g_idx = (b * T_len + t) * H_len + h;
            float b_g = exp(convert_float(g[g_idx]));
            float b_beta = convert_float(beta[g_idx]);

            int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + (hk + query_offset) * K_len;
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

            float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#        pragma unroll
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
                    h_state_f[v_idx][k_idx] += k_norm[k_idx] * update_val;
                    if (T_len - 1 - t < SNAPSHOT_COUNT)
                        FUNC(write_cache_state)(OPTIONAL_SHAPE_INFO_TENSOR updated_cache,
                                                write_control,
                                                b,
                                                T_len - 1 - t,
                                                B,
                                                valid,
                                                h,
                                                k_idx,
                                                curr_iv,
                                                h_state_f[v_idx][k_idx]);
                    b_output = fma(h_state_f[v_idx][k_idx], q_norm[k_idx], b_output);
                }

                output[out_offset + curr_iv] = TO_OUTPUT_TYPE(b_output);
            }
        }
#    endif
    }
}
#endif
