// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cm/cm.h>
#include <cm/cmtl.h>

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);

// load and convert
template <typename DST_TYPE, typename SRC_TYPE, int N, typename std::enable_if<!std::is_same<SRC_TYPE, DST_TYPE>::value, bool>::type = true>
CM_INLINE void cm_load_by_row(vector_ref<DST_TYPE, N> out, SurfaceIndex base, uint offset) {
    static_assert(std::is_same<SRC_TYPE, half>::value);
    if constexpr (std::is_same<SRC_TYPE, half>::value) {
        if constexpr (N >= 2) {
            out = vector<half, N>(cm_load<uint, N / 2>(base, offset).format<half>());
        } else {
            out = vector<half, 1>(cm_load<uint, 1>(base, offset).format<half>().select<1, 1>(0));
        }
    }
}
// load
template <typename DST_TYPE, typename SRC_TYPE, int N, typename std::enable_if<std::is_same<SRC_TYPE, DST_TYPE>::value, bool>::type = true>
CM_INLINE void cm_load_by_row(vector_ref<DST_TYPE, N> out, SurfaceIndex base, uint offset) {
    constexpr int multiplier = std::is_same<SRC_TYPE, half>::value ? 2 : 1;
    // unified the total numbers in terms of uint other than original type.
    constexpr int unified_N = N / multiplier;
    if constexpr (unified_N == 128) {
        out.select<unified_N / 2, 1>(0).format<uint>() = cm_load<uint, 64>(base, offset);
        out.select<unified_N / 2, 1>(unified_N / 2).format<uint>() = cm_load<uint, 64>(base, offset + unified_N / 2 * sizeof(uint));
    } else if constexpr (unified_N <= 64) {
        out.format<uint>() = cm_load<uint, unified_N>(base, offset);
    }
}

// store
template <typename DST_TYPE, typename SRC_TYPE, int N, typename std::enable_if<std::is_same<SRC_TYPE, DST_TYPE>::value, bool>::type = true>
CM_INLINE void cm_store_by_row(SurfaceIndex base, vector_ref<SRC_TYPE, N> data, uint offset) {
    constexpr int multiplier = std::is_same<SRC_TYPE, half>::value ? 2 : 1;
    // unified the total numbers in terms of uint other than original type.
    constexpr int unified_N = N / multiplier;
    if constexpr (unified_N == 128) {
        cm_store<uint, 64>(base, offset, data.format<uint>().select<64, 1>(0));
        cm_store<uint, 64>(base, offset + 64 * sizeof(uint), data.format<uint>().select<64, 1>(unified_N / 2));
    } else if constexpr (unified_N <= 64) {
        cm_store<uint, unified_N>(base, offset, data.format<uint>().select<unified_N, 1>(0));
    }
}

// convert and store
template <typename DST_TYPE, typename SRC_TYPE, int N, typename std::enable_if<!std::is_same<SRC_TYPE, DST_TYPE>::value, bool>::type = true>
CM_INLINE void cm_store_by_row(SurfaceIndex base, vector_ref<SRC_TYPE, N> data, uint offset) {
    static_assert(std::is_same<DST_TYPE, half>::value);
    if constexpr (std::is_same<DST_TYPE, half>::value) {
        vector<half, N> temp(data);
        if constexpr (N >= 2) {
            cm_store<uint, N / 2>(base, offset, temp.format<uint>());
        } else {
            write(base, offset / sizeof(half), 0, half(data[0]));
        }
    }
}

template <int N>
CM_INLINE void cm_prefetch_by_row(SurfaceIndex base, uint offset) {
    if constexpr (N == 128) {
        cm_prefetch<64, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(base, offset);
        cm_prefetch<64, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(base, offset);
    } else if constexpr (N <= 64) {
        cm_prefetch<N, DataSize::U32, CacheHint::Cached, CacheHint::Cached>(base, offset);
    }
}

template <typename IN_OUT_DTYPE,
          int k_num_heads,
          int v_num_heads,
          int k_head_dims,
          int v_head_dims,
          bool use_qk_l2norm,
          int PRE_FETCH_DPT = 0,
          int PRE_FETCH_CNT = 1>
void recurrent_linear_attn(int b_idx,
                           int head_idx,
                           int head_dim_t_idx,
                           int seq,
                           int key_offset,
                           int value_offset,
                           SurfaceIndex q [[type("buffer_t")]],
                           SurfaceIndex k [[type("buffer_t")]],
                           SurfaceIndex v [[type("buffer_t")]],
                           SurfaceIndex g [[type("buffer_t")]],
                           SurfaceIndex beta [[type("buffer_t")]],
                           SurfaceIndex initial_state [[type("buffer_t")]],
                           SurfaceIndex output [[type("buffer_t")]],
                           SurfaceIndex output_state [[type("buffer_t")]]) {
#ifdef CM_HAS_LSC_UNTYPED_2D
    // xe 2 has large grf
    constexpr int v_head_dim_per_t = v_head_dims / 8;  // 16
#else
    constexpr int v_head_dim_per_t = v_head_dims / 16;  // 8
#endif
    vector<float, v_head_dim_per_t * k_head_dims> h0;
// h0 [B, H, V, K]
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * v_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims + v_head_dim_idx * k_head_dims;
        if constexpr (std::is_same<IN_OUT_DTYPE, float>::value && k_head_dims == 128) {
            auto h0_row = h0.select<k_head_dims, 1>(k_head_dims * i);
            cm_load_by_row<float, IN_OUT_DTYPE, 64>(h0_row.select<64, 1>(0), initial_state, stride * sizeof(IN_OUT_DTYPE));
            cm_load_by_row<float, IN_OUT_DTYPE, 64>(h0_row.select<64, 1>(64), initial_state, (stride + 64) * sizeof(IN_OUT_DTYPE));
        } else {
            cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(h0.select<k_head_dims, 1>(k_head_dims * i), initial_state, stride * sizeof(IN_OUT_DTYPE));
        }
    }

    const int group_size = v_num_heads / k_num_heads;
    const int qk_head_idx = head_idx / group_size;
    for (int s = 0; s < seq; s++) {
        // beta B, T, H
        // g B, T, H
        int stride = b_idx * seq * v_num_heads + s * v_num_heads + head_idx;
        vector<float, 1> b_beta;  // cm_load<float, 1>(beta, stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, 1>(b_beta, beta, stride * sizeof(IN_OUT_DTYPE));
        vector<float, 1> b_g;  // cm_load<float, 1>(g, stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, 1>(b_g, g, stride * sizeof(IN_OUT_DTYPE));
        // B, T, HK, K
        int q_stride = b_idx * seq * k_num_heads * k_head_dims + s * k_num_heads * k_head_dims + qk_head_idx * k_head_dims;
        int k_stride = b_idx * seq * k_num_heads * k_head_dims + s * (k_num_heads + key_offset) * k_head_dims + (qk_head_idx + key_offset) * k_head_dims;
        if constexpr (PRE_FETCH_DPT > 0) {
            if ((s % PRE_FETCH_CNT == 0) && head_dim_t_idx < PRE_FETCH_CNT) {
                const int q_stride =
                    (b_idx * seq * k_num_heads * k_head_dims + (s + PRE_FETCH_DPT + head_dim_t_idx) * k_num_heads * k_head_dims + qk_head_idx * k_head_dims) *
                    sizeof(IN_OUT_DTYPE);
                const int k_stride =
                    (b_idx * seq * k_num_heads * k_head_dims + (s + PRE_FETCH_DPT + head_dim_t_idx) * (k_num_heads + key_offset) * k_head_dims + (qk_head_idx + key_offset) * k_head_dims) *
                    sizeof(IN_OUT_DTYPE);
                const int v_stride =
                    (b_idx * seq * v_num_heads * v_head_dims + (s + PRE_FETCH_DPT + head_dim_t_idx) * (v_num_heads + value_offset) * v_head_dims + (head_idx + value_offset) * v_head_dims) *
                    sizeof(IN_OUT_DTYPE);
                cm_prefetch_by_row<k_head_dims>(q, q_stride);
                cm_prefetch_by_row<k_head_dims>(k, k_stride);
                cm_prefetch_by_row<k_head_dims>(v, v_stride);
            }
        }
        // read q k
        vector<float, k_head_dims> b_q;  // cm_load<float, k_head_dims>(q, qk_stride * 4);
        cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(b_q, q, q_stride * sizeof(IN_OUT_DTYPE));
        vector<float, k_head_dims> b_k;  // cm_load<float, k_head_dims>(k, qk_stride * 4);
        cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(b_k, k, k_stride * sizeof(IN_OUT_DTYPE));
        if constexpr (use_qk_l2norm) {
            float eps = 0.000001;
            float q_sum = cm_sum<float>(b_q * b_q);
            b_q = b_q * cm_rsqrt(q_sum + eps);
            float k_sum = cm_sum<float>(b_k * b_k);
            b_k = b_k * cm_rsqrt(k_sum + eps);
        }
        // read_v
        // B, T, HV, V
        int v_stride = b_idx * seq * v_num_heads * v_head_dims + s * (v_num_heads + value_offset) * v_head_dims + (head_idx + value_offset) * v_head_dims + head_dim_t_idx * v_head_dim_per_t;// + VALUE_OFFSET * v_head_dims;
        vector<float, v_head_dim_per_t> b_v;  // cm_load<uint, v_head_dim_per_t>(v, v_stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, v_head_dim_per_t>(b_v, v, v_stride * sizeof(IN_OUT_DTYPE));
        if constexpr (PRE_FETCH_DPT > 0) {
            if (s % 8 == 7) {
                cm_fence(CM_LOCAL_BARRIER);
            }
        } else {
            cm_barrier();
        }

        vector<float, v_head_dim_per_t> cur_output;
        vector<float, v_head_dim_per_t> h_k;
        constexpr float log2e = 1.4426950408889634f;
        b_q *= SCALE_FACTOR;
        float g_cur = cm_exp(b_g[0] * log2e);
#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) * g_cur;
            h_k[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_k);
        }
        vector<float, v_head_dim_per_t> delta_full = (b_v - h_k) * b_beta[0];

#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            float detla = delta_full[i];
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) + b_k * detla;
            cur_output[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_q);
        }
        // B, T, HV, V
        int output_stride =
            b_idx * seq * v_num_heads * v_head_dims + s * v_num_heads * v_head_dims + head_idx * v_head_dims + head_dim_t_idx * v_head_dim_per_t;
        cm_store_by_row<IN_OUT_DTYPE, float, v_head_dim_per_t>(output, cur_output, output_stride * sizeof(IN_OUT_DTYPE));
    }
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * v_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims + v_head_dim_idx * k_head_dims;
        if constexpr (std::is_same<IN_OUT_DTYPE, float>::value && k_head_dims == 128) {
            auto h0_row = h0.select<k_head_dims, 1>(k_head_dims * i);
            cm_store_by_row<IN_OUT_DTYPE, float, 64>(output_state, h0_row.select<64, 1>(0), stride * sizeof(IN_OUT_DTYPE));
            cm_store_by_row<IN_OUT_DTYPE, float, 64>(output_state, h0_row.select<64, 1>(64), (stride + 64) * sizeof(IN_OUT_DTYPE));
            cm_store_by_row<IN_OUT_DTYPE, float, 64>(initial_state, h0_row.select<64, 1>(0), stride * sizeof(IN_OUT_DTYPE));
            cm_store_by_row<IN_OUT_DTYPE, float, 64>(initial_state, h0_row.select<64, 1>(64), (stride + 64) * sizeof(IN_OUT_DTYPE));
        } else {
            cm_store_by_row<IN_OUT_DTYPE, float, k_head_dims>(output_state, h0.select<k_head_dims, 1>(k_head_dims * i), stride * sizeof(IN_OUT_DTYPE));
            cm_store_by_row<IN_OUT_DTYPE, float, k_head_dims>(initial_state, h0.select<k_head_dims, 1>(k_head_dims * i), stride * sizeof(IN_OUT_DTYPE));
        }
    }
}

// used for inplaced update states
template <typename IN_OUT_DTYPE,
          int k_num_heads,
          int v_num_heads,
          int k_head_dims,
          int v_head_dims,
          bool use_qk_l2norm,
          int PRE_FETCH_DPT = 0,
          int PRE_FETCH_CNT = 1>
void recurrent_linear_attn(int b_idx,
                           int head_idx,
                           int head_dim_t_idx,
                           int seq,
                           int key_offset,
                           int value_offset,
                           SurfaceIndex q [[type("buffer_t")]],
                           SurfaceIndex k [[type("buffer_t")]],
                           SurfaceIndex v [[type("buffer_t")]],
                           SurfaceIndex g [[type("buffer_t")]],
                           SurfaceIndex beta [[type("buffer_t")]],
                           SurfaceIndex initial_state [[type("buffer_t")]],
                           SurfaceIndex output [[type("buffer_t")]]) {
    recurrent_linear_attn<IN_OUT_DTYPE, k_num_heads, v_num_heads, k_head_dims, v_head_dims, use_qk_l2norm, PRE_FETCH_DPT, PRE_FETCH_CNT>(b_idx,
                                                                                                                                         head_idx,
                                                                                                                                         head_dim_t_idx,
                                                                                                                                         seq,
                                                                                                                                         key_offset,
                                                                                                                                         value_offset,
                                                                                                                                         q,
                                                                                                                                         k,
                                                                                                                                         v,
                                                                                                                                         g,
                                                                                                                                         beta,
                                                                                                                                         initial_state,
                                                                                                                                         output,
                                                                                                                                         initial_state);
}