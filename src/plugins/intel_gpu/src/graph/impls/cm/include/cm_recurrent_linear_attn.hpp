//# CM kernel for flash attn, reference
#include <cm/cm.h>
#include <cm/cmtl.h>

//# CM-compiler is C++17
static_assert(__cplusplus >= 201703L);

// load and convert
template <typename DST_TYPE, typename SRC_TYPE, int N, std::enable_if_t<!std::is_same<SRC_TYPE, DST_TYPE>::value, bool> = true>
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
template <typename DST_TYPE, typename SRC_TYPE, int N, std::enable_if_t<std::is_same<SRC_TYPE, DST_TYPE>::value, bool> = true>
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
template <typename DST_TYPE, typename SRC_TYPE, int N, std::enable_if_t<std::is_same<SRC_TYPE, DST_TYPE>::value, bool> = true>
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
template <typename DST_TYPE, typename SRC_TYPE, int N, std::enable_if_t<!std::is_same<SRC_TYPE, DST_TYPE>::value, bool> = true>
CM_INLINE void cm_store_by_row(SurfaceIndex base, vector_ref<SRC_TYPE, N> data, uint offset) {
    static_assert(std::is_same<DST_TYPE, half>::value);
    if constexpr (std::is_same<DST_TYPE, half>::value) {
        vector<half, N> temp(data);
        cm_store<uint, N/2>(base, offset, temp.format<uint>());
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

// TO DO: Support different input data types
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
                           SurfaceIndex q [[type("buffer_t")]],
                           SurfaceIndex k [[type("buffer_t")]],
                           SurfaceIndex v [[type("buffer_t")]],
                           SurfaceIndex g [[type("buffer_t")]],
                           SurfaceIndex beta [[type("buffer_t")]],
                           SurfaceIndex initial_state [[type("buffer_t")]],
                           SurfaceIndex output [[type("buffer_t")]]) {
    constexpr int v_head_dim_per_t = v_head_dims / 8;  // 16
    vector<float, v_head_dim_per_t * k_head_dims> h0;
// h0 [B, H, V, K]
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * k_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims +
                     v_head_dim_idx * k_head_dims;
        cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(h0.select<k_head_dims, 1>(k_head_dims * i),
                                                         initial_state,
                                                         stride * sizeof(IN_OUT_DTYPE));
    }

    for (int s = 0; s < SEQ_LEN; s++) {
        // beta B, T, HV
        // g B, T, HV
        int stride = b_idx * SEQ_LEN * v_num_heads + s * v_num_heads + head_idx;
        vector<float, 1> b_beta; // cm_load<float, 1>(beta, stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, 1>(b_beta, beta, stride * sizeof(IN_OUT_DTYPE));
        vector<float, 1> b_g;// cm_load<float, 1>(g, stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, 1>(b_g, g, stride * sizeof(IN_OUT_DTYPE));
        // if (head_dim_t_idx == 0) {
        //     printf("b_idx %d head_idx %d head_dim_t_idx %d beta_cur %f b_g %f\n",
        //            b_idx,
        //            head_idx,
        //            head_dim_t_idx,
        //            b_beta[0],
        //            b_g[0]);
        // }
        // B, T, HK, K
        int qk_stride =
            b_idx * SEQ_LEN * k_num_heads * k_head_dims + s * k_num_heads * k_head_dims + head_idx * k_head_dims;
        if constexpr (PRE_FETCH_DPT > 0) {
            if ((s % PRE_FETCH_CNT == 0) && head_dim_t_idx < PRE_FETCH_CNT) {
                const int qkv_stride =
                    (b_idx * SEQ_LEN * k_num_heads * k_head_dims +
                     (s + PRE_FETCH_DPT + head_dim_t_idx) * k_num_heads * k_head_dims + head_idx * k_head_dims) *
                    sizeof(IN_OUT_DTYPE);
                cm_prefetch_by_row<k_head_dims>(q, qkv_stride);
                cm_prefetch_by_row<k_head_dims>(k, qkv_stride);
                cm_prefetch_by_row<k_head_dims>(v, qkv_stride);
            }
        }
        // read q k
        vector<float, k_head_dims> b_q;  // cm_load<float, k_head_dims>(q, qk_stride * 4);
        cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(b_q, q, qk_stride * sizeof(IN_OUT_DTYPE));
        vector<float, k_head_dims> b_k;  // cm_load<float, k_head_dims>(k, qk_stride * 4);
        cm_load_by_row<float, IN_OUT_DTYPE, k_head_dims>(b_k, k, qk_stride * sizeof(IN_OUT_DTYPE));
        if constexpr (use_qk_l2norm) {
            float eps = 0.000001;
            float q_sum = cm_sum<float>(b_q * b_q);
            b_q = b_q * cm_rsqrt(q_sum + eps);
            float k_sum = cm_sum<float>(b_k * b_k);
            b_k = b_k * cm_rsqrt(k_sum + eps);
        }
        // read_v
        // B, T, HV, V
        int v_stride = b_idx * SEQ_LEN * v_num_heads * v_head_dims + s * v_num_heads * v_head_dims +
                       head_idx * v_head_dims + head_dim_t_idx * v_head_dim_per_t;
        vector<float, v_head_dim_per_t> b_v;//cm_load<uint, v_head_dim_per_t>(v, v_stride * sizeof(IN_OUT_DTYPE));
        cm_load_by_row<float, IN_OUT_DTYPE, v_head_dim_per_t>(b_v, v, v_stride * sizeof(IN_OUT_DTYPE));
        if constexpr (PRE_FETCH_DPT > 0) {
            if (s % 8 == 7) {
                cm_fence(CM_LOCAL_BARRIER);
            }
        } else {
            cm_barrier();
        }
        // if (head_dim_t_idx == 0) {
        //     printf("b_idx %d head_idx %d head_dim_t_idx %d b_q %f b_q %f\n", b_idx, head_idx, head_dim_t_idx, b_q[0],
        //     b_q[1]); printf("b_idx %d head_idx %d head_dim_t_idx %d b_k %f b_k %f\n", b_idx, head_idx,
        //     head_dim_t_idx, b_k[0], b_k[1]);
        // }
        vector<float, v_head_dim_per_t> cur_output;
        vector<float, v_head_dim_per_t> h_k;
        constexpr float log2e = 1.4426950408889634f;
        float g_cur = cm_exp(b_g[0] * log2e);
#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) * g_cur;
            h_k[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_k);
            // if (head_dim_t_idx == 0 && b_idx == 1) {
            //     printf("b_idx %d head_idx %d head_dim_t_idx %d h_k %f g_cur %f\n",
            //            b_idx,
            //            head_idx,
            //            head_dim_t_idx,
            //            h_k[i],
            //            g_cur);
            // }
        }
        vector<float, v_head_dim_per_t> delta_full = (b_v - h_k) * b_beta[0];
        // if (head_dim_t_idx == 0 && b_idx == 1) {
        //     printf("b_idx %d head_idx %d head_dim_t_idx %d delta_full %f\n",
        //            b_idx,
        //            head_idx,
        //            head_dim_t_idx,
        //            delta_full[0]);
        // }

#pragma unroll
        for (int i = 0; i < v_head_dim_per_t; i++) {
            float detla = delta_full[i];
            h0.select<k_head_dims, 1>(k_head_dims * i) = h0.select<k_head_dims, 1>(k_head_dims * i) + b_k * detla;
            cur_output[i] = cm_sum<float>(h0.select<k_head_dims, 1>(k_head_dims * i) * b_q);
            // if (head_dim_t_idx == 0 && b_idx == 1) {
            //     printf("b_idx %d head_idx %d head_dim_t_idx %d h_k %f\n",
            //            b_idx,
            //            head_idx,
            //            head_dim_t_idx,
            //            h0.select<k_head_dims, 1>(k_head_dims * i)[0]);
            // }
        }
        // B, T, HV, V
        int output_stride = b_idx * SEQ_LEN * v_num_heads * v_head_dims + s * v_num_heads * v_head_dims +
                            head_idx * v_head_dims + head_dim_t_idx * v_head_dim_per_t;
        cm_store_by_row<IN_OUT_DTYPE, float, v_head_dim_per_t>(output, cur_output, output_stride * sizeof(IN_OUT_DTYPE));
    }
#pragma unroll
    for (int i = 0; i < v_head_dim_per_t; i++) {
        int v_head_dim_idx = head_dim_t_idx * v_head_dim_per_t + i;
        int stride = b_idx * k_num_heads * v_head_dims * k_head_dims + head_idx * v_head_dims * k_head_dims +
                     v_head_dim_idx * k_head_dims;
        cm_store_by_row<IN_OUT_DTYPE, float, k_head_dims>(initial_state,
                                                          h0.select<k_head_dims, 1>(k_head_dims * i),
                                                          stride * sizeof(IN_OUT_DTYPE));
        // if constexpr (k_head_dims == 128) {
        //     cm_store<float, 64>(initial_state, stride * 4, h0.select<64, 1>(k_head_dims * i));
        //     cm_store<float, 64>(initial_state, stride * 4 + 4 * 64, h0.select<64, 1>(k_head_dims * i + 64));
        // } else if constexpr (k_head_dims <= 64) {
        //     cm_store<float, k_head_dims>(initial_state, stride * 4, h0.select<k_head_dims, 1>(k_head_dims * i));
        // }
    }
}

extern "C" _GENX_MAIN_ void recurrent_gated_delta_rule(SurfaceIndex q [[type("buffer_t")]],
                                                       SurfaceIndex k [[type("buffer_t")]],
                                                       SurfaceIndex v [[type("buffer_t")]],
                                                       SurfaceIndex g [[type("buffer_t")]],
                                                       SurfaceIndex beta [[type("buffer_t")]],
                                                       SurfaceIndex initial_state [[type("buffer_t")]],
                                                       SurfaceIndex output [[type("buffer_t")]]) {
    int b_idx = cm_group_id(0);
    int head_idx = cm_group_id(1);
    int head_dim_t_idx = cm_local_id(2);
    constexpr int k_num_heads = K_HEAD_NUMS;
    constexpr int v_num_heads = V_HEAD_NUMS;
    constexpr int k_head_dims = K_HEAD_DIMS;
    constexpr int v_head_dims = V_HEAD_DIMS;
#if IO_TYPE == 0
    recurrent_linear_attn<half, k_num_heads, v_num_heads, k_head_dims, v_head_dims, true, 2, 4>(b_idx,
                                                                                                head_idx,
                                                                                                head_dim_t_idx,
                                                                                                q,
                                                                                                k,
                                                                                                v,
                                                                                                g,
                                                                                                beta,
                                                                                                initial_state,
                                                                                                output);
#elif IO_TYPE == 1
    recurrent_linear_attn<float, k_num_heads, v_num_heads, k_head_dims, v_head_dims, true, 2, 4>(b_idx,
                                                                                                 head_idx,
                                                                                                 head_dim_t_idx,
                                                                                                 q,
                                                                                                 k,
                                                                                                 v,
                                                                                                 g,
                                                                                                 beta,
                                                                                                 initial_state,
                                                                                                 output);
#endif
}
