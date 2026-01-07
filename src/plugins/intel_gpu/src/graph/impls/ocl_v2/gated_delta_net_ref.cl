#define V_BLOCK_SIZE 4

inline float l2norm_scale(float sum, float extra_scale) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + 0.000001f) * extra_scale;
}

REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(gated_delta_net_ref)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* initial_state,
 __global INPUT4_TYPE* g,
 __global INPUT5_TYPE* beta,
 __global OUTPUT_TYPE* output,
 __global OUTPUT1_TYPE* output_state,
 int seq_len) {
    const int T_len = seq_len;
    const int H_len = K_HEAD_NUMS;
    const int K_len = K_HEAD_DIMS;
    const int V_len = V_HEAD_DIMS;

    const int start_iv = get_group_id(2) * V_BLOCK_SIZE;
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int lid = get_sub_group_local_id();

    const int QK_T_STRIDE = H_len * K_len;
    const int QK_B_STRIDE = T_len * QK_T_STRIDE;
    const int V_T_STRIDE = H_len * V_len;
    const int V_B_STRIDE = T_len * V_T_STRIDE;
    const int STATE_BASE = (b * H_len + h) * (K_len * V_len);

#if (K_HEAD_DIMS == 128)
    float8 h_state[V_BLOCK_SIZE];
#else
    float h_state_f[V_BLOCK_SIZE][K_HEAD_DIMS];
#endif

#if (K_HEAD_DIMS == 128)
// 1. LOAD STATE BLOCK: 4 V-columns, 8 K-rows per lane
#    pragma unroll
    for (int row_chunk = 0; row_chunk < (K_HEAD_DIMS / 16); row_chunk++) {
        int row_idx = row_chunk * 16 + lid;
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            h_state[v_idx][row_chunk] = convert_float(initial_state[STATE_BASE + row_idx * V_len + curr_iv]);
        }
    }
#else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                h_state_f[v_idx][k_idx] = convert_float(initial_state[STATE_BASE + k_idx * V_len + curr_iv]);
            }
        }
    }
#endif

    for (int t = 0; t < T_len; t++) {
#if (K_HEAD_DIMS == 128)
        // 2. LOAD COMMON TIMESTEP DATA
        int g_idx = (b * T_len + t) * H_len + h;
        float b_g = exp(convert_float(g[g_idx]));
        float b_beta = convert_float(beta[g_idx]);

        int qk_offset = b * QK_B_STRIDE + t * QK_T_STRIDE + h * K_len;
        int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + h * V_len;
        // Each lane owns an 8-wide K slice.
#    define DATA_VEC_K MAKE_VECTOR_TYPE(INPUT1_TYPE, 8)
        DATA_VEC_K b_k_vec = BLOCK_READN(INPUT1_TYPE, 8, k, qk_offset);
#    undef DATA_VEC_K
#    define DATA_VEC_Q MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
        DATA_VEC_Q b_q_vec = BLOCK_READN(INPUT0_TYPE, 8, q, qk_offset);
#    undef DATA_VEC_Q
        float8 b_k = convert_float8(b_k_vec);
        float8 b_q = convert_float8(b_q_vec);

        // normalize k and q (l2norm + q scale)
        float k_sum = 0.0f;
#    pragma unroll
        for (int j = 0; j < 8; j++)
            k_sum += b_k[j] * b_k[j];
        float k_scale = l2norm_scale(k_sum, 1.0f);
        b_k *= (float8)(k_scale);

        float q_sum = 0.0f;
#    pragma unroll
        for (int j = 0; j < 8; j++)
            q_sum += b_q[j] * b_q[j];
        float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
        b_q *= (float8)(q_scale);

        // V load
        float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
        }

// 3. RECURRENT UPDATE
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            h_state[v_idx] *= b_g;

            float dot_part_k = 0.0f;
#    pragma unroll
            for (int j = 0; j < 8; j++)
                dot_part_k += h_state[v_idx][j] * b_k[j];
            float h_k = sub_group_reduce_add(dot_part_k);
            h_k = sub_group_broadcast(h_k, 0);

            float update_val = (b_v_vec[v_idx] - h_k) * b_beta;
            h_state[v_idx] += (b_k * update_val);

            float dot_part_q = 0.0f;
#    pragma unroll
            for (int j = 0; j < 8; j++)
                dot_part_q += h_state[v_idx][j] * b_q[j];
            float b_output = sub_group_reduce_add(dot_part_q);
            b_output = sub_group_broadcast(b_output, 0);

            output[v_offset + start_iv + v_idx] = convert_OUTPUT_TYPE(b_output);
        }
#else
        if (lid == 0) {
            int g_idx = (b * T_len + t) * H_len + h;
            float b_g = exp(convert_float(g[g_idx]));
            float b_beta = convert_float(beta[g_idx]);

            int qk_offset = b * QK_B_STRIDE + t * QK_T_STRIDE + h * K_len;
            int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + h * V_len;

            float k_sum = 0.0f;
            float q_sum = 0.0f;
            for (int k_idx = 0; k_idx < K_len; k_idx++) {
                float k_val = convert_float(k[qk_offset + k_idx]);
                float q_val = convert_float(q[qk_offset + k_idx]);
                k_sum += k_val * k_val;
                q_sum += q_val * q_val;
            }
            float k_scale = l2norm_scale(k_sum, 1.0f);
            float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);

            float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#    pragma unroll
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
            }

            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                float h_k = 0.0f;
                for (int k_idx = 0; k_idx < K_len; k_idx++) {
                    float k_norm = convert_float(k[qk_offset + k_idx]) * k_scale;
                    h_state_f[v_idx][k_idx] *= b_g;
                    h_k += h_state_f[v_idx][k_idx] * k_norm;
                }

                float update_val = (b_v_vec[v_idx] - h_k) * b_beta;

                float b_output = 0.0f;
                for (int k_idx = 0; k_idx < K_len; k_idx++) {
                    float k_norm = convert_float(k[qk_offset + k_idx]) * k_scale;
                    float q_norm = convert_float(q[qk_offset + k_idx]) * q_scale;
                    h_state_f[v_idx][k_idx] += k_norm * update_val;
                    b_output += h_state_f[v_idx][k_idx] * q_norm;
                }

                output[v_offset + start_iv + v_idx] = convert_OUTPUT_TYPE(b_output);
            }
        }
#endif
    }

#if (K_HEAD_DIMS == 128)
// 4. WRITE BACK STATE BLOCK
#    pragma unroll
    for (int row_chunk = 0; row_chunk < 8; row_chunk++) {
        int row_idx = row_chunk * 16 + lid;
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            output_state[STATE_BASE + row_idx * V_len + curr_iv] = convert_OUTPUT1_TYPE(h_state[v_idx][row_chunk]);
        }
    }
#else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                output_state[STATE_BASE + k_idx * V_len + curr_iv] = convert_OUTPUT1_TYPE(h_state_f[v_idx][k_idx]);
            }
        }
    }
#endif
}