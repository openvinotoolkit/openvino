#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"
#define V_BLOCK_SIZE 4

inline float l2norm_scale(float sum, float extra_scale) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + 0.000001f) * extra_scale;
}

inline float sum8(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
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
    const int V_len = K_HEAD_DIM;

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

#if (K_HEAD_DIM == 128)
    float8 h_state[V_BLOCK_SIZE][K_HEAD_DIM / (SUBGROUP_SIZE * 8)];
#else
    float h_state_f[V_BLOCK_SIZE][K_HEAD_DIM];
#endif

#if (K_HEAD_DIM == 128)
// 1. LOAD STATE BLOCK: 4 V-columns, 8 K-rows per lane
#    pragma unroll
    for (int row_chunk = 0; row_chunk < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); row_chunk++) {
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            float8 lane_state = (float8)(0.0f);
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = lid * (K_HEAD_DIM / SUBGROUP_SIZE) + row_chunk * 8 + j;
                lane_state[j] = convert_float(initial_state[STATE_BASE + curr_iv * K_len + row_idx]);
            }
            h_state[v_idx][row_chunk] = lane_state;
        }
    }
#else
    if (lid == 0) {
        for (int k_idx = 0; k_idx < K_len; k_idx++) {
            for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
                int curr_iv = start_iv + v_idx;
                h_state_f[v_idx][k_idx] = convert_float(initial_state[STATE_BASE + curr_iv * K_len + k_idx]);
            }
        }
    }
#endif

    for (int t = 0; t < T_len; t++) {
#if (K_HEAD_DIM == 128)
        // 2. LOAD COMMON TIMESTEP DATA
        int g_idx = (b * T_len + t) * H_len + h;
        float b_g = exp(convert_float(g[g_idx]));
        float b_beta = convert_float(beta[g_idx]);
        const int lane_k_base = lid * (K_HEAD_DIM / SUBGROUP_SIZE);

        const int group_size = H_len / HK_len;
        const int hk = h / group_size;
        int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + hk * K_len;
        int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
        int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
        int out_offset = b * T_len * H_len * V_len + t * H_len * V_len + h * V_len;
        float8 b_k[K_HEAD_DIM / (SUBGROUP_SIZE * 8)];
        float8 b_q[K_HEAD_DIM / (SUBGROUP_SIZE * 8)];

        // normalize k and q (l2norm + q scale)
        float k_sum = 0.0f;
#    pragma unroll
        for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++) {
            float8 lane_k = (float8)(0.0f);
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                int k_idx = lane_k_base + c * 8 + j;
                lane_k[j] = convert_float(k[k_offset + k_idx]);
                k_sum += lane_k[j] * lane_k[j];
            }
            b_k[c] = lane_k;
        }
        float k_scale = l2norm_scale(k_sum, 1.0f);
        for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
            b_k[c] *= (float8)(k_scale);

        float q_sum = 0.0f;
#    pragma unroll
        for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++) {
            float8 lane_q = (float8)(0.0f);
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                int q_idx = lane_k_base + c * 8 + j;
                lane_q[j] = convert_float(q[q_offset + q_idx]);
                q_sum += lane_q[j] * lane_q[j];
            }
            b_q[c] = lane_q;
        }
        float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
        for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
            b_q[c] *= (float8)(q_scale);

        // V load
        float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            if (curr_iv < V_len)
                b_v_vec[v_idx] = convert_float(v[v_offset + curr_iv]);
        }

        // 3. RECURRENT UPDATE
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            if (curr_iv >= V_len)
                continue;

            for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
                h_state[v_idx][c] *= (float8)(b_g);

            float dot_part_k = 0.0f;
#    pragma unroll
            for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
                dot_part_k += sum8(h_state[v_idx][c] * b_k[c]);
            float h_k = sub_group_reduce_add(dot_part_k);

            float update_val = (b_v_vec[v_idx] - h_k) * b_beta;
            for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
                h_state[v_idx][c] += (b_k[c] * (float8)(update_val));

            float dot_part_q = 0.0f;
#    pragma unroll
            for (int c = 0; c < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); c++)
                dot_part_q += sum8(h_state[v_idx][c] * b_q[c]);
            float b_output = sub_group_reduce_add(dot_part_q);

            if (lid == 0) {
                output[out_offset + curr_iv] = TO_OUTPUT_TYPE(b_output);
            }
        }
#else
        if (lid == 0) {
            int g_idx = (b * T_len + t) * H_len + h;
            float b_g = exp(convert_float(g[g_idx]));
            float b_beta = convert_float(beta[g_idx]);

            const int group_size = H_len / HK_len;
            const int hk = h / group_size;
            int q_offset = b * Q_B_STRIDE + t * Q_T_STRIDE + hk * K_len;
            int k_offset = b * K_B_STRIDE + t * K_T_STRIDE + (hk + key_offset) * K_len;
            int v_offset = b * V_B_STRIDE + t * V_T_STRIDE + (h + value_offset) * V_len;
            int out_offset = b * T_len * H_len * V_len + t * H_len * V_len + h * V_len;

            float k_sum = 0.0f;
            float q_sum = 0.0f;
            for (int k_idx = 0; k_idx < K_len; k_idx++) {
                float k_val = convert_float(k[k_offset + k_idx]);
                float q_val = convert_float(q[q_offset + k_idx]);
                k_sum += k_val * k_val;
                q_sum += q_val * q_val;
            }
            float k_scale = rsqrt(k_sum + 0.000001f);
            float q_scale = rsqrt(q_sum + 0.000001f) * SCALE_FACTOR;

            float4 b_v_vec = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
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
                    float k_norm = convert_float(k[k_offset + k_idx]) * k_scale;
                    h_state_f[v_idx][k_idx] *= b_g;
                    h_k += h_state_f[v_idx][k_idx] * k_norm;
                }

                float update_val = (b_v_vec[v_idx] - h_k) * b_beta;

                float b_output = 0.0f;
                for (int k_idx = 0; k_idx < K_len; k_idx++) {
                    float k_norm = convert_float(k[k_offset + k_idx]) * k_scale;
                    float q_norm = convert_float(q[q_offset + k_idx]) * q_scale;
                    h_state_f[v_idx][k_idx] += k_norm * update_val;
                    b_output += h_state_f[v_idx][k_idx] * q_norm;
                }

                output[out_offset + curr_iv] = TO_OUTPUT_TYPE(b_output);
            }
        }
#endif
    }

#if (K_HEAD_DIM == 128)
// 4. WRITE BACK STATE BLOCK
#    pragma unroll
    for (int row_chunk = 0; row_chunk < (K_HEAD_DIM / (SUBGROUP_SIZE * 8)); row_chunk++) {
#    pragma unroll
        for (int v_idx = 0; v_idx < V_BLOCK_SIZE; v_idx++) {
            int curr_iv = start_iv + v_idx;
            if (curr_iv >= V_len)
                continue;
#    pragma unroll
            for (int j = 0; j < 8; j++) {
                int row_idx = lid * (K_HEAD_DIM / SUBGROUP_SIZE) + row_chunk * 8 + j;
#if OUTPUT_STATE
                output_state[STATE_BASE + curr_iv * K_len + row_idx] = (OUTPUT1_TYPE)(h_state[v_idx][row_chunk][j]);
#else
                initial_state[STATE_BASE + curr_iv * K_len + row_idx] = (INPUT3_TYPE)(h_state[v_idx][row_chunk][j]);
#endif
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
#if OUTPUT_STATE
                output_state[STATE_BASE + curr_iv * K_len + k_idx] = (OUTPUT1_TYPE)(h_state_f[v_idx][k_idx]);
#else
                initial_state[STATE_BASE + curr_iv * K_len + k_idx] = (INPUT3_TYPE)(h_state_f[v_idx][k_idx]);
#endif
            }
        }
    }
#endif
}