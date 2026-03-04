#include "include/batch_headers/common.cl"
#include "include/batch_headers/fetch_data.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

#ifndef TO_INPUT5_TYPE2
#    define TO_INPUT5_TYPE2(x) CAT(convert_, MAKE_VECTOR_TYPE(INPUT5_TYPE, 2))(x)
#endif

#ifndef TO_INPUT5_TYPE8
#    define TO_INPUT5_TYPE8(x) CAT(convert_, MAKE_VECTOR_TYPE(INPUT5_TYPE, 8))(x)
#endif
#define V_BLOCK_SIZE 4
float sum2(float2 v) {
    return v.s0 + v.s1;
}
float sum8(float8 v) {
    return v.s0 + v.s1 + v.s2 + v.s3 + v.s4 + v.s5 + v.s6 + v.s7;
}
inline float l2norm_scale(float sum, float extra_scale) {
    sum = sub_group_reduce_add(sum);
    sum = sub_group_broadcast(sum, 0);
    return rsqrt(sum + 0.000001f) * extra_scale;
}

#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
inline void normalize_kq_128_sg8(float8* b_k, float8* b_q) {
    float k_sum = sum8(b_k[0] * b_k[0]) + sum8(b_k[1] * b_k[1]);
    float k_scale = l2norm_scale(k_sum, 1.0f);
    b_k[0] *= k_scale;
    b_k[1] *= k_scale;

    float q_sum = sum8(b_q[0] * b_q[0]) + sum8(b_q[1] * b_q[1]);
    float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
    b_q[0] *= q_scale;
    b_q[1] *= q_scale;
}
#    else
inline void normalize_kq_128(float8* b_k, float8* b_q) {
    float k_sum = sum8((*b_k) * (*b_k));
    float k_scale = l2norm_scale(k_sum, 1.0f);
    *b_k *= k_scale;

    float q_sum = sum8((*b_q) * (*b_q));
    float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
    *b_q *= q_scale;
}
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
inline void normalize_kq_32_sg16(float2* b_k, float2* b_q, int id_sg_local) {
    float k_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        k_sum += sum2(b_k[idx] * b_k[idx]);
    }
    float k_scale = l2norm_scale(k_sum, 1.0f);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        b_k[idx] *= (float2)(k_scale);
    }

    float q_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        q_sum += sum2(b_q[idx] * b_q[idx]);
    }
    float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
        b_q[idx] *= (float2)(q_scale);
    }
}
#    else
inline void normalize_kq_32(float2* b_k, float2* b_q, int id_sg_local) {
    float k_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        k_sum = fma(b_k[idx], b_k[idx], k_sum);
    }
    k_sum = sub_group_reduce_add(k_sum);
    k_sum = sub_group_broadcast(k_sum, 0);
    float k_scale = rsqrt(k_sum + 0.000001f);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_k[idx] *= k_scale;
    }

    float q_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        q_sum = fma(b_q[idx], b_q[idx], q_sum);
    }
    float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_q[idx] *= q_scale;
    }
}
#    endif
#else
inline void normalize_kq_generic(float* b_k, float* b_q, int id_sg_local) {
    float k_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        k_sum = fma(b_k[idx], b_k[idx], k_sum);
    }
    float k_scale = l2norm_scale(k_sum, 1.0f);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_k[idx] *= k_scale;
    }

    float q_sum = 0.0f;
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        q_sum = fma(b_q[idx], b_q[idx], q_sum);
    }
    float q_scale = l2norm_scale(q_sum, SCALE_FACTOR);
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_q[idx] *= q_scale;
    }
}
#endif

#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
inline void load_init_state_128_sg8(float8* init_state_vec, const __global INPUT5_TYPE* initial_state, int init_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8_0 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
    DATA_VEC h8_1 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base + (SUBGROUP_SIZE * 8));
#        undef DATA_VEC
    init_state_vec[0] = convert_float8(h8_0);
    init_state_vec[1] = convert_float8(h8_1);
}

inline void load_kq_128_sg8(float8* b_k, float8* b_q, const __global INPUT1_TYPE* k_ptr, const __global INPUT0_TYPE* q_ptr, int k_base, int q_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
    b_k[0] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base));
    b_k[1] = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base + (SUBGROUP_SIZE * 8)));
    b_q[0] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base));
    b_q[1] = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base + (SUBGROUP_SIZE * 8)));
#        undef DATA_VEC
}
#    else
inline void load_init_state_128(float8* init_state_vec, const __global INPUT5_TYPE* initial_state, int init_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8 = BLOCK_READN(INPUT5_TYPE, 8, initial_state, init_base);
#        undef DATA_VEC
    *init_state_vec = convert_float8(h8);
}

inline void load_kq_128(float8* b_k, float8* b_q, const __global INPUT1_TYPE* k_ptr, const __global INPUT0_TYPE* q_ptr, int k_base, int q_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 8)
    *b_k = convert_float8(BLOCK_READN(INPUT1_TYPE, 8, k_ptr, k_base));
    *b_q = convert_float8(BLOCK_READN(INPUT0_TYPE, 8, q_ptr, q_base));
#        undef DATA_VEC
}
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
inline void load_init_state_32_sg16(float2* init_state_vec, const __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
        DATA_VEC h2 = BLOCK_READN(INPUT5_TYPE, 2, initial_state, init_base + (j - id_sg_local));
#        undef DATA_VEC
        init_state_vec[idx] = convert_float2(h2);
    }
}

inline void load_kq_32_sg16(float2* b_k, float2* b_q, const __global INPUT1_TYPE* k_ptr, const __global INPUT0_TYPE* q_ptr, int k_base, int q_base, int id_sg_local) {
#        pragma unroll
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT0_TYPE, 2)
    b_k[idx] = convert_float2(BLOCK_READN(INPUT1_TYPE, 2, k_ptr, k_base + (j - id_sg_local)));
    b_q[idx] = convert_float2(BLOCK_READN(INPUT0_TYPE, 2, q_ptr, q_base + (j - id_sg_local)));
#        undef DATA_VEC
    }
}
#    else
inline void load_init_state_32(float2* init_state_vec, const __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        init_state_vec[idx] = convert_float(initial_state[init_base + j]);
    }
}

inline void load_kq_32(float2* b_k, float2* b_q, const __global INPUT1_TYPE* k_ptr, const __global INPUT0_TYPE* q_ptr, int k_base, int q_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_k[idx] = convert_float(k_ptr[k_base + j]);
        b_q[idx] = convert_float(q_ptr[q_base + j]);
    }
}
#    endif
#else
inline void load_init_state_generic(float* init_state_vec, const __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        init_state_vec[idx] = convert_float(initial_state[init_base + j]);
    }
}

inline void load_kq_generic(float* b_k, float* b_q, const __global INPUT1_TYPE* k_ptr, const __global INPUT0_TYPE* q_ptr, int k_base, int q_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        b_k[idx] = convert_float(k_ptr[k_base + j]);
        b_q[idx] = convert_float(q_ptr[q_base + j]);
    }
}
#endif

#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
inline void store_init_state_128_sg8(const float8* init_state_vec, __global INPUT5_TYPE* initial_state, int init_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8_0 = TO_INPUT5_TYPE8(init_state_vec[0]);
    DATA_VEC h8_1 = TO_INPUT5_TYPE8(init_state_vec[1]);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base, h8_0);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base + (SUBGROUP_SIZE * 8), h8_1);
#        undef DATA_VEC
}
#    else
inline void store_init_state_128(const float8* init_state_vec, __global INPUT5_TYPE* initial_state, int init_base) {
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 8)
    DATA_VEC h8 = TO_INPUT5_TYPE8(*init_state_vec);
    BLOCK_WRITEN(INPUT5_TYPE, 8, initial_state, init_base, h8);
#        undef DATA_VEC
}
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
inline void store_init_state_32_sg16(const float2* init_state_vec, __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
        int idx = j >> 5;
#        define DATA_VEC MAKE_VECTOR_TYPE(INPUT5_TYPE, 2)
        DATA_VEC h2 = TO_INPUT5_TYPE2(init_state_vec[idx]);
        BLOCK_WRITEN(INPUT5_TYPE, 2, initial_state, init_base + (j - id_sg_local), h2);
#        undef DATA_VEC
    }
}
#    else
inline void store_init_state_32(const float2* init_state_vec, __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        initial_state[init_base + j] = TO_INPUT5_TYPE(init_state_vec[idx]);
    }
}
#    endif
#else
inline void store_init_state_generic(const float* init_state_vec, __global INPUT5_TYPE* initial_state, int init_base, int id_sg_local) {
    for (int j = id_sg_local; j < K_HEAD_DIMS; j += SUBGROUP_SIZE) {
        int idx = j / SUBGROUP_SIZE;
        initial_state[init_base + j] = TO_INPUT5_TYPE(init_state_vec[idx]);
    }
}
#endif
REQD_SUB_GROUP_SIZE(SUBGROUP_SIZE)
KERNEL(linear_attention_ref)
(__global INPUT0_TYPE* q,
 __global INPUT1_TYPE* k,
 __global INPUT2_TYPE* v,
 __global INPUT3_TYPE* g,
 __global INPUT4_TYPE* beta,
 __global INPUT5_TYPE* initial_state,
 __global OUTPUT_TYPE* output,
#if OUTPUT_STATE
 __global OUTPUT1_TYPE* output_state,
#endif
 int seq_len,
 int key_offset,
 int value_offset) {
    int b = get_global_id(0);
    int gid1 = get_global_id(1);
    int BATCH_STRIDE = Q_HEAD_NUMS * seq_len * K_HEAD_DIMS;
    int STEP_STRIDE = Q_HEAD_NUMS * K_HEAD_DIMS;
    int OUTPUT_STEP_STRIDE = V_HEAD_NUMS * K_HEAD_DIMS;
    int KEY_STEP_STRIDE = (Q_HEAD_NUMS + key_offset) * K_HEAD_DIMS;
    int VALUE_STEP_STRIDE = (V_HEAD_NUMS + value_offset) * K_HEAD_DIMS;
    int KEY_BATCH_STRIDE = KEY_STEP_STRIDE * seq_len;
    int VALUE_BATCH_STRIDE = VALUE_STEP_STRIDE * seq_len;
    int v_blocks = (K_HEAD_DIMS + V_BLOCK_SIZE - 1) / V_BLOCK_SIZE;
    int h = gid1 / v_blocks;
    int group_size = V_HEAD_NUMS / Q_HEAD_NUMS;
    int qk_h = h / group_size;
    int v_block_id = gid1 - h * v_blocks;
    int i_v_base = v_block_id * V_BLOCK_SIZE;
    const __global INPUT0_TYPE* q_ptr = q + b * BATCH_STRIDE;
    const __global INPUT1_TYPE* k_ptr = k + b * KEY_BATCH_STRIDE;
    const __global INPUT2_TYPE* v_ptr = v + b * VALUE_BATCH_STRIDE;
    const __global INPUT3_TYPE* g_ptr = g + b * V_HEAD_NUMS * seq_len;
    const __global INPUT4_TYPE* beta_ptr = beta + b * V_HEAD_NUMS * seq_len;
    int out_base = b * V_HEAD_NUMS * seq_len * K_HEAD_DIMS + h * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
    float8 init_state[V_BLOCK_SIZE][2];
    float8 b_k[2];
    float8 b_q[2];
#    else
    float8 init_state[V_BLOCK_SIZE];
    float8 b_k;
    float8 b_q;
#    endif
#elif (K_HEAD_DIMS % 32) == 0
    float2 init_state[V_BLOCK_SIZE][K_HEAD_DIMS / 32];
    float2 b_k[K_HEAD_DIMS / 32];
    float2 b_q[K_HEAD_DIMS / 32];
#else
    float init_state[V_BLOCK_SIZE][CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
    float b_k[CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
    float b_q[CEIL_DIV(K_HEAD_DIMS, SUBGROUP_SIZE)] = {0};
#endif
    int id_sg_local = get_sub_group_local_id();

    // load initial state
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
        load_init_state_128_sg8(init_state[iv], initial_state, init_base);
#    else
        load_init_state_128(&init_state[iv], initial_state, init_base);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        load_init_state_32_sg16(init_state[iv], initial_state, init_base, id_sg_local);
#    else
        load_init_state_32(init_state[iv], initial_state, init_base, id_sg_local);
#    endif
#else
        load_init_state_generic(init_state[iv], initial_state, init_base, id_sg_local);
#endif
    }

        int q_base = qk_h * K_HEAD_DIMS;
        int k_base = (qk_h + key_offset) * K_HEAD_DIMS;
        int v_base = (h + value_offset) * K_HEAD_DIMS;
        int out_i_base = out_base;
    //  loop over time step
        for (int i = 0; i < seq_len; i++, q_base += STEP_STRIDE, k_base += KEY_STEP_STRIDE, v_base += VALUE_STEP_STRIDE, out_i_base += OUTPUT_STEP_STRIDE) {
        float b_g = exp(convert_float(g_ptr[i * V_HEAD_NUMS + h]));
        float b_beta = convert_float(beta_ptr[i * V_HEAD_NUMS + h]);
        // load k and q
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
        load_kq_128_sg8(b_k, b_q, k_ptr, q_ptr, k_base, q_base);
#    else
        load_kq_128(&b_k, &b_q, k_ptr, q_ptr, k_base, q_base);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        load_kq_32_sg16(b_k, b_q, k_ptr, q_ptr, k_base, q_base, id_sg_local);
#    else
        load_kq_32(b_k, b_q, k_ptr, q_ptr, k_base, q_base, id_sg_local);
#    endif
#else
        load_kq_generic(b_k, b_q, k_ptr, q_ptr, k_base, q_base, id_sg_local);
#endif
        // normalize k and q
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
        normalize_kq_128_sg8(b_k, b_q);
#    else
        normalize_kq_128(&b_k, &b_q);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        normalize_kq_32_sg16(b_k, b_q, id_sg_local);
#    else
        normalize_kq_32(b_k, b_q, id_sg_local);
#    endif
#else
        normalize_kq_generic(b_k, b_q, id_sg_local);
#endif

        for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
            int i_v = i_v_base + iv;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
            init_state[iv][0] *= b_g;
            init_state[iv][1] *= b_g;
            float hk_acc = sum8(init_state[iv][0] * b_k[0]) + sum8(init_state[iv][1] * b_k[1]);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv][0] = fma(b_k[0], (float8)(b_v), init_state[iv][0]);
            init_state[iv][1] = fma(b_k[1], (float8)(b_v), init_state[iv][1]);

            float out_acc = sum8(init_state[iv][0] * b_q[0]) + sum8(init_state[iv][1] * b_q[1]);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
            }
#    else
            init_state[iv] *= b_g;
            float hk_acc = sum8(init_state[iv] * b_k);
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            init_state[iv] = fma(b_k, (float8)(b_v), init_state[iv]);

            float out_acc = sum8(init_state[iv] * b_q);
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
            }
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                hk_acc += sum2(init_state[iv][idx] * b_k[idx]);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int j = id_sg_local; j < K_HEAD_DIMS; j += 32) {
                int idx = j >> 5;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
            }
#    else
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            int v_base_aligned = v_base + (i_v & ~(SUBGROUP_SIZE - 1));
            int v_lane = i_v & (SUBGROUP_SIZE - 1);
            INPUT2_TYPE v_val_h = AS_INPUT0_TYPE(BLOCK_READN(INPUT2_TYPE, 1, v_ptr, v_base_aligned));
            float v_val = convert_float(v_val_h);
            float b_v = sub_group_broadcast(v_val, v_lane);
            b_v -= hk_acc;
            b_v *= b_beta;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] = fma(b_k[idx], (float2)(b_v), init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                out_acc += sum2(init_state[iv][idx] * b_q[idx]);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
            }
#    endif
#else
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] *= b_g;
            }
            float hk_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                hk_acc = fma(init_state[iv][idx], b_k[idx], hk_acc);
            }
            hk_acc = sub_group_reduce_add(hk_acc);
            hk_acc = sub_group_broadcast(hk_acc, 0);

            float b_v = convert_float(v_ptr[v_base + i_v]);
            b_v -= hk_acc;
            b_v *= b_beta;

            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                init_state[iv][idx] = fma(b_k[idx], b_v, init_state[iv][idx]);
            }

            float out_acc = 0.0f;
            for (int n = id_sg_local; n < K_HEAD_DIMS; n += SUBGROUP_SIZE) {
                int idx = n / SUBGROUP_SIZE;
                out_acc = fma(init_state[iv][idx], b_q[idx], out_acc);
            }
            out_acc = sub_group_reduce_add(out_acc);
            out_acc = sub_group_broadcast(out_acc, 0);
            if (id_sg_local == 0) {
                output[out_i_base + i_v] = TO_OUTPUT_TYPE(out_acc);
            }
#endif
        }
    }
        // store final state
        __global INPUT5_TYPE* state_out = initial_state;
    #if OUTPUT_STATE
        state_out = (__global INPUT5_TYPE*)output_state;
    #endif
    for (int iv = 0; iv < V_BLOCK_SIZE; iv++) {
        int i_v = i_v_base + iv;
        int init_base = b * V_HEAD_NUMS * K_HEAD_DIMS * K_HEAD_DIMS + h * K_HEAD_DIMS * K_HEAD_DIMS + i_v * K_HEAD_DIMS;
#if (K_HEAD_DIMS == 128)
#    if (SUBGROUP_SIZE == 8)
        store_init_state_128_sg8(init_state[iv], state_out, init_base);
#    else
        store_init_state_128(&init_state[iv], state_out, init_base);
#    endif
#elif (K_HEAD_DIMS % 32) == 0
#    if (SUBGROUP_SIZE == 16)
        store_init_state_32_sg16(init_state[iv], state_out, init_base, id_sg_local);
#    else
        store_init_state_32(init_state[iv], state_out, init_base, id_sg_local);
#    endif
#else
        store_init_state_generic(init_state[iv], state_out, init_base, id_sg_local);
#endif
    }
}
