// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/fetch_data.cl"

#define N_ROUNDS 10
#define STATISTIC_MAXIMIZING_MULTIPLIER_N 0xD2511F53UL
#define STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER 0xCD9E8D57UL
#define CRUSH_RESISTANCE_CONST_LOWER_VALUE 0x9E3779B9U
#define CRUSH_RESISTANCE_CONST_UPPER_VALUE 0xBB67AE85U

#define FNAME(type) fill_##type
#define FUNC_NAME(type) FNAME(type)
#define FILL_FUNC(func, result, min_value, max_value, output, output_index) FUNC_CALL(func)(result, min_value, max_value, output, output_index)

inline ulong FUNC(unite_high_low)(uint high, uint low) {
    return ((ulong)high << 32) + low;
}

// Runs single "round" of Philox algorithm.
inline ulong2 FUNC(calculate_round)(ulong key, ulong2 counter_n) {
    uint2 counter_lr = as_uint2(counter_n[0]);
    uint2 key_lr = as_uint2(key);
    uint2 n_lr = as_uint2(counter_n[1]);

    uint2 prod0 = as_uint2(STATISTIC_MAXIMIZING_MULTIPLIER_N * n_lr[0]);
    uint2 prod1 = as_uint2(STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER * counter_lr[0]);

    n_lr[0] = prod1[1] ^ n_lr[1] ^ key_lr[0];
    n_lr[1] = prod1[0];
    counter_lr[0] = prod0[1] ^ counter_lr[1] ^ key_lr[1];
    counter_lr[1] = prod0[0];

    return (ulong2)(FUNC_CALL(unite_high_low)(counter_lr[1], counter_lr[0]),  FUNC_CALL(unite_high_low)(n_lr[1], n_lr[0]));
}

inline ulong FUNC(raise_key)(uint2 key_lr) {
    key_lr[0] += CRUSH_RESISTANCE_CONST_LOWER_VALUE;
    key_lr[1] += CRUSH_RESISTANCE_CONST_UPPER_VALUE;
    return FUNC_CALL(unite_high_low)(key_lr[1], key_lr[0]);
}

inline uint4 FUNC(run_philox)(ulong n) {
    ulong key = GLOBAL_SEED;
    ulong counter = OP_SEED;
    ulong2 counter_n = {counter, n};
    for (size_t i = 0; i < N_ROUNDS; i++) {
        counter_n = FUNC_CALL(calculate_round)(key, counter_n);
        key = FUNC_CALL(raise_key)(as_uint2(key));
    }
    uint2 res1 = as_uint2(counter_n[1]);
    uint2 res2 = as_uint2(counter_n[0]);
    uint4 res;
    res[0] = res1[0];
    res[1] = res1[1];
    res[2] = res2[0];
    res[3] = res2[1];
    return res;
}


inline float FUNC(uint32_to_float)(uint x) {
    uint out_val = (127u << 23) | (x & 0x7fffffu);
    float float_val = as_float(out_val);
    return float_val - 1.0f;
}

inline void FUNC(fill_float)(const uint4 res,
                float min_val,
                float max_val,
                __global float *output,
                uint output_index) {
    float diff = max_val - min_val;
    for (uint i = 0; i < 4; ++i) {
        if (output_index + i < OUTPUT_LENGTH) {
            output[output_index + i] = FUNC_CALL(uint32_to_float)(res[i]) * diff + min_val;
        }
    }
}

inline void FUNC(fill_int)(const uint4 res,
              int min_val,
              int max_val,
              __global int *output,
              uint output_index) {
    int diff = max_val - min_val;
    for (uint i = 0; i < 4; ++i) {
        if (output_index + i < OUTPUT_LENGTH) {
            output[output_index + i] = (int) (res[i] % diff + min_val);
        }
    }
}

inline void FUNC(fill_long)(const uint4 res,
               long min_val,
               long max_val,
               __global long *output,
               uint output_index) {
    long diff = max_val - min_val;
    output[output_index] = (long)(FUNC_CALL(unite_high_low)(res[1], res[0]) % diff + min_val);
    if (output_index + 1 < OUTPUT_LENGTH) {
        output[output_index + 1] = (long)(FUNC_CALL(unite_high_low)(res[3], res[2]) % diff + min_val);
    }
}

inline half FUNC(uint32_to_float16)(uint x) {
    ushort x_uint16 = (ushort) x;
    ushort out_val = (ushort)(15 << 10) | (x_uint16 & 0x3ffu);
    return as_half(out_val) - 1;
}

inline void FUNC(fill_half)(const uint4 res,
               half min_val,
               half max_val,
               __global half *output,
               uint output_index) {
    half diff = max_val - min_val;
    for (uint i = 0; i < 4; ++i) {
        if (output_index + i < OUTPUT_LENGTH) {
            output[output_index + i] = FUNC_CALL(uint32_to_float16)(res[i]) * diff + min_val;
        }
    }
}

KERNEL(random_uniform_ref)(const __global INPUT0_TYPE* shape, const __global INPUT1_TYPE *min_val,
                            const __global INPUT2_TYPE* max_val, __global OUTPUT_TYPE *output) {
    const uint plain_index = get_global_id(0);
    uint4 result = FUNC_CALL(run_philox)(plain_index);
    FILL_FUNC(FUNC_NAME(OUTPUT_TYPE), result, *min_val, *max_val, output, plain_index * OUTPUT_STEP);
}

#undef FILL_FUNC
#undef FUNC_NAME
#undef FNAME

#undef STATISTIC_MAXIMIZING_MULTIPLIER_N
#undef STATISTIC_MAXIMIZING_MULTIPLIER_COUNTER
#undef CRUSH_RESISTANCE_CONST_LOWER_VALUE
#undef CRUSH_RESISTANCE_CONST_UPPER_VALUE
#undef N_ROUNDS
