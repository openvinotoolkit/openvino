// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// JIT Parameters:
// SIMD                        - sub-group size/simd width, one of {16};
// DECOMPRESSION_GROUP_SIZE    - group size for weight int4 compression
// FILTER_LAYOUT_OS_IS_YX_TYPE - 0: OS_IS_YX_OSV16, 1: OS_IS_YX_OSV32_ISV2, 2: OS_IS_YX_OSV64_ISV2

#define KERNEL_LAYOUT_OS_IS_YX_OSV16      (FILTER_LAYOUT_OS_IS_YX_TYPE == 0)
#define KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2 (FILTER_LAYOUT_OS_IS_YX_TYPE == 1)
#define KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2 (FILTER_LAYOUT_OS_IS_YX_TYPE == 2)
#define SUBGROUP_SIZE                     SIMD
#ifndef DECOMPRESSION_SCALE_GROUP_SIZE
#    define DECOMPRESSION_SCALE_GROUP_SIZE 128
#endif
#define DECOMPRESSION_GROUP_SIZE_SRC DECOMPRESSION_SCALE_GROUP_SIZE

// Verify JIT parameters.
#if SIMD != 16
#    error "fully_connected_gpu_gemv.cl - SIMD must be 16"
#endif

#ifdef DECOMPRESSION_SCALE_TERM
#    if DECOMPRESSION_GROUP_SIZE_SRC < 32
#        error "fully_connected_gpu_gemv.cl - DECOMPRESSION_GROUP_SIZE_SRC must be >= 32"
#    endif
// CW
#    if WEIGHTS_K == DECOMPRESSION_GROUP_SIZE_SRC && WEIGHTS_K > 128
#        define SINGLE_GROUP_NUM
#    endif
#    ifdef SINGLE_GROUP_NUM
#        define SCALE_GROUP_NUM          (WEIGHTS_K / 128)
#        define DECOMPRESSION_GROUP_SIZE 128
#    else
#        define SCALE_GROUP_NUM          (WEIGHTS_K / DECOMPRESSION_GROUP_SIZE_SRC)
#        define DECOMPRESSION_GROUP_SIZE DECOMPRESSION_GROUP_SIZE_SRC
#    endif
#else
#    define SCALE_GROUP_NUM          (WEIGHTS_K / 128)
#    define DECOMPRESSION_GROUP_SIZE 128
#endif

#if KERNEL_LAYOUT_OS_IS_YX_OSV16 && WEIGHTS_K % 32 != 0
#    error "fully_connected_gpu_gemv.cl - KERNEL_LAYOUT_OS_IS_YX_OSV16 must be WEIGHTS_K % 32 != 0"
#endif

#if KERNEL_LAYOUT_OS_IS_YX_OSV16
#    define INPUT_TILE_SIZE 2
#elif KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2
#    define INPUT_TILE_SIZE 1
#elif KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2
#    define INPUT_TILE_SIZE 1
#else
#    error "fully_connected_gpu_gemv.cl - Unsupported layout!"
#endif

// Macros for vectorized types.
#define GEMV_INPUT_VEC_TYPE               MAKE_VECTOR_TYPE(INPUT0_TYPE, INPUT_TILE_SIZE)
#define GEMV_ACCUMULATOR_VEC_TYPE         MAKE_VECTOR_TYPE(float, 8)
#define GEMV_FILTER_VEC_TYPE              MAKE_VECTOR_TYPE(half, 16)
#define GEMV_FILTER_PACKED_VEC_TYPE       MAKE_VECTOR_TYPE(char, 16)
#define GEMV_OUTPUT_VEC_TYPE              MAKE_VECTOR_TYPE(OUTPUT_TYPE, 1)
#define TO_GEMV_OUTPUT_VEC_TYPE(x)        CAT(convert_, GEMV_OUTPUT_VEC_TYPE)(x)
#define TO_GEMV_FILTER_VEC_TYPE(x)        CAT(convert_, GEMV_FILTER_VEC_TYPE)(x)
#define TO_GEMV_FILTER_PACKED_VEC_TYPE(x) CAT(convert_, GEMV_FILTER_PACKED_VEC_TYPE)(x)

#define GEMV_INPUT_BLOCK_READ(ptr, offset)  BLOCK_READN(INPUT0_TYPE, INPUT_TILE_SIZE, ptr, offset)
#define GEMV_FILTER_BLOCK_READ(ptr, offset) BLOCK_READN(FILTER_TYPE, 16, ptr, offset)

inline int FUNC(get_4bit_weight_index)(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (n % OSV) + (k / 2) * OSV;
}

inline int FUNC(get_4bit_weight_index_no_isv)(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (k / 2) * OSV;
}

inline void FUNC(thread_task_splitter)(const int group_num, const int thr_num, const int thr_id, int* n_start, int* n_end) {
    if (thr_num <= 1 || group_num == 0) {
        *n_start = 0;
        *n_end = group_num;
    } else {
        int num = (group_num + thr_num - 1) / thr_num;
        int num_minus = num - 1;
        int last = group_num - num_minus * thr_num;
        *n_end = thr_id < last ? num : num_minus;
        *n_start = thr_id <= last ? thr_id * num : last * num + (thr_id - last) * num_minus;
    }
    *n_end += *n_start;
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(fully_connected_gpu_gemv)(
    OPTIONAL_SHAPE_INFO_ARG __global INPUT0_TYPE* input,
#if DECOMPRESSION_SCALE_TERM
    const __global DECOMPRESSION_SCALE_TYPE* scales,
#endif
#if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global DECOMPRESSION_ZP_TYPE* zps,
#endif
    __global OUTPUT_TYPE* output,
    const __global FILTER_TYPE* weights
#if BIAS_TERM
    ,
    const __global BIAS_TYPE* bias
#endif
#if HAS_FUSED_OPS_DECLS
    ,
    FUSED_OPS_DECLS
#endif
) {
#if KERNEL_LAYOUT_OS_IS_YX_OSV16
    // global:[N, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0);              // N
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int gk0, gk1;
    FUNC_CALL(thread_task_splitter)(SCALE_GROUP_NUM, thr_num, thr_id, &gk0, &gk1);

#    if DECOMPRESSION_ZP_SCALAR
    char zp_scalar_value = (char)(DECOMPRESSION_ZP_VALUE);
#    endif

    __local float all_sum_even[16][16];  // [wi_id, thr_id]
    __local float all_sum_odd[16][16];

    // Scale layout is byfx
    scales += n;
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    zps += n;
#    endif

    float sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
        const __global FILTER_TYPE* B =
            weights + FUNC_CALL(get_4bit_weight_index_no_isv)(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 16);

        GEMV_ACCUMULATOR_VEC_TYPE sum = 0;
#    ifdef SINGLE_GROUP_NUM
        float scale_1 = convert_float(scales[0]);
#    else
        float scale_1 = convert_float(scales[gk * WEIGHTS_N]);
#    endif

#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
#        ifdef SINGLE_GROUP_NUM
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)(zps[0]);
#        else
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)(zps[gk * WEIGHTS_N]);
#        endif
#    elif DECOMPRESSION_ZP_SCALAR
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)(zp_scalar_value);
#    else
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)0;
#    endif

        __attribute__((opencl_unroll_hint(4))) for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 32, B += 16 * 16) {
            GEMV_INPUT_VEC_TYPE input_value = GEMV_INPUT_BLOCK_READ(A, g);
            GEMV_FILTER_PACKED_VEC_TYPE bx16 = TO_GEMV_FILTER_PACKED_VEC_TYPE(GEMV_FILTER_BLOCK_READ(B, 0));

#    if WEI_UINT4
            GEMV_FILTER_VEC_TYPE i4x16_even = TO_GEMV_FILTER_VEC_TYPE((bx16 & (char16)0xF)) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = TO_GEMV_FILTER_VEC_TYPE(as_char16(as_uchar16(bx16) >> 4)) - zpx16;
#    else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7);
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7);
            GEMV_FILTER_VEC_TYPE i4x16_even = TO_GEMV_FILTER_VEC_TYPE(i4x16_even_c16) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = TO_GEMV_FILTER_VEC_TYPE(i4x16_odd_c16) - zpx16;
#    endif

            sum[0] += as_half(sub_group_broadcast(input_value.s0, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value.s0, 4)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value.s0, 8)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value.s0, 12)) * i4x16_even.s6;
            sum[1] += as_half(sub_group_broadcast(input_value.s0, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value.s0, 5)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value.s0, 9)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value.s0, 13)) * i4x16_odd.s6;

            sum[2] += as_half(sub_group_broadcast(input_value.s0, 2)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value.s0, 6)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value.s0, 10)) * i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value.s0, 14)) * i4x16_even.s7;
            sum[3] += as_half(sub_group_broadcast(input_value.s0, 3)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value.s0, 7)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value.s0, 11)) * i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value.s0, 15)) * i4x16_odd.s7;

            sum[4] += as_half(sub_group_broadcast(input_value.s1, 0)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value.s1, 4)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value.s1, 8)) * i4x16_even.sc +
                      as_half(sub_group_broadcast(input_value.s1, 12)) * i4x16_even.se;
            sum[5] += as_half(sub_group_broadcast(input_value.s1, 1)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value.s1, 5)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value.s1, 9)) * i4x16_odd.sc +
                      as_half(sub_group_broadcast(input_value.s1, 13)) * i4x16_odd.se;

            sum[6] += as_half(sub_group_broadcast(input_value.s1, 2)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value.s1, 6)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value.s1, 10)) * i4x16_even.sd +
                      as_half(sub_group_broadcast(input_value.s1, 14)) * i4x16_even.sf;
            sum[7] += as_half(sub_group_broadcast(input_value.s1, 3)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value.s1, 7)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value.s1, 11)) * i4x16_odd.sd +
                      as_half(sub_group_broadcast(input_value.s1, 15)) * i4x16_odd.sf;
        }

        sum_all += (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5] + sum[6] + sum[7]) * scale_1;
    }

    all_sum_even[wi_id][thr_id] = sum_all;
    barrier(CLK_LOCAL_MEM_FENCE);

    float2 sum_value;
    sum_value[0] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_even[thr_id]));
    sum_value[0] = sub_group_reduce_add(sum_value[0]);
    if (wi_id == 0) {
        int cur_n = n + thr_id;
#    if BIAS_TERM
        sum_value[0] += bias[cur_n];
#    endif
#    if HAS_FUSED_OPS
        for (int i = 0; i < 1; i++) {
            FUSED_OPS_VEC
            output[cur_n + i] = FUSED_OPS_RESULT_VEC;
        }
#    else
        for (int i = 0; i < 1; i++) {
            output[cur_n + i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
        }
#    endif
    }
}

#elif KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2
    // global:[N//2, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0) * 2;          // N
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int gk0, gk1;
    FUNC_CALL(thread_task_splitter)(SCALE_GROUP_NUM, thr_num, thr_id, &gk0, &gk1);

#    if DECOMPRESSION_ZP_SCALAR
    char zp_scalar_value = (char)(DECOMPRESSION_ZP_VALUE);
#    endif

    __local float all_sum_even[16][16];  // [wi_id, thr_id]
    __local float all_sum_odd[16][16];

    // Scale layout is fbyx
    scales += (n / 32) * 32 + (n % 32) / 2;
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    zps += (n / 32) * 32 + (n % 32) / 2;
#    endif

    float2 sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
        const __global FILTER_TYPE* B =
            weights + FUNC_CALL(get_4bit_weight_index)(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 32);

        GEMV_ACCUMULATOR_VEC_TYPE sum = 0;
#    ifdef SINGLE_GROUP_NUM
        float scale_0 = convert_float(scales[0]);
        float scale_1 = convert_float(scales[16]);
#    else
        float scale_0 = convert_float(scales[gk * WEIGHTS_N]);
        float scale_1 = convert_float(scales[gk * WEIGHTS_N + 16]);
#    endif

#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
#        ifdef SINGLE_GROUP_NUM
        half zp0 = zps[0];
        half zp1 = zps[16];
#        else
        half zp0 = zps[gk * WEIGHTS_N];
        half zp1 = zps[gk * WEIGHTS_N + 16];
#        endif
        GEMV_FILTER_VEC_TYPE zpx16 = {zp0, zp1, zp0, zp1, zp0, zp1, zp0, zp1, zp0, zp1, zp0, zp1, zp0, zp1, zp0, zp1};
#    elif DECOMPRESSION_ZP_SCALAR
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)(zp_scalar_value);
#    else
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)0;
#    endif

        __attribute__((opencl_unroll_hint(4))) for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 16, B += 16 * 16) {
            // read 16 elements of A
            GEMV_INPUT_VEC_TYPE input_value = GEMV_INPUT_BLOCK_READ(A, g);

            // read 16x16 int8 = (16x2)x16 int4

            GEMV_FILTER_PACKED_VEC_TYPE bx16 = TO_GEMV_FILTER_PACKED_VEC_TYPE(GEMV_FILTER_BLOCK_READ(B, 0));

#    if WEI_UINT4
            GEMV_FILTER_VEC_TYPE i4x16_even = TO_GEMV_FILTER_VEC_TYPE(bx16 & (char16)0xF) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = TO_GEMV_FILTER_VEC_TYPE(as_char16(as_uchar16(bx16) >> 4)) - zpx16;
#    else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7);
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7);
            GEMV_FILTER_VEC_TYPE i4x16_even = TO_GEMV_FILTER_VEC_TYPE(i4x16_even_c16) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = TO_GEMV_FILTER_VEC_TYPE(i4x16_odd_c16) - zpx16;
#    endif

            sum[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.sc;

            sum[1] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even.sd;

            sum[2] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.sc;

            sum[3] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd.sd;

            sum[4] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s6 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.se;

            sum[5] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.s7 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even.sf;

            sum[6] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s6 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.se;

            sum[7] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.s7 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd.sf;
        }

        sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale_0;
        sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale_1;
    }

    all_sum_even[wi_id][thr_id] = sum_all[0];
    all_sum_odd[wi_id][thr_id] = sum_all[1];
    barrier(CLK_LOCAL_MEM_FENCE);

    float2 sum_value;
    sum_value[0] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_even[thr_id]));
    sum_value[1] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_odd[thr_id]));
    sum_value[0] = sub_group_reduce_add(sum_value[0]);
    sum_value[1] = sub_group_reduce_add(sum_value[1]);

    if (wi_id == 0) {
        int cur_n = n + thr_id;

        // bias
#    if BIAS_TERM
        sum_value[0] += bias[cur_n];
        sum_value[1] += bias[cur_n + 16];
#    endif

// fused_op
#    if HAS_FUSED_OPS
        for (int i = 0; i < 2; i++) {
            FUSED_OPS_VEC
            output[cur_n + 16 * i] = FUSED_OPS_RESULT_VEC;
        }
#    else
        for (int i = 0; i < 2; i++) {
            output[cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
        }
#    endif
    }
}
#elif KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2
    // global:[N//4, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0) * 4;          // N
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int gk0, gk1;
    FUNC_CALL(thread_task_splitter)(SCALE_GROUP_NUM, thr_num, thr_id, &gk0, &gk1);

    __local float all_sum_0[16][16];  // [wi_id, thr_id]
    __local float all_sum_1[16][16];  // [wi_id, thr_id]
    __local float all_sum_2[16][16];  // [wi_id, thr_id]
    __local float all_sum_3[16][16];  // [wi_id, thr_id]

    scales += (n / 64) * 64 + (n % 64) / 4;
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    zps += (n / 64) * 64 + (n % 64) / 4;
#    endif

    float4 sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global INPUT0_TYPE* A = input + gk * DECOMPRESSION_GROUP_SIZE;
        const __global FILTER_TYPE* B =
            weights + FUNC_CALL(get_4bit_weight_index)(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 64);

        GEMV_ACCUMULATOR_VEC_TYPE sum = 0;
#    ifdef SINGLE_GROUP_NUM
        float scale_0 = convert_float(scales[0]);
        float scale_1 = convert_float(scales[16]);
        float scale_2 = convert_float(scales[2 * 16]);
        float scale_3 = convert_float(scales[3 * 16]);
#    else
        float scale_0 = convert_float(scales[gk * WEIGHTS_N]);
        float scale_1 = convert_float(scales[gk * WEIGHTS_N + 1 * 16]);
        float scale_2 = convert_float(scales[gk * WEIGHTS_N + 2 * 16]);
        float scale_3 = convert_float(scales[gk * WEIGHTS_N + 3 * 16]);
#    endif
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
#        ifdef SINGLE_GROUP_NUM
        half zp0 = zps[0];
        half zp1 = zps[1 * 16];
        half zp2 = zps[2 * 16];
        half zp3 = zps[3 * 16];
#        else
        half zp0 = zps[gk * WEIGHTS_N];
        half zp1 = zps[gk * WEIGHTS_N + 1 * 16];
        half zp2 = zps[gk * WEIGHTS_N + 2 * 16];
        half zp3 = zps[gk * WEIGHTS_N + 3 * 16];
#        endif
        GEMV_FILTER_VEC_TYPE zpx16 = {zp0, zp1, zp2, zp3, zp0, zp1, zp2, zp3, zp0, zp1, zp2, zp3, zp0, zp1, zp2, zp3};
#    elif DECOMPRESSION_ZP_SCALAR
        half zp_scalar_value = (half)(DECOMPRESSION_ZP_VALUE);
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)(zp_scalar_value);
#    else
        GEMV_FILTER_VEC_TYPE zpx16 = (GEMV_FILTER_VEC_TYPE)0;
#    endif

        __attribute__((opencl_unroll_hint(2))) for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 16, B += 16 * 32) {
            // read 16 elements of A
            GEMV_INPUT_VEC_TYPE input_value = GEMV_INPUT_BLOCK_READ(A, g);
            GEMV_FILTER_PACKED_VEC_TYPE bx16 = TO_GEMV_FILTER_PACKED_VEC_TYPE(GEMV_FILTER_BLOCK_READ(B, 0));
            GEMV_FILTER_PACKED_VEC_TYPE bx16_second =
                TO_GEMV_FILTER_PACKED_VEC_TYPE(GEMV_FILTER_BLOCK_READ(B, 16 * 16));

#    if WEI_UINT4
            GEMV_FILTER_VEC_TYPE i4x16_even = convert_half16((bx16 & (char16)0xF)) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4)) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_even_second = convert_half16((bx16_second & (char16)0xF)) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd_second = convert_half16(as_char16(as_uchar16(bx16_second) >> 4)) - zpx16;
#    else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7);
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7);

            char16 i4x16_even_c16_second = (bx16_second & (char16)0xF);
            char16 i4x16_odd_c16_second = (as_char16(as_uchar16(bx16_second) >> 4));
            i4x16_even_c16_second =
                select(i4x16_even_c16_second, i4x16_even_c16_second - (char16)16, i4x16_even_c16_second > (char16)7);
            i4x16_odd_c16_second =
                select(i4x16_odd_c16_second, i4x16_odd_c16_second - (char16)16, i4x16_odd_c16_second > (char16)7);

            GEMV_FILTER_VEC_TYPE i4x16_even = convert_half16(i4x16_even_c16) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd = convert_half16(i4x16_odd_c16) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_even_second = convert_half16(i4x16_even_c16_second) - zpx16;
            GEMV_FILTER_VEC_TYPE i4x16_odd_second = convert_half16(i4x16_odd_c16_second) - zpx16;
#    endif

            sum[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.sc;
            sum[0] += as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s0 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.s4 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.s8 +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.sc;
            sum[1] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.sd;
            sum[1] += as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s1 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.s5 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.s9 +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.sd;
            sum[2] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s6 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.se;
            sum[2] += as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s2 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.s6 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.sa +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.se;
            sum[3] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s7 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even.sf;
            sum[3] += as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s3 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.s7 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.sb +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.sf;
            sum[4] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.sc;
            sum[4] += as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s0 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.s4 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.s8 +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.sc;
            sum[5] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.sd;
            sum[5] += as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s1 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.s5 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.s9 +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.sd;
            sum[6] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s6 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.se;
            sum[6] += as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s2 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.s6 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.sa +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.se;
            sum[7] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd.s7 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd.sf;
            sum[7] += as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s3 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.s7 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.sb +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.sf;
        }

        sum_all[0] += (sum[0] + sum[4]) * scale_0;
        sum_all[1] += (sum[1] + sum[5]) * scale_1;
        sum_all[2] += (sum[2] + sum[6]) * scale_2;
        sum_all[3] += (sum[3] + sum[7]) * scale_3;
    }

    all_sum_0[wi_id][thr_id] = sum_all[0];
    all_sum_1[wi_id][thr_id] = sum_all[1];
    all_sum_2[wi_id][thr_id] = sum_all[2];
    all_sum_3[wi_id][thr_id] = sum_all[3];
    barrier(CLK_LOCAL_MEM_FENCE);

    float4 sum_value;
    sum_value[0] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_0[thr_id]));
    sum_value[1] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_1[thr_id]));
    sum_value[2] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_2[thr_id]));
    sum_value[3] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_3[thr_id]));

    for (int i = 0; i < 4; i++) {
        sum_value[i] = sub_group_reduce_add(sum_value[i]);
    }

    if (wi_id == 0) {
        int cur_n = n + thr_id;
#    if BIAS_TERM
        for (int i = 0; i < 4; i++) {
            sum_value[i] += bias[cur_n + 16 * i];
        }
#    endif
#    if HAS_FUSED_OPS
        for (int i = 0; i < 4; i++) {
            FUSED_OPS_VEC
            output[cur_n + 16 * i] = FUSED_OPS_RESULT_VEC;
        }
#    else
        for (int i = 0; i < 4; i++) {
            output[cur_n + 16 * i] = TO_GEMV_OUTPUT_VEC_TYPE(ACTIVATION_TYPED(sum_value[i], ACTIVATION_PARAMS_TYPED));
        }
#    endif
    }
}
#endif

#undef INPUT_TILE_SIZE
#undef GEMV_FILTER_BLOCK_READ
#undef GEMV_INPUT_BLOCK_READ
#undef TO_GEMV_FILTER_PACKED_VEC_TYPE
#undef TO_GEMV_FILTER_VEC_TYPE
#undef TO_GEMV_OUTPUT_VEC_TYPE
#undef GEMV_OUTPUT_VEC_TYPE
#undef GEMV_FILTER_PACKED_VEC_TYPE
#undef GEMV_ACCUMULATOR_VEC_TYPE
#undef GEMV_INPUT_VEC_TYPE
#undef SUBGROUP_SIZE
#undef KERNEL_LAYOUT_OS_IS_YX_OSV16
#undef KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2
#undef KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2
#undef DECOMPRESSION_GROUP_SIZE_SRC