// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "include/batch_headers/common.cl"
#include "include/batch_headers/sub_group_block_read.cl"
#include "include/batch_headers/sub_group_block_write.cl"
#include "include/batch_headers/sub_group_shuffle.cl"

// JIT Parameters:
// SIMD         - sub-group size/simd width, one of {16};
// DECOMPRESSION_GROUP_SIZE    - group size for weight int4 compression
// FILTER_LAYOUT_OS_IS_YX_TYPE - 0: OS_IS_YX_OSV16, 1: OS_IS_YX_OSV32_ISV2, 2: OS_IS_YX_OSV32_ISV2


#define KERNEL_LAYOUT_OS_IS_YX_OSV16  (FILTER_LAYOUT_OS_IS_YX_TYPE == 0)
#define KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2 (FILTER_LAYOUT_OS_IS_YX_TYPE == 1)
#define KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2 (FILTER_LAYOUT_OS_IS_YX_TYPE == 2)
#define SUBGROUP_SIZE                     SIMD
#define DECOMPRESSION_GROUP_SIZE          DECOMPRESSION_SCALE_GROUP_SIZE

// Verify JIT parameters.
#if SIMD != 16
#    error "fully_connected_gpu_gemv.cl - SIMD must be 16"
#endif

#if DECOMPRESSION_GROUP_SIZE < 32
#   error "fully_connected_gpu_gemv.cl - DECOMPRESSION_GROUP_SIZE must >= 32"
#endif

#if KERNEL_LAYOUT_OS_IS_YX_OSV16 && WEIGHTS_K % 32 != 0
#   error "fully_connected_gpu_gemv.cl - KERNEL_LAYOUT_OS_IS_YX_OSV16 must be WEIGHTS_K % 32 != 0"
#endif

inline int get_4bit_weight_index(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (n % OSV) + (k / 2) * OSV;
}

inline int get_4bit_weight_index_no_isv(int k, int n, int K, int N, int OSV) {
    return (n / OSV) * (OSV * K / 2) + (k / 2) * OSV;
}

inline void thread_task_splitter(const int group_num, const int thr_num, const int thr_id, int* n_start, int* n_end) {
    if (thr_num <= 1 || group_num == 0) {
        *n_start = 0;
        *n_end = group_num;
    } else {
        int n1 = (group_num + thr_num - 1) / thr_num;
        int n2 = n1 - 1;
        int last = group_num - n2 * thr_num;
        *n_end = thr_id < last ? n1 : n2;
        *n_start = thr_id <= last ? thr_id * n1 : last * n1 + (thr_id - last) * n2;
    }
    *n_end += *n_start;
}

#define SCALE_GROUP_NUM (WEIGHTS_K / DECOMPRESSION_GROUP_SIZE)

#if KERNEL_LAYOUT_OS_IS_YX_OSV16
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(fully_connected_gpu_gemv)(
    OPTIONAL_SHAPE_INFO_ARG
    __global half* input,
#    if DECOMPRESSION_SCALE_TERM
    const __global half* scales,
#    endif
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global char* zps,
#    endif
    __global half* output,
    const __global uchar* weights
#    if BIAS_TERM
    , const __global half* bias
#    endif
#    if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#    endif
) {
    // global:[N, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0);              // N
    int m = get_global_id(1);              // M==1
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int group_num = WEIGHTS_K / DECOMPRESSION_GROUP_SIZE;
    int gk0, gk1;
    thread_task_splitter(group_num, thr_num, thr_id, &gk0, &gk1);

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

    // if(n==0 && get_global_id(2)==0) {
    //     #if HAS_FUSED_OPS
    //     printf("HAS_FUSED_OPS...\n");
    //     #endif
    //     #if HAS_FUSED_OPS_DECLS
    //     printf("HAS_FUSED_OPS_DECLS...\n");
    //     #endif
    //     #if BIAS_TERM
    //     printf("BIAS_TERM...\n");
    //     #endif
    //     #if DECOMPRESSION_SCALE_TERM
    //     printf("DECOMPRESSION_SCALE_TERM...\n");
    //     #endif
    //     #if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    //     printf("DECOMPRESSION_ZP_TERM...\n");
    //     #endif
    //     printf("group_num = %d, WEIGHTS_K = %d, WEIGHTS_K = %d\n", group_num, WEIGHTS_K, WEIGHTS_N);
    //     #if DECOMPRESSION_ZP_SCALAR
    //     printf("zp_scalar_value = %d\n", zp_scalar_value);
    //     #endif
    // }

    float sum_all = 0;
    for (int gk = gk0; gk < gk1; gk++) {
        __global half* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
        const __global uchar* B =
            weights + get_4bit_weight_index_no_isv(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 16);

        float8 sum = 0;
        float scale_1 = convert_float(scales[gk * WEIGHTS_N]);

#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        char16 zpx16 = (char16)(zps[gk * WEIGHTS_N]);
#    elif DECOMPRESSION_ZP_SCALAR
        char16 zpx16 = (char16)(zp_scalar_value);
#    else
        char16 zpx16 = (char16)0;
#    endif
        char16 mask16 = (char16)0xF;

        __attribute__((opencl_unroll_hint(4))) for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 32, B += 16 * 16) {
            // read 16 elements of A
            ushort2 input_value = intel_sub_group_block_read_us2((const __global ushort*)(A + g));            
            char16 bx16 = as_char16(intel_sub_group_block_read_uc16(B));

#if WEI_UINT4
            half16 i4x16_even = convert_half16((bx16 & mask16) - zpx16);
            half16 i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4) - zpx16);
#else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7) - zpx16;
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7) - zpx16;
            half16 i4x16_even = convert_half16(i4x16_even_c16);
            half16 i4x16_odd = convert_half16(i4x16_odd_c16);
#endif

            sum[0] += as_half(sub_group_broadcast(input_value.s0, 0)) * i4x16_even.s0 +
                      as_half(sub_group_broadcast(input_value.s0, 4)) * i4x16_even.s2 +
                      as_half(sub_group_broadcast(input_value.s0, 8)) * i4x16_even.s4 +
                      as_half(sub_group_broadcast(input_value.s0, 12))* i4x16_even.s6;
            sum[1] += as_half(sub_group_broadcast(input_value.s0, 1)) * i4x16_odd.s0 +
                      as_half(sub_group_broadcast(input_value.s0, 5)) * i4x16_odd.s2 +
                      as_half(sub_group_broadcast(input_value.s0, 9)) * i4x16_odd.s4 +
                      as_half(sub_group_broadcast(input_value.s0, 13))* i4x16_odd.s6;

            sum[2] += as_half(sub_group_broadcast(input_value.s0, 2)) * i4x16_even.s1 +
                      as_half(sub_group_broadcast(input_value.s0, 6)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value.s0, 10))* i4x16_even.s5 +
                      as_half(sub_group_broadcast(input_value.s0, 14))* i4x16_even.s7;
            sum[3] += as_half(sub_group_broadcast(input_value.s0, 3)) * i4x16_odd.s1 +
                      as_half(sub_group_broadcast(input_value.s0, 7)) * i4x16_odd.s3 +
                      as_half(sub_group_broadcast(input_value.s0, 11))* i4x16_odd.s5 +
                      as_half(sub_group_broadcast(input_value.s0, 15))* i4x16_odd.s7;

            sum[4] += as_half(sub_group_broadcast(input_value.s1, 0)) * i4x16_even.s8 +
                      as_half(sub_group_broadcast(input_value.s1, 4)) * i4x16_even.sa +
                      as_half(sub_group_broadcast(input_value.s1, 8)) * i4x16_even.sc +
                      as_half(sub_group_broadcast(input_value.s1, 12))* i4x16_even.se;
            sum[5] += as_half(sub_group_broadcast(input_value.s1, 1)) * i4x16_odd.s8 +
                      as_half(sub_group_broadcast(input_value.s1, 5)) * i4x16_odd.sa +
                      as_half(sub_group_broadcast(input_value.s1, 9)) * i4x16_odd.sc +
                      as_half(sub_group_broadcast(input_value.s1, 13))* i4x16_odd.se;

            sum[6] += as_half(sub_group_broadcast(input_value.s1, 2)) * i4x16_even.s9 +
                      as_half(sub_group_broadcast(input_value.s1, 6)) * i4x16_even.sb +
                      as_half(sub_group_broadcast(input_value.s1, 10))* i4x16_even.sd +
                      as_half(sub_group_broadcast(input_value.s1, 14))* i4x16_even.sf;
            sum[7] += as_half(sub_group_broadcast(input_value.s1, 3)) * i4x16_odd.s9 +
                      as_half(sub_group_broadcast(input_value.s1, 7)) * i4x16_odd.sb +
                      as_half(sub_group_broadcast(input_value.s1, 11))* i4x16_odd.sd +
                      as_half(sub_group_broadcast(input_value.s1, 15))* i4x16_odd.sf;
        }

        // scales applied once
        sum_all += (sum[0] + sum[1] + sum[2] + sum[3] + sum[4] + sum[5]  + sum[6]  + sum[7]) * scale_1;
    }

    all_sum_even[wi_id][thr_id] = sum_all;
    barrier(CLK_LOCAL_MEM_FENCE);

    float2 sum_value;
    sum_value[0] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_even[thr_id]));
    sum_value[0] = sub_group_reduce_add(sum_value[0]);
    if (wi_id == 0)
    {
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
            output[cur_n + i] = sum_value[i];
        }
#    endif
    }
}

#elif KERNEL_LAYOUT_OS_IS_YX_OSV32_ISV2
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(fully_connected_gpu_gemv)(
    OPTIONAL_SHAPE_INFO_ARG __global half* input,
#    if DECOMPRESSION_SCALE_TERM
    const __global half* scales,
#    endif
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global char* zps,
#    endif
    __global half* output,
    const __global uchar* weights
#    if BIAS_TERM
    , const __global half* bias
#    endif
#    if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#    endif
) {
    // global:[N//2, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0) * 2;          // N
    int m = get_global_id(1);              // M==1
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int group_num = WEIGHTS_K / DECOMPRESSION_GROUP_SIZE;
    int gk0, gk1;
    thread_task_splitter(group_num, thr_num, thr_id, &gk0, &gk1);

#    if DECOMPRESSION_ZP_SCALAR
    char zp_scalar_value = (char)(DECOMPRESSION_ZP_VALUE);
#    endif

    __local float all_sum_even[16][16];  // [wi_id, thr_id]
    __local float all_sum_odd[16][16];
#ifdef SWIGLU_LENGTH
    __local float all_sum_even_second[16][16];  // [wi_id, thr_id]
    __local float all_sum_odd_second[16][16];
#endif

    // Scale layout is byfx
    scales += (n / 32) * 32 + (n % 32) / 2;
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    zps += (n / 32) * 32 + (n % 32) / 2;
#    endif

#ifdef SWIGLU_LENGTH
    float4 sum_all = 0;
#else
    float2 sum_all = 0;
#endif
    for (int gk = gk0; gk < gk1; gk++) {
        __global half* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
        const __global uchar* B =
            weights + get_4bit_weight_index(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 32);

#ifdef SWIGLU_LENGTH
        float8 sum_second = 0;
#endif
        float8 sum = 0;
        half scale_0 = scales[gk * WEIGHTS_N];
        half scale_1 = scales[gk * WEIGHTS_N + 16];

#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        char16 zpx16_0 = (char16)(zps[gk * WEIGHTS_N]);
        char16 zpx16_1 = (char16)(zps[gk * WEIGHTS_N + 16]);
#    elif DECOMPRESSION_ZP_SCALAR
        char16 zpx16_0 = (char16)(zp_scalar_value);
        char16 zpx16_1 = (char16)(zp_scalar_value);
#    else
        char16 zpx16_0 = (char16)0;
        char16 zpx16_1 = (char16)0;
#    endif
        char16 mask16 = (char16)0xF;

#ifdef SWIGLU_LENGTH
        half scale_2 = scales[gk * WEIGHTS_N + SWIGLU_LENGTH];
        half scale_3 = scales[gk * WEIGHTS_N + SWIGLU_LENGTH + 16];
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        char16 zpx16_2 = (char16)(zps[gk * WEIGHTS_N + SWIGLU_LENGTH]);
        char16 zpx16_3 = (char16)(zps[gk * WEIGHTS_N + SWIGLU_LENGTH + 16]);
#    elif DECOMPRESSION_ZP_SCALAR
        char16 zpx16_2 = (char16)(zp_scalar_value);
        char16 zpx16_3 = (char16)(zp_scalar_value);
#    else
        char16 zpx16_2 = (char16)0;
        char16 zpx16_3 = (char16)0;
#    endif
#endif

#ifdef SWIGLU_LENGTH
        __attribute__((opencl_unroll_hint(2)))
#else
        __attribute__((opencl_unroll_hint(4)))
#endif
        for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 16, B += 16 * 16) {
            // read 16 elements of A
            ushort input_value = intel_sub_group_block_read_us((const __global ushort*)(A + g));

            // read 16x16 int8 = (16x2)x16 int4
            char16 bx16 = as_char16(intel_sub_group_block_read_uc16(B));

#ifdef SWIGLU_LENGTH
            char16 bx16_second = as_char16(intel_sub_group_block_read_uc16(B + SWIGLU_LENGTH * WEIGHTS_K));
#endif

#if WEI_UINT4
            half16 i4x16_even = convert_half16((bx16 & mask16) - zpx16_0);
            half16 i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4) - zpx16_0);
#else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7) - zpx16_0;
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7) - zpx16_1;
            half16 i4x16_even = convert_half16(i4x16_even_c16);
            half16 i4x16_odd = convert_half16(i4x16_odd_c16);
#endif

#ifdef SWIGLU_LENGTH
#if WEI_UINT4
            half16 i4x16_even_second = (bx16 & (char16)0xF) - zpx16_2;
            half16 i4x16_odd_second = (as_char16(as_uchar16(bx16) >> 4)) - zpx16_3;
#else
            char16 i4x16_even_c16_second = (bx16_second & (char16)0xF);
            char16 i4x16_odd_c16_second = (as_char16(as_uchar16(bx16_second) >> 4));
            i4x16_even_c16_second = select(i4x16_even_c16_second, i4x16_even_c16_second - (char16)16, i4x16_even_c16_second > (char16)7) - zpx16_2;
            i4x16_odd_c16_second = select(i4x16_odd_c16_second, i4x16_odd_c16_second - (char16)16, i4x16_odd_c16_second > (char16)7) - zpx16_3;
            half16 i4x16_even_second = convert_half16(i4x16_even_c16_second);
            half16 i4x16_odd_second = convert_half16(i4x16_odd_c16_second);
#endif
#endif

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

#ifdef SWIGLU_LENGTH
            sum_second[0] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even_second.s0 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even_second.s4 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s8 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.sc;
            sum_second[1] += as_half(sub_group_broadcast(input_value, 0)) * i4x16_even_second.s1 +
                      as_half(sub_group_broadcast(input_value, 4)) * i4x16_even_second.s5 +
                      as_half(sub_group_broadcast(input_value, 8)) * i4x16_even_second.s9 +
                      as_half(sub_group_broadcast(input_value, 12)) * i4x16_even_second.sd;
            sum_second[2] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd_second.s0 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd_second.s4 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s8 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.sc;
            sum_second[3] += as_half(sub_group_broadcast(input_value, 1)) * i4x16_odd_second.s1 +
                      as_half(sub_group_broadcast(input_value, 5)) * i4x16_odd_second.s5 +
                      as_half(sub_group_broadcast(input_value, 9)) * i4x16_odd_second.s9 +
                      as_half(sub_group_broadcast(input_value, 13)) * i4x16_odd_second.sd;
            sum_second[4] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even_second.s2 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even_second.s6 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.sa +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.se;
            sum_second[5] += as_half(sub_group_broadcast(input_value, 2)) * i4x16_even.s3 +
                      as_half(sub_group_broadcast(input_value, 6)) * i4x16_even_second.s7 +
                      as_half(sub_group_broadcast(input_value, 10)) * i4x16_even_second.sb +
                      as_half(sub_group_broadcast(input_value, 14)) * i4x16_even_second.sf;
            sum_second[6] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd_second.s2 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd_second.s6 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.sa +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.se;
            sum_second[7] += as_half(sub_group_broadcast(input_value, 3)) * i4x16_odd_second.s3 +
                      as_half(sub_group_broadcast(input_value, 7)) * i4x16_odd_second.s7 +
                      as_half(sub_group_broadcast(input_value, 11)) * i4x16_odd_second.sb +
                      as_half(sub_group_broadcast(input_value, 15)) * i4x16_odd_second.sf;
#endif
        }

        // scales applied once
        sum_all[0] += (sum[0] + sum[2] + sum[4] + sum[6]) * scale_0;
        sum_all[1] += (sum[1] + sum[3] + sum[5] + sum[7]) * scale_1;
#ifdef SWIGLU_LENGTH
        sum_all[2] += (sum_second[0] + sum_second[2] + sum_second[4] + sum_second[6]) * scale_2;
        sum_all[3] += (sum_second[1] + sum_second[3] + sum_second[5] + sum_second[7]) * scale_3;
#endif
    }

    all_sum_even[wi_id][thr_id] = sum_all[0];
    all_sum_odd[wi_id][thr_id] = sum_all[1];
#ifdef SWIGLU_LENGTH
    all_sum_even_second[wi_id][thr_id] = sum_all[2];
    all_sum_odd_second[wi_id][thr_id] = sum_all[3];
#endif
    barrier(CLK_LOCAL_MEM_FENCE);

#ifdef SWIGLU_LENGTH
    float4 sum_value;
#else
    float2 sum_value;
#endif
    sum_value[0] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_even[thr_id]));
    sum_value[1] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_odd[thr_id]));
    sum_value[0] = sub_group_reduce_add(sum_value[0]);
    sum_value[1] = sub_group_reduce_add(sum_value[1]);

#ifdef SWIGLU_LENGTH
    sum_value[2] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_even_second[thr_id]));
    sum_value[3] = as_float(intel_sub_group_block_read((const __local uint*)all_sum_odd_second[thr_id]));
    sum_value[2] = sub_group_reduce_add(sum_value[2]);
    sum_value[3] = sub_group_reduce_add(sum_value[3]);
#endif


    if (wi_id == 0) {
        int cur_n = n + thr_id;
#ifdef SWIGLU_LENGTH
        sum_value[0] = sum_value[0] * sum_value[2] / (1.0f + native_exp(-(1.0f * sum_value[0])));
        sum_value[1] = sum_value[1] * sum_value[3] / (1.0f + native_exp(-(1.0f * sum_value[1])));
#endif

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
            output[cur_n + 16 * i] = sum_value[i];
        }
#    endif
    }
}
#elif KERNEL_LAYOUT_OS_IS_YX_OSV64_ISV2
__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE))) KERNEL(fully_connected_gpu_gemv)(
    OPTIONAL_SHAPE_INFO_ARG __global half* input,
#    if DECOMPRESSION_SCALE_TERM
    const __global half* scales,
#    endif
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
    const __global char* zps,
#    endif
    __global half* output,
    const __global uchar* weights
#    if BIAS_TERM
    , const __global half* bias
#    endif
#    if HAS_FUSED_OPS_DECLS
    , FUSED_OPS_DECLS
#    endif
) {
    // global:[N//4, M, 16]
    // local: [16, 1, 16]
    int n = get_global_id(0) * 4;          // N
    int m = get_global_id(1);              // M==1
    int thr_id = get_local_id(2);          // 0~15
    int thr_num = get_local_size(2);       // 16
    int wi_id = get_sub_group_local_id();  // 0~15

    int group_num = WEIGHTS_K / DECOMPRESSION_GROUP_SIZE;
    int gk0, gk1;
    thread_task_splitter(group_num, thr_num, thr_id, &gk0, &gk1);

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
        __global half* A = input + m * WEIGHTS_K + gk * DECOMPRESSION_GROUP_SIZE;
        const __global uchar* B =
            weights + get_4bit_weight_index(gk * DECOMPRESSION_GROUP_SIZE, n, WEIGHTS_K, WEIGHTS_N, 64);

        float8 sum = 0;
        half scale_0 = scales[gk * WEIGHTS_N];
        half scale_1 = scales[gk * WEIGHTS_N + 1 * 16];
        half scale_2 = scales[gk * WEIGHTS_N + 2 * 16];
        half scale_3 = scales[gk * WEIGHTS_N + 3 * 16];
#    if DECOMPRESSION_ZP_TERM && !DECOMPRESSION_ZP_SCALAR
        char16 zpx16_0 = (char16)(zps[k * WEIGHTS_N]);
        char16 zpx16_1 = (char16)(zps[k * WEIGHTS_N + 1 * 16]);
        char16 zpx16_2 = (char16)(zps[k * WEIGHTS_N + 2 * 16]);
        char16 zpx16_3 = (char16)(zps[k * WEIGHTS_N + 3 * 16]);
#    elif DECOMPRESSION_ZP_SCALAR
        char zp_scalar_value = (char)(DECOMPRESSION_ZP_VALUE);
        char16 zpx16_0 = (char16)(zp_scalar_value);
        char16 zpx16_1 = (char16)(zp_scalar_value);
        char16 zpx16_2 = (char16)(zp_scalar_value);
        char16 zpx16_3 = (char16)(zp_scalar_value);
#    else
        char16 zpx16_0 = (char16)0;
        char16 zpx16_1 = (char16)0;
        char16 zpx16_2 = (char16)0;
        char16 zpx16_3 = (char16)0;
#    endif
        char16 mask16 = (char16)0xF;

        __attribute__((opencl_unroll_hint(2))) for (int g = 0; g < DECOMPRESSION_GROUP_SIZE; g += 16, B += 16 * 32) {
            // read 16 elements of A
            ushort input_value = intel_sub_group_block_read_us((const __global ushort*)(A + g));
            char16 bx16 = as_char16(intel_sub_group_block_read_uc16(B));
            char16 bx16_second = as_char16(intel_sub_group_block_read_uc16(B + 16 * 16));

#if WEI_UINT4
            half16 i4x16_even = convert_half16((bx16 & mask16) - zpx16_0);
            half16 i4x16_odd = convert_half16(as_char16(as_uchar16(bx16) >> 4) - zpx16_1);
            half16 i4x16_even_second = convert_half16((bx16_second & mask16) - zpx16_2);
            half16 i4x16_odd_second = convert_half16(as_char16(as_uchar16(bx16_second) >> 4) - zpx16_3);
#else
            char16 i4x16_even_c16 = (bx16 & (char16)0xF);
            char16 i4x16_odd_c16 = (as_char16(as_uchar16(bx16) >> 4));
            i4x16_even_c16 = select(i4x16_even_c16, i4x16_even_c16 - (char16)16, i4x16_even_c16 > (char16)7) - zpx16_0;
            i4x16_odd_c16 = select(i4x16_odd_c16, i4x16_odd_c16 - (char16)16, i4x16_odd_c16 > (char16)7) - zpx16_1;

            char16 i4x16_even_c16_second = (bx16_second & (char16)0xF);
            char16 i4x16_odd_c16_second = (as_char16(as_uchar16(bx16_second) >> 4));
            i4x16_even_c16_second = select(i4x16_even_c16_second, i4x16_even_c16_second - (char16)16, i4x16_even_c16_second > (char16)7) - zpx16_2;
            i4x16_odd_c16_second = select(i4x16_odd_c16_second, i4x16_odd_c16_second - (char16)16, i4x16_odd_c16_second > (char16)7) - zpx16_3;

            half16 i4x16_even = convert_half16(i4x16_even_c16);
            half16 i4x16_odd = convert_half16(i4x16_odd_c16);
            half16 i4x16_even_second = convert_half16(i4x16_even_c16_second);
            half16 i4x16_odd_second = convert_half16(i4x16_odd_c16_second);
#endif

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

        // scales applied once
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
            output[cur_n + 16 * i] = sum_value[i];
        }
#    endif
    }
}
#endif
