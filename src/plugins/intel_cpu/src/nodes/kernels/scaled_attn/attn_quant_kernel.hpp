// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#if defined(HAVE_SVE)
#    include "arm_sve.h"
#endif

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<SRC_PREC == ov::element::u8, bool>::type = true>
void attn_dequant_kernel(const uint8_t* src, TDST* dst, size_t n, float scale, float zp) {
    size_t i = 0;
    // loadu_si128/epi64 does not support const qualifier
    uint8_t* src_nc = const_cast<uint8_t*>(src);
#if defined(HAVE_AVX512F)
    auto v_zp = _mm512_set1_ps(zp);
    auto v_scale = _mm512_set1_ps(scale);
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_512 = _mm512_cvtepu8_epi32(v0_128);
        auto v0_value = _mm512_cvtepi32_ps(v0_512);
        v0_value = _mm512_sub_ps(v0_value, v_zp);
        auto v0_out = _mm512_mul_ps(v0_value, v_scale);
        mm512_uni_storeu_ps(dst + i, v0_out);
    }
#elif defined(HAVE_AVX2)
    auto v_zp = _mm256_set1_ps(zp);
    auto v_scale = _mm256_set1_ps(scale);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        v0_value = _mm256_sub_ps(v0_value, v_zp);
        auto v0_out = _mm256_mul_ps(v0_value, v_scale);
        mm256_uni_storeu_ps(dst + i, v0_out);
    }
#endif
    for (; i < n; ++i) {
        float tmp = src_nc[i];
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<SRC_PREC == ov::element::u4, bool>::type = true>
void attn_dequant_kernel(const uint8_t* src, TDST* dst, size_t n, float scale, float zp) {
    // 2 4bit data form a byte
    /* 0,1|2,3|4,5|6,7
          /      \
       0,2,4,6|1,3,5,7
              |
           permute
              |
       0,1,2,3,4,5,6,7
    */
    size_t i = 0;
    uint8_t* src_nc = const_cast<uint8_t*>(src);
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(scale);
    auto v_zp_scale = _mm512_set1_ps(zp * scale);
    for (; i + vec_len_f32_avx512 * 2 <= n; i += vec_len_f32_avx512 * 2) {
        auto data = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i / 2));
        auto v_i32 = _mm512_cvtepu8_epi32(data);

        auto v_512_low_half = _mm512_srli_epi32(v_i32, 4);
        auto v_f32_low_half = _mm512_cvtepi32_ps(v_512_low_half);

        auto mask = _mm512_set1_epi32(0x0F);
        auto v_512_high_half = _mm512_and_si512(v_i32, mask);
        auto v_f32_high_half = _mm512_cvtepi32_ps(v_512_high_half);
        // q * scale- zp * scale
        v_f32_low_half = _mm512_fmsub_ps(v_f32_low_half, v_scale, v_zp_scale);
        v_f32_high_half = _mm512_fmsub_ps(v_f32_high_half, v_scale, v_zp_scale);
        __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
        __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
        __m512 first_half = _mm512_permutex2var_ps(v_f32_low_half, idx1, v_f32_high_half);
        __m512 second_half = _mm512_permutex2var_ps(v_f32_low_half, idx2, v_f32_high_half);
        mm512_uni_storeu_ps(dst + i, first_half);
        mm512_uni_storeu_ps(dst + i + vec_len_f32_avx512, second_half);
    }
#elif defined(HAVE_AVX2)
    auto v256_zp = _mm256_set1_ps(zp);
    auto v256_scale = _mm256_set1_ps(scale);
    for (; i + vec_len_f32_avx2 * 2 <= n; i += vec_len_f32_avx2 * 2) {
        auto data = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src_nc + i / 2));

        auto v_i32 = _mm256_cvtepu8_epi32(data);
        auto v_256_low_half = _mm256_srli_epi32(v_i32, 4);
        auto v_f32_low_half = _mm256_cvtepi32_ps(v_256_low_half);

        auto mask = _mm256_set1_epi32(0x0F);
        auto v_256_high_half = _mm256_and_si256(v_i32, mask);
        auto v_f32_high_half = _mm256_cvtepi32_ps(v_256_high_half);
        // q - zp
        v_f32_low_half = _mm256_sub_ps(v_f32_low_half, v256_zp);
        v_f32_high_half = _mm256_sub_ps(v_f32_high_half, v256_zp);

        v_f32_low_half = _mm256_mul_ps(v_f32_low_half, v256_scale);
        v_f32_high_half = _mm256_mul_ps(v_f32_high_half, v256_scale);

        // 0,2,4,6,8,10,12,14 | 1,3,5,7,9,11,13,15
        //         _mm256_permute2f128_ps
        // 0,2,4,6,1,3,5,7    | 8,10,12,14,9,11,13,15
        //         _mm256_permutevar8x32_ps
        // 0,1,2,3,4,5,6,7    | 8,9,10,11,12,13,14,15
        __m256 first_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x20);
        auto idx1 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
        first_half = _mm256_permutevar8x32_ps(first_half, idx1);
        __m256 second_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x31);
        second_half = _mm256_permutevar8x32_ps(second_half, idx1);

        mm256_uni_storeu_ps(dst + i, first_half);
        mm256_uni_storeu_ps(dst + i + vec_len_f32_avx2, second_half);
    }
#endif
    auto extract_half_byte = [&](uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;
        return (uint8_t)((val >> shift) & 0x000F);
    };
    for (; i < n; ++i) {
        float tmp = extract_half_byte(src_nc[i / 2], (uint8_t)(i % 2));
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}

#if defined(HAVE_SVE)
void inline attn_dequant_u8_kernel(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    size_t i = 0;
    uint8_t* src_nc = const_cast<uint8_t*>(src);
    size_t nvec = n / svcntw();
    size_t lvec = svcntw();
    auto sve_pg = svptrue_b32();
    for (size_t j = 0; j < nvec; ++j) {
        svuint32_t reg1 = svld1ub_u32(sve_pg, src_nc + j * lvec);
        svfloat32_t reg2 = svcvt_f32_u32_z(sve_pg, reg1);
        svfloat32_t reg3 = svsub_f32_z(sve_pg, reg2, svdup_n_f32(zp));
        svfloat32_t reg4 = svmul_f32_z(sve_pg, reg3, svdup_n_f32(scale));
        svst1_f32(sve_pg, dst + j * lvec, reg4);
    }
    i = n - n % svcntw();
    for (; i < n; ++i) {
        float tmp = src_nc[i];
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}
#endif

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov