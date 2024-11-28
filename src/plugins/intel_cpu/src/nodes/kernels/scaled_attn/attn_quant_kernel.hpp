// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template<typename TDST>
void attn_dequant_u8_kernel(const uint8_t* src, TDST* dst, size_t n, float scale, float zp) {
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

template<typename TDST>
void attn_dequant_u4_kernel(const uint8_t* src, TDST* dst, size_t n, float scale, float zp) {
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
    auto extract_half_byte2 = [&](uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;
        return (uint8_t) ((val >> shift) & 0x000F);
    };
    auto v_zp = _mm512_set1_ps(zp);
    auto v_scale = _mm512_set1_ps(scale);
    for (; i + vec_len_f32_avx512 * 2 <= n; i += vec_len_f32_avx512 * 2) {
        auto high_half = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i / 2));
        __m128i low_half = _mm_srli_epi16(high_half, 4);
        const __m128i mask = _mm_set1_epi8(0x0F);
        low_half = _mm_and_si128(mask, low_half);
        high_half = _mm_and_si128(mask, high_half);

        //cvt to f32
        auto v_256_low_half = _mm512_cvtepu8_epi32(low_half);
        auto v_256_high_half = _mm512_cvtepu8_epi32(high_half);
        auto v_f32_low_half = _mm512_cvtepi32_ps(v_256_low_half);
        auto v_f32_high_half = _mm512_cvtepi32_ps(v_256_high_half);
        // q - zp
        v_f32_low_half = _mm512_sub_ps(v_f32_low_half, v_zp);
        v_f32_high_half = _mm512_sub_ps(v_f32_high_half, v_zp);
        // (q - zp) * scale
        v_f32_low_half = _mm512_mul_ps(v_f32_low_half, v_scale);
        v_f32_high_half = _mm512_mul_ps(v_f32_high_half, v_scale);
    
        __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
        __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
        __m512 first_half = _mm512_permutex2var_ps(v_f32_low_half, idx1, v_f32_high_half);
        __m512 second_half = _mm512_permutex2var_ps(v_f32_low_half, idx2, v_f32_high_half);
        mm512_uni_storeu_ps(dst + i, first_half);
        mm512_uni_storeu_ps(dst + i + vec_len_f32_avx512, second_half);
    }
#endif
    auto extract_half_byte = [&](uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;
        return (uint8_t) ((val >> shift) & 0x000F);
    };
    for (; i < n; ++i) {
        float tmp = extract_half_byte(src_nc[i / 2], (uint8_t)(i % 2));
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}

template<typename TDST>
void attn_dequant_s4_kernel(const uint8_t* src, TDST* dst, size_t n, float scale) {
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
    for (; i + vec_len_f32_avx512 * 2 <= n; i += vec_len_f32_avx512 * 2) {
        auto high_half = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i / 2));
        __m128i low_half = _mm_srli_epi16(high_half, 4);
        const __m128i mask = _mm_set1_epi8(0x0F);
        low_half = _mm_and_si128(mask, low_half);
        auto v_scale = _mm512_set1_ps(1/scale);
        //cvt to f32
        auto v_256_low_half = _mm512_cvtepi8_epi32(low_half);
        auto v_256_high_half = _mm512_cvtepi8_epi32(high_half);
        v_256_high_half = _mm512_slli_epi32(v_256_high_half, 28);
        v_256_high_half = _mm512_srai_epi32(v_256_high_half, 28);
        auto v_f32_low_half = _mm512_cvtepi32_ps(v_256_low_half);
        auto v_f32_high_half = _mm512_cvtepi32_ps(v_256_high_half);
        // q * scale
        v_f32_low_half = _mm512_mul_ps(v_f32_low_half, v_scale);
        v_f32_high_half = _mm512_mul_ps(v_f32_high_half, v_scale);

        __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
        __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
        __m512 first_half = _mm512_permutex2var_ps(v_f32_low_half, idx1, v_f32_high_half);
        __m512 second_half = _mm512_permutex2var_ps(v_f32_low_half, idx2, v_f32_high_half);  
        mm512_uni_storeu_ps(dst + i, first_half);
        mm512_uni_storeu_ps(dst + i + vec_len_f32_avx512, second_half);  
    }
    auto extract_half_byte = [&](uint8_t val, bool high_half) -> uint8_t {
        uint8_t shift = high_half ? 0 : 4;
        return (int8_t) ((val >> shift) & 0x000F);
    };
    for (; i < n; ++i) {
        float tmp = extract_half_byte(src_nc[i / 2], (uint8_t)(i % 2));
        tmp = tmp * scale;
        dst[i] = tmp;
    }
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov