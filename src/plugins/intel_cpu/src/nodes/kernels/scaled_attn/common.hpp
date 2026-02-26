// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include "openvino/core/type/bfloat16.hpp"
#endif

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#if defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
#        include "arm_sve.h"
#    endif
#    include "arm_neon.h"
#endif

namespace ov::Extensions::Cpu::XARCH {

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
static constexpr size_t vec_len_neon = 16lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);
static constexpr size_t vec_len_f32_neon = vec_len_neon / sizeof(float);
static constexpr size_t vec_len_f16_neon = vec_len_neon / sizeof(ov::float16);
static constexpr size_t vec_len_epi8_avx2 = vec_len_avx2 / sizeof(int8_t);

#if defined(HAVE_SVE)
inline size_t vec_len_f32_sve() {
    static size_t len = svcntw();
    return len;
}
inline size_t vec_len_f16_sve() {
    static size_t len = svcnth();
    return len;
}
#endif

constexpr size_t get_sub_byte_multiplier(ov::element::Type type) {
    return ov::intel_cpu::any_of(type, ov::element::i4, ov::element::u4) ? 2 : 1;
}

uint8_t inline insert_half_byte(uint8_t dst, uint8_t val, bool high_half) {
    uint8_t shift = high_half ? 0 : 4;
    return dst | static_cast<uint8_t>(val << shift);
}

uint8_t inline extract_half_byte(uint8_t val, bool high_half) {
    uint8_t shift = high_half ? 0 : 4;

    return static_cast<uint8_t>((val >> shift) & 0x000F);
};

#ifdef HAVE_AVX512F
inline __m512 cvt_bf16_to_fp32(const __m256i src) {
    __m512i y = _mm512_cvtepu16_epi32(src);
    return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
}

// load addr to __m512 reg
inline __m512 mm512_uni_loadu_ps(const float* a) {
    return _mm512_loadu_ps(a);
}

inline __m512 mm512_uni_loadu_ps(const ov::bfloat16* a) {
    auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    return cvt_bf16_to_fp32(vec_bf16);
}

inline __m512 mm512_uni_loadu_ps(const ov::float16* a) {
    auto vec_f16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
    return _mm512_cvtph_ps(vec_f16);
}

// load addr to __m512 reg
inline __m512 mm512_uni_loadu_tail_ps(const float* a, size_t count) {
    __mmask16 mask = (1 << count) - 1;
    return _mm512_maskz_loadu_ps(mask, a);
}

inline __m512 mm512_uni_loadu_tail_ps(const ov::bfloat16* a, size_t count) {
    auto mask = (1 << count) - 1;
    auto bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
    return cvt_bf16_to_fp32(bf16_vec);
}

inline __m512 mm512_uni_loadu_tail_ps(const ov::float16* a, size_t count) {
    auto mask = (1 << count) - 1;
    auto f16_vec = _mm256_maskz_loadu_epi16(mask, a);
    return _mm512_cvtph_ps(f16_vec);
}

// store __m512 reg to addr
inline void mm512_uni_storeu_ps(float* a, __m512 v) {
    _mm512_storeu_ps(a, v);
}
inline void mm512_uni_storeu_ps(ov::bfloat16* addr, __m512 xps) {
    __m512i xpi32 = _mm512_castps_si512(xps);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones);  // LSB = x[16]
    x = _mm512_add_epi32(x, vec_bias);                              // rounding_bias = 0x7fff + LSB
    x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);          // x = (x + rounding_bias) >> 16;
    x = _mm512_mask_blend_epi32(mask, nan, x);                      // Check NaN before converting back to bf16
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr), _mm512_cvtepi32_epi16(x));
}

inline void mm512_uni_storeu_ps(ov::float16* addr, __m512 v) {
    __m256i vec_f16 = _mm512_cvtps_ph(v, 0);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(addr), vec_f16);
}

// store __m512 reg to addr
inline void mm512_uni_mask_storeu_ps(ov::bfloat16* addr, __mmask16 mask_addr, __m512 xps) {
    __m512i xpi32 = _mm512_castps_si512(xps);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones);  // LSB = x[16]
    x = _mm512_add_epi32(x, vec_bias);                              // rounding_bias = 0x7fff + LSB
    x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);          // x = (x + rounding_bias) >> 16;
    x = _mm512_mask_blend_epi32(mask, nan, x);                      // Check NaN before converting back to bf16
    _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
}

inline void mm512_uni_storeu_tail_ps(float* addr, __m512 v, size_t count) {
    __mmask16 mask_addr = (1 << count) - 1;
    _mm512_mask_storeu_ps(addr, mask_addr, v);
}

inline void mm512_uni_storeu_tail_ps(ov::bfloat16* addr, __m512 v, size_t count) {
    __mmask16 mask_addr = (1 << count) - 1;
    __m512i xpi32 = _mm512_castps_si512(v);
    __m512i nan = _mm512_set1_epi32(0xffff);
    auto mask = _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q);
    __m512i ones = _mm512_set1_epi32(0x1);
    __m512i vec_bias = _mm512_set1_epi32(0x7fff);
    auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones);  // LSB = x[16]
    x = _mm512_add_epi32(x, vec_bias);                              // rounding_bias = 0x7fff + LSB
    x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);          // x = (x + rounding_bias) >> 16;
    x = _mm512_mask_blend_epi32(mask, nan, x);                      // Check NaN before converting back to bf16
    _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
}

inline void mm512_uni_storeu_tail_ps(ov::float16* addr, __m512 v, size_t count) {
    __mmask16 mask_addr = (1 << count) - 1;
    __m256i vec_f16 = _mm512_cvtps_ph(v, 0);
    _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(addr), mask_addr, vec_f16);
}

inline void mm512_loadu_u4_to_f32(uint8_t* src_data, __m512& first_half, __m512& second_half) {
    auto data = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_data));
    auto v_i32 = _mm512_cvtepu8_epi32(data);

    auto v_512_low_half = _mm512_srli_epi32(v_i32, 4);
    auto v_f32_low_half = _mm512_cvtepi32_ps(v_512_low_half);

    auto mask = _mm512_set1_epi32(0x0F);
    auto v_512_high_half = _mm512_and_si512(v_i32, mask);
    auto v_f32_high_half = _mm512_cvtepi32_ps(v_512_high_half);
    __m512i idx1 = _mm512_set_epi32(23, 7, 22, 6, 21, 5, 20, 4, 19, 3, 18, 2, 17, 1, 16, 0);
    __m512i idx2 = _mm512_set_epi32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
    first_half = _mm512_permutex2var_ps(v_f32_low_half, idx1, v_f32_high_half);
    second_half = _mm512_permutex2var_ps(v_f32_low_half, idx2, v_f32_high_half);
}

inline void mm512_storeu_u4(uint8_t* dst_data, __m512i& v0, __m512i& v1) {
    __m512i idx1 = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i idx2 = _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    auto first_half = _mm512_permutex2var_epi32(v0, idx1, v1);
    auto second_half = _mm512_permutex2var_epi32(v0, idx2, v1);
    first_half = _mm512_slli_epi32(first_half, 4);
    auto mask = _mm512_set1_epi32(0x0F);
    second_half = _mm512_and_epi32(second_half, mask);
    auto combined = _mm512_or_epi32(first_half, second_half);
    _mm512_mask_cvtepi32_storeu_epi8(dst_data, 0xffff, combined);
}

#endif

#ifdef HAVE_AVX2
inline __m128i get_8bit_tail_mask_for_16bit_elts(size_t num_16bit_tail_elts) {
    // num_tail_elts may take from 0 to 8
    static int8_t masks[9][16] = {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};
    return _mm_loadu_si128(reinterpret_cast<__m128i*>(masks[num_16bit_tail_elts]));
}
inline __m256i get_mask(int N7) {
    static int32_t masks[9][8] = {{0, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, 0, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, 0, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, 0, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, 0, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, 0, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, 0, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, 0},
                                  {-1, -1, -1, -1, -1, -1, -1, -1}};
    return _mm256_loadu_si256(reinterpret_cast<__m256i*>(masks[N7]));
}

// load addr to __m256 reg
inline __m256 mm256_uni_loadu_ps(const float* a) {
    return _mm256_loadu_ps(a);
}

inline __m256 mm256_uni_loadu_ps(const ov::bfloat16* a) {
    auto vec_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    auto o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vec_bf16), 16));
    return o;
}

inline __m256 mm256_uni_loadu_ps(const ov::float16* a) {
    auto vec_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
    auto o = _mm256_cvtph_ps(vec_f16);
    return o;
}

// load addr tail to __m256 reg
inline __m256 mm256_uni_loadu_tail_ps(const float* a, const size_t count) {
    auto mask = get_mask(static_cast<int>(count));
    return _mm256_maskload_ps(a, mask);
}

inline __m256 mm256_uni_loadu_tail_ps(const ov::bfloat16* a, const size_t count) {
    assert("AVX2 version of bfloat16 tail load is just for compilation pass");
    ov::bfloat16 tmp_values[8] = {0};
    std::memcpy(tmp_values, a, count * sizeof(ov::bfloat16));
    return mm256_uni_loadu_ps(tmp_values);
}

inline __m256 mm256_uni_loadu_tail_ps(const ov::float16* a, const size_t count) {
    ov::float16 tmp_values[8] = {0};
    std::memcpy(tmp_values, a, count * sizeof(ov::float16));
    return mm256_uni_loadu_ps(tmp_values);
}

// store __m256 reg to addr
inline void mm256_uni_storeu_ps(float* a, __m256 v) {
    _mm256_storeu_ps(a, v);
}

inline __m128i __convert_avx2_packed_float_to_packed_ov_bfloat16(__m256 xps) {
    __m256i xpi32 = _mm256_castps_si256(xps);
    __m256i nan = _mm256_set1_epi32(0xffff);
    __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(xps, xps, _CMP_ORD_Q));
    __m256i ones = _mm256_set1_epi32(0x1);
    __m256i vec_bias = _mm256_set1_epi32(0x7fff);
    auto x = _mm256_and_si256(_mm256_srli_epi32(xpi32, 16), ones);  // LSB = x[16]
    x = _mm256_add_epi32(x, vec_bias);                              // rounding_bias = 0x7fff + LSB
    x = _mm256_srli_epi32(_mm256_add_epi32(x, xpi32), 16);          // x = (x + rounding_bias) >> 16;
    x = _mm256_blendv_epi8(nan, x, mask);                           // Check NaN before converting back to bf16
    x = _mm256_packus_epi32(x, x);
    x = _mm256_permute4x64_epi64(x, 0xd8);
    __m128i bf16_o = _mm256_extractf128_si256(x, 0);
    return bf16_o;
}

inline void mm256_uni_storeu_ps(ov::bfloat16* addr, __m256 xps) {
    __m128i bf16_o = __convert_avx2_packed_float_to_packed_ov_bfloat16(xps);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(addr), bf16_o);
}

inline void mm256_uni_storeu_ps(ov::float16* a, __m256 v) {
    __m128i vec_f16 = _mm256_cvtps_ph(v, 0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(a), vec_f16);
}

// store __m256 to addr
inline void mm256_uni_storeu_tail_ps(float* addr, __m256 v, size_t count) {
    auto mask = get_mask(static_cast<int>(count));
    return _mm256_maskstore_ps(addr, mask, v);
}

inline void mm256_uni_storeu_tail_ps(ov::float16* addr, __m256 v, size_t count) {
    auto mask = get_8bit_tail_mask_for_16bit_elts(count);
    __m128i vec_f16 = _mm256_cvtps_ph(v, 0);
    return _mm_maskmoveu_si128(vec_f16, mask, reinterpret_cast<char*>(addr));
}

inline void mm256_uni_storeu_tail_ps(ov::bfloat16* addr, __m256 v, size_t count) {
    auto mask = get_8bit_tail_mask_for_16bit_elts(count);
    __m128i bf16_o = __convert_avx2_packed_float_to_packed_ov_bfloat16(v);
    return _mm_maskmoveu_si128(bf16_o, mask, reinterpret_cast<char*>(addr));
}

inline void mm256_loadu_u4_to_f32(uint8_t* src, __m256& first_half, __m256& second_half) {
    auto data = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src));

    auto v_i32 = _mm256_cvtepu8_epi32(data);
    auto v_256_low_half = _mm256_srli_epi32(v_i32, 4);
    auto v_f32_low_half = _mm256_cvtepi32_ps(v_256_low_half);

    auto mask = _mm256_set1_epi32(0x0F);
    auto v_256_high_half = _mm256_and_si256(v_i32, mask);
    auto v_f32_high_half = _mm256_cvtepi32_ps(v_256_high_half);

    // 0,2,4,6,8,10,12,14 | 1,3,5,7,9,11,13,15
    //         _mm256_permute2f128_ps
    // 0,2,4,6,1,3,5,7    | 8,10,12,14,9,11,13,15
    //         _mm256_permutevar8x32_ps
    // 0,1,2,3,4,5,6,7    | 8,9,10,11,12,13,14,15
    first_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x20);
    auto idx1 = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
    first_half = _mm256_permutevar8x32_ps(first_half, idx1);
    second_half = _mm256_permute2f128_ps(v_f32_low_half, v_f32_high_half, 0x31);
    second_half = _mm256_permutevar8x32_ps(second_half, idx1);
}

inline void mm256_storeu_u4(uint8_t* dst_data, __m256i& v0_i32, __m256i& v1_i32) {
    auto idx1 = _mm256_set_epi32(7, 5, 3, 1, 6, 4, 2, 0);
    v0_i32 = _mm256_permutevar8x32_epi32(v0_i32, idx1);
    v1_i32 = _mm256_permutevar8x32_epi32(v1_i32, idx1);
    //    0,1,2,3,4,5,6,7 | 8,9,10,11,12,13,14,15
    //       _mm256_permutevar8x32_epi32
    //    0,2,4,6,1,3,5,7 | 8,10,12,14,9,11,13,15
    //       _mm256_permute2x128_si256
    // 0,2,4,6,8,10,12,14 | 1,3,5,7,9,11,13,15
    //          shift + mask + or
    //     [0,1],[2,3], ..., [12,13], [14,15]
    auto first_half = _mm256_permute2x128_si256(v0_i32, v1_i32, 0x20);
    auto second_half = _mm256_permute2x128_si256(v0_i32, v1_i32, 0x31);
    first_half = _mm256_slli_epi32(first_half, 4);
    auto mask = _mm256_set1_epi32(0x0F);
    second_half = _mm256_and_si256(second_half, mask);
    auto combined = _mm256_or_si256(first_half, second_half);

    auto high4 = _mm256_extractf128_si256(combined, 1);
    auto low4 = _mm256_castsi256_si128(combined);
    // ignore sign bit for u4 case
    auto packed = _mm_packus_epi32(low4, high4);
    packed = _mm_packus_epi16(packed, packed);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(dst_data), packed);
}

inline void hsum(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
}
inline void hmax(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_max_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_max_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_max_ps(x, y);              // x: 01234567 x x x x x x x
}
inline void hmin(__m256& x) {
    __m256 y;                             // x:  0 1 2 3   4 5 6 7
    y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
    x = _mm256_min_ps(x, y);              // X:  01 12 23 30  45 56 67 74
    y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
    x = _mm256_min_ps(x, y);              // x: 0123 x x x   4567 x x x
    y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
    x = _mm256_min_ps(x, y);              // x: 01234567 x x x x x x x
}
#endif

#ifdef OPENVINO_ARCH_ARM64
#    if defined(HAVE_SVE)
inline svfloat32_t exp_ps_sve(svbool_t& pg, svfloat32_t& src) {
    const auto c1 = svreinterpret_f32_u32(svdup_n_u32(0x3f7ffff6));
    const auto c2 = svreinterpret_f32_u32(svdup_n_u32(0x3efffedb));
    const auto c3 = svreinterpret_f32_u32(svdup_n_u32(0x3e2aaf33));
    const auto c4 = svreinterpret_f32_u32(svdup_n_u32(0x3d2b9f17));
    const auto c5 = svreinterpret_f32_u32(svdup_n_u32(0x3c072010));

    const auto shift = svreinterpret_f32_u32(svdup_n_u32(0x4b00007f));  // 2^23 + 127 = 0x1.0000fep23f
    const auto one = svdup_n_f32(1.0f);                                 // 1
    const auto two = svdup_n_f32(2.0f);                                 // 2
    const auto inv_ln2 = svreinterpret_f32_u32(svdup_n_u32(0x3fb8aa3b));
    const auto neg_ln2_hi = svreinterpret_f32_u32(svdup_n_u32(0xbf317200));
    const auto neg_ln2_lo = svreinterpret_f32_u32(svdup_n_u32(0xb5bfbe8e));

    const auto inf = svdup_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = svdup_n_f32(88.37f);  // Approximately ln(2^127.5)
    const auto zero = svdup_n_f32(0.F);
    const auto min_input = svdup_n_f32(-86.64f);  // Approximately ln(2^-125)

    auto x = svmin_f32_z(pg, src, max_input);
    x = svmax_f32_z(pg, x, min_input);

    const auto z = svmla_f32_z(pg, shift, x, inv_ln2);
    auto n = svsub_f32_z(pg, z, shift);

    const auto r_hi = svmla_f32_z(pg, x, n, neg_ln2_hi);
    const auto r = svmla_f32_z(pg, r_hi, n, neg_ln2_lo);
    n = svsub_f32_z(pg, n, one);

    const auto n_int = svcvt_s32_f32_z(pg, n);
    const auto exponent_bias = svdup_n_s32(127);
    const auto n_int_bias = svadd_s32_z(pg, n_int, exponent_bias);
    const auto scale = svreinterpret_f32_s32(svlsl_n_s32_z(pg, n_int_bias, 23));  // 2^(n-1)
    const auto r2 = svmul_f32_z(pg, r, r);

    const auto p1 = svmul_f32_z(pg, c1, r);
    const auto p23 = svmla_f32_z(pg, c2, c3, r);
    const auto p45 = svmla_f32_z(pg, c4, c5, r);
    const auto p2345 = svmla_f32_z(pg, p23, p45, r2);
    const auto p12345 = svmla_f32_z(pg, p1, p2345, r2);

    auto poly = svmla_f32_z(pg, scale, p12345, scale);
    poly = svmul_f32_z(pg, poly, two);

    poly = svsel_f32(svcmplt_f32(pg, src, min_input), zero, poly);
    poly = svsel_f32(svcmpgt_f32(pg, src, max_input), inf, poly);

    return poly;
}
#    endif
inline float32x4_t exp_ps_neon_f32(const float32x4_t& src) {
    const auto c1 = vreinterpretq_f32_u32(vdupq_n_u32(0x3f7ffff6));
    const auto c2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3efffedb));
    const auto c3 = vreinterpretq_f32_u32(vdupq_n_u32(0x3e2aaf33));
    const auto c4 = vreinterpretq_f32_u32(vdupq_n_u32(0x3d2b9f17));
    const auto c5 = vreinterpretq_f32_u32(vdupq_n_u32(0x3c072010));

    const auto shift = vreinterpretq_f32_u32(vdupq_n_u32(0x4b00007f));  // 2^23 + 127 = 0x1.0000fep23f
    const auto one = vdupq_n_f32(1.0f);                                 // 1
    const auto two = vdupq_n_f32(2.0f);                                 // 2
    const auto inv_ln2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3fb8aa3b));
    const auto neg_ln2_hi = vreinterpretq_f32_u32(vdupq_n_u32(0xbf317200));
    const auto neg_ln2_lo = vreinterpretq_f32_u32(vdupq_n_u32(0xb5bfbe8e));

    const auto inf = vdupq_n_f32(std::numeric_limits<float>::infinity());
    const auto max_input = vdupq_n_f32(88.37f);  // Approximately ln(2^127.5)
    const auto zero = vdupq_n_f32(0.F);
    const auto min_input = vdupq_n_f32(-86.64f);  // Approximately ln(2^-125)

    auto x = vminq_f32(src, max_input);
    x = vmaxq_f32(x, min_input);

    const auto z = vmlaq_f32(shift, x, inv_ln2);
    auto n = z - shift;

    const auto r_hi = vfmaq_f32(x, n, neg_ln2_hi);
    const auto r = vfmaq_f32(r_hi, n, neg_ln2_lo);
    n = vsubq_f32(n, one);

    const auto n_int = vcvtq_s32_f32(n);
    const auto exponent_bias = vdupq_n_s32(127);
    const auto n_int_bias = vaddq_s32(n_int, exponent_bias);
    const auto scale = vreinterpretq_f32_s32(vshlq_n_s32(n_int_bias, 23));  // 2^(n-1)

    const auto r2 = r * r;

    const auto p1 = c1 * r;
    const auto p23 = vfmaq_f32(c2, c3, r);
    const auto p45 = vfmaq_f32(c4, c5, r);
    const auto p2345 = vfmaq_f32(p23, p45, r2);
    const auto p12345 = vfmaq_f32(p1, p2345, r2);

    auto poly = vfmaq_f32(scale, p12345, scale);
    poly = vmulq_f32(poly, two);

    poly = vbslq_f32(vcltq_f32(src, min_input), zero, poly);
    poly = vbslq_f32(vcgtq_f32(src, max_input), inf, poly);

    return poly;
}
inline float32x4_t __vld1q_f32(const ov::bfloat16* a) {
    uint16x4_t vec_bf16 = vld1_u16(reinterpret_cast<const uint16_t*>(a));

    float32x4_t vec_f32 = vcvtq_f32_u32(vmovl_u16(vec_bf16));
    return vec_f32;
}
inline float32x4_t __vld1q_f32(const float* a) {
    return vld1q_f32(a);
}
inline float32x4_t __vld1q_f32(const ov::float16* a) {
    auto _a = reinterpret_cast<const float16_t*>(a);
    return vcvt_f32_f16(vld1_f16(_a));
}
inline void __vst1q_f32(float* a, float32x4_t b) {
    vst1q_f32(a, b);
}
inline void __vst1q_f32(ov::float16* a, float32x4_t b) {
    float16x4_t v_f16 = vcvt_f16_f32(b);
    vst1_f16(reinterpret_cast<float16_t*>(a), v_f16);
}
inline void __vst1q_f32(ov::bfloat16* a, float32x4_t b) {
    uint32x4_t v_int32 = vreinterpretq_u32_f32(b);
    uint16x4_t v_bf16 = vshrn_n_u32(v_int32, 16);

    vst1_u16(reinterpret_cast<uint16_t*>(a), v_bf16);
}

#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#    if defined(HAVE_SVE)
inline svfloat16_t exp_ps_sve_f16(svbool_t& pg, svfloat16_t& src) {
    svbool_t pg_f32 = svtrn1_b16(pg, svpfalse());

    // Extract lower and upper halves of src into two separate vecs and convert
    svfloat16_t zero = svdup_n_f16(0.0);
    svfloat16_t low_f16 = svtrn1_f16(src, zero);
    svfloat16_t high_f16 = svtrn2_f16(src, zero);
    svfloat32_t low_f32 = svcvt_f32_f16_z(pg, low_f16);
    svfloat32_t high_f32 = svcvt_f32_f16_z(pg, high_f16);

    // Perform exp and convert back to f16
    svfloat32_t low_exp_f32 = exp_ps_sve(pg_f32, low_f32);
    svfloat32_t high_exp_f32 = exp_ps_sve(pg_f32, high_f32);
    svfloat16_t low_exp_f16 = svcvt_f16_f32_z(pg_f32, low_exp_f32);
    svfloat16_t high_exp_f16 = svcvt_f16_f32_z(pg_f32, high_exp_f32);

    // Interleave both to get final result
    svfloat16_t res = svtrn1_f16(low_exp_f16, high_exp_f16);
    return res;
}
#    else
inline float16x8_t exp_ps_neon_f16(float16x8_t x) {
    const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
    const float32x4_t x_low = vcvt_f32_f16(vget_low_f16(x));

    // We use f32 to maintain accuracy
    const float16x8_t res = vcombine_f16(vcvt_f16_f32(exp_ps_neon_f32(x_low)), vcvt_f16_f32(exp_ps_neon_f32(x_high)));
    return res;
}
#    endif
inline float16_t hsum(float16x8_t vec) {
    float16x4_t sum1 = vpadd_f16(vget_low_f16(vec), vget_high_f16(vec));
    float16x4_t sum2 = vpadd_f16(sum1, sum1);
    float16x4_t sum3 = vpadd_f16(sum2, sum2);
    return vget_lane_f16(sum3, 0);
}
#endif

template <typename TA, typename TB>
void cvt_copy(TA* a, TB* b, size_t m, size_t n, size_t src_stride, size_t dst_stride) {
    for (size_t j = 0; j < m; j++) {
        size_t i = 0;
#if defined(HAVE_AVX512F)
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto vb = mm512_uni_loadu_ps(b + i + j * src_stride);
            mm512_uni_storeu_ps(a + i + j * dst_stride, vb);
        }
#elif defined(HAVE_AVX2)
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto vb = mm256_uni_loadu_ps(b + i + j * src_stride);
            mm256_uni_storeu_ps(a + i + j * dst_stride, vb);
        }
#endif
        for (; i < n; i++) {
            a[i + j * dst_stride] = b[i + j * src_stride];
        }
    }
}

template <typename TDST, typename TA, typename TB>
void cvt_add(TDST* dst, TA* a, TB* b, size_t m, size_t n, size_t a_stride, size_t b_stride, size_t dst_stride) {
    for (size_t j = 0; j < m; j++) {
        size_t i = 0;
#if defined(HAVE_AVX512F)
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i + j * a_stride);
            auto vb = mm512_uni_loadu_ps(b + i + j * b_stride);
            auto vd = _mm512_add_ps(va, vb);
            mm512_uni_storeu_ps(dst + i + j * dst_stride, vd);
        }
#elif defined(HAVE_AVX2)
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i + j * a_stride);
            auto vb = mm256_uni_loadu_ps(b + i + j * b_stride);
            auto vd = _mm256_add_ps(va, vb);
            mm256_uni_storeu_ps(dst + i + j * dst_stride, vd);
        }
#endif
        for (; i < n; i++) {
            dst[i + j * dst_stride] = a[i + j * a_stride] + b[i + j * b_stride];
        }
    }
}

template <typename TA, typename TB>
float dot_product(TA* a,
                  TB* b,
                  size_t n,
                  float* scale,
                  float* zp,
                  float* head_sum,
                  [[maybe_unused]] size_t group_size) {
    size_t i = 0;
    float sum = 0.0F;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);
        auto vb2 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 2);
        auto vb3 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0 = mm256_uni_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);

#elif defined(OPENVINO_ARCH_ARM64)
    // TODO correct handle bf16/f16 for this path CVS-182514
#    if defined(HAVE_SVE)
    svbool_t pg = svptrue_b32();
    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);
    auto vec_len = vec_len_f32_sve();

    auto _a = reinterpret_cast<float32_t*>(a);
    auto _b = reinterpret_cast<float32_t*>(b);

    for (; i + 4 * vec_len <= n; i += 4 * vec_len) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t a1 = svld1_f32(pg, _a + i + vec_len);
        svfloat32_t a2 = svld1_f32(pg, _a + i + vec_len * 2);
        svfloat32_t a3 = svld1_f32(pg, _a + i + vec_len * 3);

        svfloat32_t b0 = svld1_f32(pg, _b + i);
        svfloat32_t b1 = svld1_f32(pg, _b + i + vec_len);
        svfloat32_t b2 = svld1_f32(pg, _b + i + vec_len * 2);
        svfloat32_t b3 = svld1_f32(pg, _b + i + vec_len * 3);

        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        sum1 = svmla_f32_z(pg, sum1, a1, b1);
        sum2 = svmla_f32_z(pg, sum2, a2, b2);
        sum3 = svmla_f32_z(pg, sum3, a3, b3);
    }
    if (i + 2 * vec_len <= n) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t a1 = svld1_f32(pg, _a + i + vec_len);

        svfloat32_t b0 = svld1_f32(pg, _b + i);
        svfloat32_t b1 = svld1_f32(pg, _b + i + vec_len);

        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        sum1 = svmla_f32_z(pg, sum1, a1, b1);
        i += 2 * vec_len;
    }
    if (i + vec_len <= n) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t b0 = svld1_f32(pg, _b + i);
        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        i += vec_len;
    }
    // Process the tail elements parallely as well (if any)
    if (i != n) {
        svbool_t pg_rem = svwhilelt_b32(0, static_cast<int>(n - i));
        svfloat32_t a0 = svld1_f32(pg_rem, _a + i);
        svfloat32_t b0 = svld1_f32(pg_rem, _b + i);
        sum0 = svmla_f32_m(pg_rem, sum0, a0, b0);
        i = n;
    }
    float32_t sum_0 = svaddv_f32(pg, sum0);
    float32_t sum_1 = svaddv_f32(pg, sum1);
    float32_t sum_2 = svaddv_f32(pg, sum2);
    float32_t sum_3 = svaddv_f32(pg, sum3);
    sum = static_cast<float>(sum_0 + sum_1 + sum_2 + sum_3);
#    else
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    float32x4_t vsum1 = vdupq_n_f32(0.0f);
    float32x4_t vsum2 = vdupq_n_f32(0.0f);
    float32x4_t vsum3 = vdupq_n_f32(0.0f);

    for (; i + 4 * vec_len_f32_neon <= n; i += vec_len_f32_neon * 4) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t va1 = __vld1q_f32(a + i + vec_len_f32_neon);
        float32x4_t va2 = __vld1q_f32(a + i + vec_len_f32_neon * 2);
        float32x4_t va3 = __vld1q_f32(a + i + vec_len_f32_neon * 3);

        float32x4_t vb0 = __vld1q_f32(b + i);
        float32x4_t vb1 = __vld1q_f32(b + i + vec_len_f32_neon);
        float32x4_t vb2 = __vld1q_f32(b + i + vec_len_f32_neon * 2);
        float32x4_t vb3 = __vld1q_f32(b + i + vec_len_f32_neon * 3);

        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        vsum1 = vmlaq_f32(vsum1, va1, vb1);
        vsum2 = vmlaq_f32(vsum2, va2, vb2);
        vsum3 = vmlaq_f32(vsum3, va3, vb3);
    }
    if (i + 2 * vec_len_f32_neon <= n) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t va1 = __vld1q_f32(a + i + vec_len_f32_neon);

        float32x4_t vb0 = __vld1q_f32(b + i);
        float32x4_t vb1 = __vld1q_f32(b + i + vec_len_f32_neon);

        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        vsum1 = vmlaq_f32(vsum1, va1, vb1);
        i += 2 * vec_len_f32_neon;
    }
    if (i + vec_len_f32_neon <= n) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t vb0 = __vld1q_f32(b + i);
        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        i += vec_len_f32_neon;
    }

    vsum0 = vaddq_f32(vsum0, vsum1);
    vsum2 = vaddq_f32(vsum2, vsum3);
    vsum0 = vaddq_f32(vsum0, vsum2);

    float32x2_t temp_sum = vadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
    temp_sum = vpadd_f32(temp_sum, temp_sum);
    sum = vget_lane_f32(temp_sum, 0);
#    endif
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

template <typename TA>
float dot_product(TA* a, uint8_t* b, size_t n, float* scale, float* zp, float* head_sum, size_t group_size) {
    float sum = 0.0f;
    size_t group_id = 0;
#if defined(HAVE_AVX512F)
    while (group_id < n / group_size) {
        auto vsum0 = _mm512_set1_ps(0.0f);
        auto vsum1 = _mm512_set1_ps(0.0f);
        auto vsum2 = _mm512_set1_ps(0.0f);
        auto vsum3 = _mm512_set1_ps(0.0f);
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        auto v_zp = _mm512_set1_ps(group_zp);
        size_t offset = group_id * group_size;
        size_t i = 0;
        for (; i + 4 * vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512 * 4) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);
            auto va2 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 2);
            auto va3 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 3);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512));
            auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512 * 2));
            auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512 * 3));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);
            auto vb2 = _mm512_cvtepi32_ps(vb2_256);
            auto vb3 = _mm512_cvtepi32_ps(vb3_256);

            vb0 = _mm512_sub_ps(vb0, v_zp);
            vb1 = _mm512_sub_ps(vb1, v_zp);
            vb2 = _mm512_sub_ps(vb2, v_zp);
            vb3 = _mm512_sub_ps(vb3, v_zp);

            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
            vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
            vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);

            vb0 = _mm512_sub_ps(vb0, v_zp);
            vb1 = _mm512_sub_ps(vb1, v_zp);

            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
            i += 2 * vec_len_f32_avx512;
        }
        if (i + vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            vb0 = _mm512_sub_ps(vb0, v_zp);
            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            i += vec_len_f32_avx512;
        }
        vsum0 = _mm512_add_ps(vsum0, vsum1);
        vsum2 = _mm512_add_ps(vsum2, vsum3);
        vsum0 = _mm512_add_ps(vsum0, vsum2);
        float group_sum = _mm512_reduce_add_ps(vsum0);
        for (; i < group_size; i++) {
            group_sum += a[offset + i] * (b[offset + i] - group_zp);
        }
        sum += group_scale * group_sum;
        group_id += 1;
    }
    return sum;

#elif defined(HAVE_AVX2)
    while (group_id < n / group_size) {
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
        size_t i = 0;
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2 * 4) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);
            auto va2 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 2);
            auto va3 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 3);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2));
            auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2 * 2));
            auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2 * 3));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);
            auto vb2 = _mm256_cvtepi32_ps(vb2_256);
            auto vb3 = _mm256_cvtepi32_ps(vb3_256);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
            vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            i += 2 * vec_len_f32_avx2;
        }
        if (i + vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            i += vec_len_f32_avx2;
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        hsum(vsum0);
        float group_sum = _mm256_cvtss_f32(vsum0);
        for (; i < group_size; i++) {
            group_sum += a[offset + i] * b[offset + i];
        }
        // B = scale * (b - zero)
        // Σ (A * B) = Σ (a * scale * (b - zero)) = scale * (Σ a * b - zero Σ a) = scale * (sum - zp * head_sum)
        group_sum = group_scale * (group_sum - group_zp * head_sum[group_id]);
        sum += group_sum;
        group_id += 1;
    }
    return sum;
#elif defined(OPENVINO_ARCH_ARM64)
    // TODO correct handle bf16/f16 for this path CVS-182514
    while (group_id < n / group_size) {
        size_t i = 0;
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
        float group_sum = 0.0F;

        float32x4_t v_group_zp = vdupq_n_f32(group_zp);

        for (; i + 16 <= group_size; i += 16) {
            uint8x16_t v_u8 = vld1q_u8(b + i + offset);

            uint16x8_t v_u16_lo = vmovl_u8(vget_low_u8(v_u8));
            uint16x8_t v_u16_hi = vmovl_u8(vget_high_u8(v_u8));

            uint32x4_t v_u32_0 = vmovl_u16(vget_low_u16(v_u16_lo));
            uint32x4_t v_u32_1 = vmovl_u16(vget_high_u16(v_u16_lo));
            uint32x4_t v_u32_2 = vmovl_u16(vget_low_u16(v_u16_hi));
            uint32x4_t v_u32_3 = vmovl_u16(vget_high_u16(v_u16_hi));

            float32x4_t v_f0 = vcvtq_f32_u32(v_u32_0);
            float32x4_t v_f1 = vcvtq_f32_u32(v_u32_1);
            float32x4_t v_f2 = vcvtq_f32_u32(v_u32_2);
            float32x4_t v_f3 = vcvtq_f32_u32(v_u32_3);

            v_f0 = vsubq_f32(v_f0, v_group_zp);
            v_f1 = vsubq_f32(v_f1, v_group_zp);
            v_f2 = vsubq_f32(v_f2, v_group_zp);
            v_f3 = vsubq_f32(v_f3, v_group_zp);

            if constexpr (std::is_same_v<TA, float>) {
                float32x4_t v_a0 = vld1q_f32(a + i + offset + 0);
                float32x4_t v_a1 = vld1q_f32(a + i + offset + 4);
                float32x4_t v_a2 = vld1q_f32(a + i + offset + 8);
                float32x4_t v_a3 = vld1q_f32(a + i + offset + 12);

                v_f0 = vmulq_f32(v_f0, v_a0);
                v_f1 = vmulq_f32(v_f1, v_a1);
                v_f2 = vmulq_f32(v_f2, v_a2);
                v_f3 = vmulq_f32(v_f3, v_a3);

                group_sum += vaddvq_f32(v_f0);
                group_sum += vaddvq_f32(v_f1);
                group_sum += vaddvq_f32(v_f2);
                group_sum += vaddvq_f32(v_f3);

            } else if constexpr (std::is_same_v<TA, ov::float16>) {
                float16x8_t v_la0 = vld1q_f16(reinterpret_cast<const float16_t*>(a) + i + offset + 0);
                float16x8_t v_la1 = vld1q_f16(reinterpret_cast<const float16_t*>(a) + i + offset + 8);

                float32x4_t v_a0 = vcvt_f32_f16(vget_low_f16(v_la0));
                float32x4_t v_a1 = vcvt_f32_f16(vget_high_f16(v_la0));
                float32x4_t v_a2 = vcvt_f32_f16(vget_low_f16(v_la1));
                float32x4_t v_a3 = vcvt_f32_f16(vget_high_f16(v_la1));

                v_f0 = vmulq_f32(v_f0, v_a0);
                v_f1 = vmulq_f32(v_f1, v_a1);
                v_f2 = vmulq_f32(v_f2, v_a2);
                v_f3 = vmulq_f32(v_f3, v_a3);

                group_sum += vaddvq_f32(v_f0);
                group_sum += vaddvq_f32(v_f1);
                group_sum += vaddvq_f32(v_f2);
                group_sum += vaddvq_f32(v_f3);
            }
        }
        for (; i < group_size; i++) {
            group_sum += a[i + offset] * (b[i + offset] - group_zp);
        }
        sum += group_scale * group_sum;
        group_id += 1;
    }
    return sum;
#else
    while (group_id < n / group_size) {
        size_t i = 0;
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
        float group_sum = 0.0F;
        for (; i < group_size; i++) {
            group_sum += a[i + offset] * (b[i + offset] - group_zp);
        }
        sum += group_scale * group_sum;
        group_id += 1;
    }
    return sum;
#endif
}

inline void multiply_scalar(float* a, float* a_dst, const float val, const size_t size) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(val);
    __m256 v_a = {0};
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);
        mm256_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx2;
    }
    if (i < size) {
        auto mask = get_mask(static_cast<int>(size - i));
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);
        mm256_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
    svfloat32_t v_scale = svdup_n_f32(val);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < size) {
        if (size - i < vec_len_f32_sve()) {
            inc = size - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        svfloat32_t v_a = svld1_f32(pg, a + i);
        v_a = svmul_f32_z(pg, v_a, v_scale);
        svst1_f32(pg, a_dst + i, v_a);
        i += inc;
    }
#    else
    float32x4_t v_scale = vdupq_n_f32(val);
    while (i + vec_len_f32_neon <= size) {
        float32x4_t v_a = vld1q_f32(a + i);
        v_a = vmulq_f32(v_a, v_scale);
        vst1q_f32(a_dst + i, v_a);
        i += vec_len_f32_neon;
    }
#    endif
#endif
    for (; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
}

template <typename T, typename = std::enable_if_t<ov::intel_cpu::any_of_v<T, ov::bfloat16, ov::float16>>>
inline void multiply_scalar(const float* a, T* a_dst, const float val, const size_t size) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#else
    for (; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

#if defined(OPENVINO_ARCH_ARM64)
inline void multiply_scalar(ov::float16* a, float* a_dst, const ov::float16 val, const size_t size) {
    float16x4_t v_a_f16;
    float32x4_t v_a, v_res;
    float32x4_t v_val = vdupq_n_f32(static_cast<float>(val));
    size_t i = 0;

    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a_f16 = vld1_f16(reinterpret_cast<const float16_t*>(a + i));
        v_a = vcvt_f32_f16(v_a_f16);

        v_res = vmulq_f32(v_a, v_val);

        vst1q_f32(reinterpret_cast<float*>(a_dst + i), v_res);
    }

    for (; i < size; ++i) {
        const auto a_f32 = static_cast<float>(a[i]);
        a_dst[i] = a_f32 * static_cast<float>(val);
    }
}
inline void multiply_scalar_f32(ov::float16* a, ov::float16* a_dst, const ov::float16 val, const size_t size) {
    float32x4_t v_a, v_res;
    float32x4_t v_val = vdupq_n_f32(val);
    size_t i = 0;
    for (; i + vec_len_f32_neon <= size; i += vec_len_f32_neon) {
        v_a = __vld1q_f32((a + i));
        v_res = vmulq_f32(v_a, v_val);
        __vst1q_f32(a_dst + i, v_res);
    }
    auto val_f32 = static_cast<float>(val);
    for (; i < size; ++i) {
        auto _a_f32 = static_cast<float>(a[i]);
        auto _a_dst_f32 = val_f32 * _a_f32;
        a_dst[i] = static_cast<ov::float16>(_a_dst_f32);
    }
}
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void multiply_scalar(ov::float16* a, ov::float16* a_dst, const ov::float16 val, const size_t size) {
    size_t i = 0;
#    if defined(HAVE_SVE)
    svfloat16_t v_scale = svdup_n_f16(val);
    size_t inc = vec_len_f16_sve();
    svbool_t pg = svptrue_b16();

    while (i < size) {
        if (size - i < vec_len_f16_sve()) {
            inc = size - i;
            pg = svwhilelt_b16(0, static_cast<int>(inc));
        }
        svfloat16_t v_a = svld1_f16(pg, reinterpret_cast<float16_t*>(a + i));
        v_a = svmul_f16_z(pg, v_a, v_scale);
        svst1_f16(pg, reinterpret_cast<float16_t*>(a_dst + i), v_a);
        i += inc;
    }
#    else
    float16x8_t v_a, v_res;
    float16x8_t v_val = vdupq_n_f16(val);
    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));
        v_res = vmulq_f16(v_a, v_val);
        vst1q_f16(reinterpret_cast<float16_t*>(a_dst + i), v_res);
    }
#    endif
    for (; i < size; ++i) {
        a_dst[i] = a[i] * val;
    }
}
#endif

}  // namespace ov::Extensions::Cpu::XARCH
