// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_quant.hpp"
#include "attn_quant_kernel.hpp"
#include "common.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"

namespace ov::Extensions::Cpu::XARCH {

using namespace ov;

template <typename T>
static void find_minmax(const T* src, size_t n, float& min, float& max) {
    max = -FLT_MAX;
    min = FLT_MAX;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v0_max = _mm512_set1_ps(-FLT_MAX);
    auto v0_min = _mm512_set1_ps(FLT_MAX);
    auto v1_max = _mm512_set1_ps(-FLT_MAX);
    auto v1_min = _mm512_set1_ps(FLT_MAX);
    auto v2_max = _mm512_set1_ps(-FLT_MAX);
    auto v2_min = _mm512_set1_ps(FLT_MAX);
    auto v3_max = _mm512_set1_ps(-FLT_MAX);
    auto v3_min = _mm512_set1_ps(FLT_MAX);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        auto v2 = mm512_uni_loadu_ps(src + i + 2 * vec_len_f32_avx512);
        auto v3 = mm512_uni_loadu_ps(src + i + 3 * vec_len_f32_avx512);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        v1_max = _mm512_max_ps(v1_max, v1);
        v1_min = _mm512_min_ps(v1_min, v1);
        v2_max = _mm512_max_ps(v2_max, v2);
        v2_min = _mm512_min_ps(v2_min, v2);
        v3_max = _mm512_max_ps(v3_max, v3);
        v3_min = _mm512_min_ps(v3_min, v3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        v1_max = _mm512_max_ps(v1_max, v1);
        v1_min = _mm512_min_ps(v1_min, v1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        v0_max = _mm512_max_ps(v0_max, v0);
        v0_min = _mm512_min_ps(v0_min, v0);
        i += vec_len_f32_avx512;
    }
    v0_max = _mm512_max_ps(v0_max, v1_max);
    v0_min = _mm512_min_ps(v0_min, v1_min);
    v2_max = _mm512_max_ps(v2_max, v3_max);
    v2_min = _mm512_min_ps(v2_min, v3_min);
    v0_max = _mm512_max_ps(v0_max, v2_max);
    v0_min = _mm512_min_ps(v0_min, v2_min);
    max = _mm512_reduce_max_ps(v0_max);
    min = _mm512_reduce_min_ps(v0_min);
#elif defined(HAVE_AVX2)
    auto v0_max = _mm256_set1_ps(-FLT_MAX);
    auto v0_min = _mm256_set1_ps(FLT_MAX);
    auto v1_max = _mm256_set1_ps(-FLT_MAX);
    auto v1_min = _mm256_set1_ps(FLT_MAX);
    auto v2_max = _mm256_set1_ps(-FLT_MAX);
    auto v2_min = _mm256_set1_ps(FLT_MAX);
    auto v3_max = _mm256_set1_ps(-FLT_MAX);
    auto v3_min = _mm256_set1_ps(FLT_MAX);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        auto v2 = mm256_uni_loadu_ps(src + i + 2 * vec_len_f32_avx2);
        auto v3 = mm256_uni_loadu_ps(src + i + 3 * vec_len_f32_avx2);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        v1_max = _mm256_max_ps(v1_max, v1);
        v1_min = _mm256_min_ps(v1_min, v1);
        v2_max = _mm256_max_ps(v2_max, v2);
        v2_min = _mm256_min_ps(v2_min, v2);
        v3_max = _mm256_max_ps(v3_max, v3);
        v3_min = _mm256_min_ps(v3_min, v3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        v1_max = _mm256_max_ps(v1_max, v1);
        v1_min = _mm256_min_ps(v1_min, v1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        v0_max = _mm256_max_ps(v0_max, v0);
        v0_min = _mm256_min_ps(v0_min, v0);
        i += vec_len_f32_avx2;
    }
    v0_max = _mm256_max_ps(v0_max, v1_max);
    v0_min = _mm256_min_ps(v0_min, v1_min);
    v2_max = _mm256_max_ps(v2_max, v3_max);
    v2_min = _mm256_min_ps(v2_min, v3_min);
    v0_max = _mm256_max_ps(v0_max, v2_max);
    v0_min = _mm256_min_ps(v0_min, v2_min);
    hmax(v0_max);
    hmin(v0_min);
    max = _mm256_cvtss_f32(v0_max);
    min = _mm256_cvtss_f32(v0_min);
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        max = std::max(max, tmp);
        min = std::min(min, tmp);
    }
}

template <typename T>
static void quant_u8(const T* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    find_minmax(src, n, min, max);
    scale = (max - min) / 255;
    if (scale == 0) {
        scale = 0.0001f;
    }
    zp = -min / scale;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(1 / scale);
    auto v_zp = _mm512_set1_ps(zp);
    auto v_zero = _mm512_setzero_epi32();
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v = mm512_uni_loadu_ps(src + i);
        v = _mm512_fmadd_ps(v, v_scale, v_zp);
        auto v_i32 = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v_i32 = _mm512_max_epi32(v_i32, v_zero);
        _mm512_mask_cvtusepi32_storeu_epi8(dst + i, 0xffff, v_i32);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(1 / scale);
    auto v_zp = _mm256_set1_ps(zp);
    auto v_zero = _mm256_setzero_si256();
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        v = _mm256_fmadd_ps(v, v_scale, v_zp);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);
        v_i32 = _mm256_max_epi32(v_i32, v_zero);
        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        tmp = std::max(tmp / scale + zp, 0.0f);
        dst[i] = static_cast<uint8_t>(std::round(tmp));
    }
}

template <typename T>
static void find_params_by_channel(const T* src,
                                   size_t seq_dim,
                                   size_t hidden_dims,
                                   size_t src_stride,
                                   size_t dst_stride,
                                   float* scale,
                                   float* zp,
                                   size_t bits) {
    size_t j = 0;
    float integer_range = static_cast<float>((1 << bits) - 1);
#if defined(HAVE_AVX512F)
    for (; j + vec_len_f32_avx512 <= hidden_dims; j += vec_len_f32_avx512) {
        auto v_max = _mm512_set1_ps(-std::numeric_limits<float>::max());
        auto v_min = _mm512_set1_ps(std::numeric_limits<float>::max());
        for (size_t i = 0; i < seq_dim; i += 1) {
            auto v_cur = mm512_uni_loadu_ps(src + i * src_stride + j);
            v_max = _mm512_max_ps(v_max, v_cur);
            v_min = _mm512_min_ps(v_min, v_cur);
        }
        auto v_scale = _mm512_sub_ps(v_max, v_min);
        v_scale = _mm512_mul_ps(v_scale, _mm512_set1_ps(1 / integer_range));
        auto v_mask = _mm512_cmp_ps_mask(v_scale, _mm512_setzero_ps(), _CMP_EQ_OQ);
        v_scale = _mm512_mask_add_ps(v_scale, v_mask, v_scale, _mm512_set1_ps(0.0001f));
        auto v_zp = _mm512_mul_ps(v_min, _mm512_set1_ps(-1.0f));
        v_zp = _mm512_div_ps(v_zp, v_scale);

        _mm512_storeu_ps(scale + j, v_scale);
        _mm512_storeu_ps(zp + j, v_zp);
    }
#endif
#if defined(HAVE_AVX2)
    for (; j + vec_len_f32_avx2 <= hidden_dims; j += vec_len_f32_avx2) {
        auto v_max = _mm256_set1_ps(-std::numeric_limits<float>::max());
        auto v_min = _mm256_set1_ps(std::numeric_limits<float>::max());
        for (size_t i = 0; i < seq_dim; i++) {
            auto v_cur = mm256_uni_loadu_ps(src + i * src_stride + j);
            v_max = _mm256_max_ps(v_max, v_cur);
            v_min = _mm256_min_ps(v_min, v_cur);
        }
        auto v_scale = _mm256_sub_ps(v_max, v_min);
        v_scale = _mm256_mul_ps(v_scale, _mm256_set1_ps(1 / integer_range));
        auto v_cond = _mm256_cmp_ps(v_scale, _mm256_setzero_ps(), _CMP_EQ_OQ);
        auto v_comp = _mm256_and_ps(v_cond, _mm256_set1_ps(0.0001f));
        v_scale = _mm256_add_ps(v_scale, v_comp);
        auto v_zp = _mm256_mul_ps(v_min, _mm256_set1_ps(-1.0f));
        v_zp = _mm256_div_ps(v_zp, v_scale);
        _mm256_storeu_ps(scale + j, v_scale);
        _mm256_storeu_ps(zp + j, v_zp);
    }
#endif
    for (; j < hidden_dims; j++) {
        float max = -std::numeric_limits<float>::max();
        float min = std::numeric_limits<float>::max();
        for (size_t i = 0; i < seq_dim; i++) {
            float tmp = src[i * src_stride + j];
            max = std::max(max, tmp);
            min = std::min(min, tmp);
        }
        float temp_scale = (max - min) / integer_range;
        if (temp_scale == 0) {
            temp_scale = 0.0001f;
        }
        float temp_zp = -min / temp_scale;
        scale[j] = temp_scale;
        zp[j] = temp_zp;
    }
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::u8, bool> = true>
static void quantize_by_channel(const T* src,
                                uint8_t* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp) {
    find_params_by_channel(src, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp, 8);
    // quantize
    size_t j = 0;
#if defined(HAVE_AVX512F)
    for (; j + vec_len_f32_avx512 <= hidden_dims; j += vec_len_f32_avx512) {
        auto v_scale = mm512_uni_loadu_ps(scale + j);
        v_scale = _mm512_div_ps(_mm512_set1_ps(1.0f), v_scale);
        auto v_zero = _mm512_setzero_epi32();
        auto v_zp = mm512_uni_loadu_ps(zp + j);
        for (size_t i = 0; i < seq_dim; i++) {
            auto v = mm512_uni_loadu_ps(src + i * src_stride + j);
            v = _mm512_fmadd_ps(v, v_scale, v_zp);
            auto v_i32 = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v_i32 = _mm512_max_epi32(v_i32, v_zero);
            _mm512_mask_cvtusepi32_storeu_epi8(dst + i * dst_stride + j, 0xffff, v_i32);
        }
    }
#endif

#if defined(HAVE_AVX2)
    for (; j + vec_len_f32_avx2 <= hidden_dims; j += vec_len_f32_avx2) {
        auto v_scale = mm256_uni_loadu_ps(scale + j);
        v_scale = _mm256_div_ps(_mm256_set1_ps(1.0f), v_scale);
        auto v_zp = mm256_uni_loadu_ps(zp + j);
        auto v_zero = _mm256_setzero_si256();
        for (size_t i = 0; i < seq_dim; i++) {
            auto v = mm256_uni_loadu_ps(src + i * src_stride + j);
            v = _mm256_fmadd_ps(v, v_scale, v_zp);
            v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
            auto v_i32 = _mm256_cvtps_epi32(v);
            v_i32 = _mm256_max_epi32(v_i32, v_zero);
            auto high4 = _mm256_extractf128_si256(v_i32, 1);
            auto low4 = _mm256_castsi256_si128(v_i32);
            auto packed = _mm_packs_epi32(low4, high4);
            packed = _mm_packus_epi16(packed, packed);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i * dst_stride + j), packed);
        }
    }
#endif
    for (; j < hidden_dims; j++) {
        for (size_t i = 0; i < seq_dim; ++i) {
            float tmp = src[i * src_stride + j];
            tmp = std::max(tmp / scale[j] + zp[j], 0.0f);
            dst[i * dst_stride + j] = static_cast<uint8_t>(std::round(tmp));
        }
    }
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::u4, bool> = true>
static void quantize_by_channel(const T* src,
                                uint8_t* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp) {
    find_params_by_channel(src, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp, 4);
    size_t j = 0;
#if defined(HAVE_AVX512F)
    for (; j + vec_len_f32_avx512 * 2 <= hidden_dims; j += vec_len_f32_avx512 * 2) {
        auto v_scale0 = mm512_uni_loadu_ps(scale + j);
        v_scale0 = _mm512_div_ps(_mm512_set1_ps(1.0f), v_scale0);
        auto v_scale1 = mm512_uni_loadu_ps(scale + j + vec_len_f32_avx512);
        v_scale1 = _mm512_div_ps(_mm512_set1_ps(1.0f), v_scale1);

        auto v_zero = _mm512_setzero_epi32();
        auto v_upper = _mm512_set1_epi32(15);

        auto v_zp0 = mm512_uni_loadu_ps(zp + j);
        auto v_zp1 = mm512_uni_loadu_ps(zp + j + vec_len_f32_avx512);
        for (size_t i = 0; i < seq_dim; i++) {
            auto v0 = mm512_uni_loadu_ps(src + i * src_stride + j);
            auto v1 = mm512_uni_loadu_ps(src + i * src_stride + j + vec_len_f32_avx512);
            v0 = _mm512_fmadd_ps(v0, v_scale0, v_zp0);
            v1 = _mm512_fmadd_ps(v1, v_scale1, v_zp1);
            auto v0_i32 = _mm512_cvt_roundps_epi32(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            auto v1_i32 = _mm512_cvt_roundps_epi32(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            v0_i32 = _mm512_max_epi32(v0_i32, v_zero);
            v1_i32 = _mm512_max_epi32(v1_i32, v_zero);
            v0_i32 = _mm512_min_epi32(v0_i32, v_upper);
            v1_i32 = _mm512_min_epi32(v1_i32, v_upper);
            mm512_storeu_u4(dst + i * dst_stride + j / 2, v0_i32, v1_i32);
        }
    }
#endif
#if defined(HAVE_AVX2)
    for (; j + vec_len_f32_avx2 * 2 <= hidden_dims; j += vec_len_f32_avx2 * 2) {
        auto v_scale0 = mm256_uni_loadu_ps(scale + j);
        v_scale0 = _mm256_div_ps(_mm256_set1_ps(1.0f), v_scale0);
        auto v_scale1 = mm256_uni_loadu_ps(scale + j + vec_len_f32_avx2);
        v_scale1 = _mm256_div_ps(_mm256_set1_ps(1.0f), v_scale1);

        auto v_zero = _mm256_setzero_si256();
        auto v_upper = _mm256_set1_epi32(15);

        auto v_zp0 = mm256_uni_loadu_ps(zp + j);
        auto v_zp1 = mm256_uni_loadu_ps(zp + j + vec_len_f32_avx2);
        for (size_t i = 0; i < seq_dim; i++) {
            auto v0 = mm256_uni_loadu_ps(src + i * src_stride + j);
            auto v1 = mm256_uni_loadu_ps(src + i * src_stride + j + vec_len_f32_avx2);
            v0 = _mm256_fmadd_ps(v0, v_scale0, v_zp0);
            v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
            v1 = _mm256_fmadd_ps(v1, v_scale1, v_zp1);
            v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);

            auto v0_i32 = _mm256_cvtps_epi32(v0);
            auto v1_i32 = _mm256_cvtps_epi32(v1);
            v0_i32 = _mm256_max_epi32(v0_i32, v_zero);
            v1_i32 = _mm256_max_epi32(v1_i32, v_zero);
            v0_i32 = _mm256_min_epi32(v0_i32, v_upper);
            v1_i32 = _mm256_min_epi32(v1_i32, v_upper);
            mm256_storeu_u4(dst + i * dst_stride + j / 2, v0_i32, v1_i32);
        }
    }
#endif
    for (; j < hidden_dims; j++) {
        for (size_t i = 0; i < seq_dim; i++) {
            float tmp = src[i * src_stride + j];
            tmp = std::round(tmp / scale[j] + zp[j]);
            uint8_t src_val = std::min(static_cast<uint8_t>(15), static_cast<uint8_t>(tmp));
            src_val = std::max(static_cast<uint8_t>(0), static_cast<uint8_t>(tmp));
            uint8_t dst_val = j % 2 == 0 ? 0 : dst[i * dst_stride + j / 2];
            dst_val = insert_half_byte(dst_val, src_val, static_cast<uint8_t>(j % 2));
            dst[i * dst_stride + j / 2] = dst_val;
        }
    }
}

template <typename T>
static void quant_u4(const T* src, void* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    find_minmax(src, n, min, max);
    auto dst_ptr = reinterpret_cast<uint8_t*>(dst);
    scale = (max - min) / ((1 << 4) - 1);
    if (scale == 0) {
        scale = 0.0001f;
    }
    zp = -min / scale;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(1 / scale);
    auto v_zp = _mm512_set1_ps(zp);
    auto v_zero = _mm512_setzero_epi32();
    auto v_upper = _mm512_set1_epi32(15);
    for (; i + 2 * vec_len_f32_avx512 <= n; i += 2 * vec_len_f32_avx512) {
        auto v0 = mm512_uni_loadu_ps(src + i);
        auto v1 = mm512_uni_loadu_ps(src + i + vec_len_f32_avx512);
        v0 = _mm512_fmadd_ps(v0, v_scale, v_zp);
        v1 = _mm512_fmadd_ps(v1, v_scale, v_zp);
        auto v0_i32 = _mm512_cvt_roundps_epi32(v0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        auto v1_i32 = _mm512_cvt_roundps_epi32(v1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v0_i32 = _mm512_max_epi32(v0_i32, v_zero);
        v1_i32 = _mm512_max_epi32(v1_i32, v_zero);
        v0_i32 = _mm512_min_epi32(v0_i32, v_upper);
        v1_i32 = _mm512_min_epi32(v1_i32, v_upper);
        mm512_storeu_u4(dst_ptr + i / 2, v0_i32, v1_i32);
    }
#endif
#if defined(HAVE_AVX2)
    auto v256_zero = _mm256_set1_epi32(0);
    auto v256_upper = _mm256_set1_epi32(15);
    auto v256_scale = _mm256_set1_ps(1 / scale);
    auto v256_zp = _mm256_set1_ps(zp);
    for (; i + vec_len_f32_avx2 * 2 <= n; i += vec_len_f32_avx2 * 2) {
        auto v0 = mm256_uni_loadu_ps(src + i);
        auto v1 = mm256_uni_loadu_ps(src + i + vec_len_f32_avx2);
        v0 = _mm256_fmadd_ps(v0, v256_scale, v256_zp);
        v1 = _mm256_fmadd_ps(v1, v256_scale, v256_zp);
        v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST);
        v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST);

        auto v0_i32 = _mm256_cvtps_epi32(v0);
        auto v1_i32 = _mm256_cvtps_epi32(v1);
        v0_i32 = _mm256_max_epi32(v0_i32, v256_zero);
        v1_i32 = _mm256_max_epi32(v1_i32, v256_zero);
        v0_i32 = _mm256_min_epi32(v0_i32, v256_upper);
        v1_i32 = _mm256_min_epi32(v1_i32, v256_upper);
        mm256_storeu_u4(dst_ptr + i / 2, v0_i32, v1_i32);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        uint8_t src_val = std::min(static_cast<uint8_t>(15), static_cast<uint8_t>(std::round(tmp / scale + zp)));
        src_val = std::max(static_cast<uint8_t>(0), static_cast<uint8_t>(std::round(tmp / scale + zp)));
        uint8_t dst_val = i % 2 == 0 ? 0 : dst_ptr[i / 2];
        dst_val = insert_half_byte(dst_val, src_val, static_cast<uint8_t>(i % 2));
        dst_ptr[i / 2] = dst_val;
    }
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::u8, bool> = true>
static void quantize(const T* src, uint8_t* dst, size_t n, float* scale_zp) {
    quant_u8(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::u4, bool> = true>
static void quantize(const T* src, void* dst, size_t n, float* scale_zp) {
    quant_u4(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T, ov::element::Type_t DST_PREC>
static void quantize_block_by_dims(const ov::intel_cpu::PlainTensor& src,
                                   const ov::intel_cpu::PlainTensor& dst,
                                   size_t b,
                                   size_t h,
                                   size_t m,
                                   size_t block_number,
                                   size_t block_offset,
                                   size_t groupe_size) {
    // The cache layout is [scale0, zp0]|[group0]|[scale1, zp1]|[group1].....
    // dst_offset is the offset among groups. The addition of 2 * sizeof(float) aims to shift to next group
    // base pointer points to the base address of next group.
    // base +  2 * sizeof(float) aims to skip the scale/zp within the group.
    constexpr size_t sub_byte_multiplier = DST_PREC == ov::element::u4 ? 2 : 1;
    size_t S = src.m_dims[3];
    for (size_t src_offset = 0, dst_offset = 0; src_offset < S;
         src_offset += groupe_size, dst_offset += groupe_size / sub_byte_multiplier + sizeof(float) + sizeof(float)) {
        auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, block_offset, 0);
        base += dst_offset;
        auto p = reinterpret_cast<float*>(base);
        uint8_t* ptr = base + sizeof(float) * 2;
        quantize<T, DST_PREC>(src.ptr<T>(b, h, m, src_offset), ptr, groupe_size, p);
    }
}

template <typename T, ov::element::Type_t DST_PREC>
static void quantize_block_by_channel(const ov::intel_cpu::PlainTensor& src,
                                      const ov::intel_cpu::PlainTensor& dst,
                                      const ov::intel_cpu::PlainTensor& past_lens,
                                      const ov::intel_cpu::PlainTensor& subsequence_begins,
                                      const ov::intel_cpu::PlainTensor& block_indices,
                                      const ov::intel_cpu::PlainTensor& block_indices_begins,
                                      float* buffer,
                                      size_t sub_seq_id,
                                      size_t h) {
    // scale f32[S] zp f32[S] offset in bytes
    auto S = src.m_dims[3];
    size_t params_offset = 2 * sizeof(float) * S;
    auto past_len = past_lens.ptr<int32_t>()[sub_seq_id];
    auto q_len = subsequence_begins.ptr<int32_t>()[sub_seq_id + 1] - subsequence_begins.ptr<int32_t>()[sub_seq_id];
    auto block_number_start = block_indices_begins.ptr<int32_t>()[sub_seq_id];
    const size_t block_size = dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(DST_PREC);
    size_t m = 0;
    // Quantized cache is either u8/u4, the plain memory is both uint8,
    // Here we use stride_bytes instead of stride which consider divide sub_byte_multiplier automatically.
    if (past_len == 0) {
        auto total_blocks =
            block_indices_begins.ptr<int32_t>()[sub_seq_id + 1] - block_indices_begins.ptr<int32_t>()[sub_seq_id];
        parallel_for(total_blocks, [&](int32_t block_count) {
            auto block_id = block_number_start + block_count;
            auto block_number = block_indices.ptr<int32_t>()[block_id];
            auto token_num = (block_id == (block_indices_begins.ptr<int32_t>()[sub_seq_id + 1] - 1))
                                 ? (q_len - block_count * block_size)
                                 : block_size;
            size_t b_in_tokens = subsequence_begins.ptr<int32_t>()[sub_seq_id] + block_count * block_size;
            auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, 0, 0);
            auto p_scales = reinterpret_cast<float*>(base);
            auto p_zps = p_scales + S;
            auto p_data = base + params_offset;
            quantize_by_channel<T, DST_PREC>(src.ptr<T>(b_in_tokens, h, m),
                                             p_data,
                                             token_num,
                                             S,
                                             src.stride(0),
                                             dst.stride_bytes(2),
                                             p_scales,
                                             p_zps);
        });
    } else {
        auto prev_nums = past_len % block_size;
        size_t block_offset = block_number_start + past_len / block_size;
        auto total_blocks = block_indices_begins.ptr<int32_t>()[sub_seq_id + 1] - block_offset;
        parallel_for(total_blocks, [&](size_t block_id) {
            size_t b_in_tokens = subsequence_begins.ptr<int32_t>()[sub_seq_id];
            auto block_number = block_indices.ptr<int32_t>()[block_id + block_offset];
            auto base = dst.ptr<uint8_t, DST_PREC>(block_number, h, 0, 0);
            auto p_scales = reinterpret_cast<float*>(base);
            auto p_zps = p_scales + S;
            auto p_data = base + params_offset;
            size_t valid_length = 0;
            bool is_first_block = block_id == 0;
            if (is_first_block) {
                valid_length = std::min(static_cast<size_t>(q_len), block_size - prev_nums);
            } else {
                // first block may have pre-filled data, the offset of first block is prev_nums, following
                // blocks have offset = block_size
                valid_length = std::min(static_cast<size_t>(q_len) + prev_nums - block_size * block_id, block_size);
            }
            if (is_first_block && prev_nums) {
                attn_dequant_by_channel_kernel<float, DST_PREC>(p_data,
                                                                buffer,
                                                                prev_nums,
                                                                S,
                                                                dst.stride_bytes(2),
                                                                S,
                                                                p_scales,
                                                                p_zps);
                cvt_copy(buffer + prev_nums * S, src.ptr<T>(b_in_tokens, h, m), valid_length, S, src.stride(0), S);
                quantize_by_channel<float, DST_PREC>(buffer,
                                                     p_data,
                                                     prev_nums + valid_length,
                                                     S,
                                                     S,
                                                     dst.stride_bytes(2),
                                                     p_scales,
                                                     p_zps);
            } else {
                quantize_by_channel<T, DST_PREC>(src.ptr<T>(b_in_tokens, h, m),
                                                 p_data,
                                                 valid_length,
                                                 S,
                                                 src.stride(0),
                                                 dst.stride_bytes(2),
                                                 p_scales,
                                                 p_zps);
            }
        });
    }
}

template <typename T, typename T2>
static void attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                          const ov::intel_cpu::PlainTensor& v_src,
                          const ov::intel_cpu::PlainTensor& k_dst,
                          const ov::intel_cpu::PlainTensor& v_dst,
                          const size_t L0,
                          float* temp_buffer,
                          const ov::intel_cpu::PlainTensor& k_scale_zp,
                          const ov::intel_cpu::PlainTensor& v_scale_zp,
                          const bool quant_key_by_channel,
                          const size_t key_group_size,
                          const size_t value_group_size) {
    // For compatibility, all input_kvs are permuted to BHLS
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3], SV = v_src.m_dims[3];
    if (quant_key_by_channel) {
        if (L0 == 0) {
            parallel_for3d(ov::intel_cpu::div_up(L1, key_group_size), B, H, [&](size_t group_id, size_t b, size_t h) {
                quantize_by_channel<T, intel_cpu::precision_of<T2>::value>(
                    k_src.ptr<T>(b, h, group_id * key_group_size),
                    k_dst.ptr<T2>(b, h, group_id * key_group_size),
                    std::min(key_group_size, L1 - group_id * key_group_size),
                    S,
                    k_src.m_strides[2],
                    k_dst.m_strides[2],
                    k_scale_zp.ptr<float>(group_id * 2, b, h),
                    k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
            });
        } else {
            size_t group_id = L0 / key_group_size;
            size_t prev_nums = L0 % key_group_size;
            parallel_for2d(B, H, [&](size_t b, size_t h) {
                auto thread_id = parallel_get_thread_num();
                float* thread_temp_buffer = temp_buffer + thread_id * key_group_size * S;
                size_t remaining_group_size = prev_nums ? (key_group_size - prev_nums) : 0;
                if (prev_nums) {
                    attn_dequant_by_channel_kernel<float, intel_cpu::precision_of<T2>::value>(
                        k_dst.ptr<uint8_t>(b, h, group_id * key_group_size),
                        thread_temp_buffer,
                        prev_nums,
                        S,
                        k_dst.stride_bytes(2),
                        S,
                        k_scale_zp.ptr<float>(group_id * 2, b, h),
                        k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                    remaining_group_size = std::min(remaining_group_size, L1);
                    cvt_copy(thread_temp_buffer + prev_nums * S,
                             k_src.ptr<T>(b, h),
                             remaining_group_size,
                             S,
                             k_src.m_strides[2],
                             S);
                    quantize_by_channel<float, intel_cpu::precision_of<T2>::value>(
                        thread_temp_buffer,
                        k_dst.ptr<T2>(b, h, group_id * key_group_size),
                        remaining_group_size + prev_nums,
                        S,
                        S,
                        k_dst.m_strides[2],
                        k_scale_zp.ptr<float>(group_id * 2, b, h),
                        k_scale_zp.ptr<float>(group_id * 2 + 1, b, h));
                }

                if (L1 > remaining_group_size) {
                    size_t new_seq = L1 - remaining_group_size;
                    for (size_t new_group_id = prev_nums ? group_id + 1 : group_id, src_offset = 0;
                         new_group_id < ov::intel_cpu::div_up(L0 + L1, key_group_size);
                         new_group_id++, src_offset += key_group_size) {
                        quantize_by_channel<T, intel_cpu::precision_of<T2>::value>(
                            k_src.ptr<T>(b, h, remaining_group_size + src_offset),
                            k_dst.ptr<T2>(b, h, new_group_id * key_group_size),
                            std::min(key_group_size, new_seq - src_offset),
                            S,
                            k_src.m_strides[2],
                            k_dst.m_strides[2],
                            k_scale_zp.ptr<float>(new_group_id * 2, b, h),
                            k_scale_zp.ptr<float>(new_group_id * 2 + 1, b, h));
                    }
                }
            });
        }
    } else {
        parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
            auto p_k = k_scale_zp.ptr<float>(L0 + m, b, h);
            for (size_t group_id = 0; group_id < S / key_group_size; group_id++) {
                quant_u8(k_src.ptr<T>(b, h, m, group_id * key_group_size),
                         k_dst.ptr<T2>(b, h, L0 + m, group_id * key_group_size),
                         key_group_size,
                         p_k[group_id * 2],
                         p_k[group_id * 2 + 1]);
            }
        });
    }
    parallel_for3d(L1, B, H, [&](size_t m, size_t b, size_t h) {
        auto p_v = v_scale_zp.ptr<float>(L0 + m, b, h);
        for (size_t group_id = 0; group_id < SV / value_group_size; group_id++) {
            quant_u8(v_src.ptr<T>(b, h, m, group_id * value_group_size),
                     v_dst.ptr<T2>(b, h, L0 + m, group_id * value_group_size),
                     value_group_size,
                     p_v[group_id * 2],
                     p_v[group_id * 2 + 1]);
        }
    });
}

template <typename T, ov::element::Type_t KEY_DST_PREC, ov::element::Type_t VALUE_DST_PREC>
static void paged_attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                                const ov::intel_cpu::PlainTensor& v_src,
                                const ov::intel_cpu::PlainTensor& k_dst,
                                const ov::intel_cpu::PlainTensor& v_dst,
                                const ov::intel_cpu::PlainTensor& past_lens,
                                const ov::intel_cpu::PlainTensor& subsequence_begins,
                                const ov::intel_cpu::PlainTensor& block_indices,
                                const ov::intel_cpu::PlainTensor& block_indices_begins,
                                const ov::intel_cpu::PlainTensor& slot_mapping,
                                ov::intel_cpu::PlainTensor& temp_buffer,
                                const bool quant_key_by_channel,
                                const bool quant_value_by_channel,
                                const size_t key_group_size,
                                const size_t value_group_size) {
    const size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2];
    const size_t block_size = quant_key_by_channel
                                  ? k_dst.m_dims[2] - 2 * sizeof(float) * get_sub_byte_multiplier(KEY_DST_PREC)
                                  : k_dst.m_dims[2];
    if (quant_key_by_channel) {
        parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
            float* buffer = temp_buffer.ptr<float>(parallel_get_thread_num());
            quantize_block_by_channel<T, KEY_DST_PREC>(k_src,
                                                       k_dst,
                                                       past_lens,
                                                       subsequence_begins,
                                                       block_indices,
                                                       block_indices_begins,
                                                       buffer,
                                                       sub_seq_id,
                                                       h);
        });
    } else {
        parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
            auto slot = slot_mapping.ptr<int32_t>(b)[m];
            if (slot < 0) {
                return;
            }
            auto block_number = slot / block_size;
            auto block_offset = slot % block_size;
            quantize_block_by_dims<T, KEY_DST_PREC>(k_src, k_dst, b, h, m, block_number, block_offset, key_group_size);
        });
    }
    // quant value
    if (quant_value_by_channel) {
        parallel_for2d(past_lens.size(0), H, [&](size_t sub_seq_id, size_t h) {
            float* buffer = temp_buffer.ptr<float>(parallel_get_thread_num());
            quantize_block_by_channel<T, VALUE_DST_PREC>(v_src,
                                                         v_dst,
                                                         past_lens,
                                                         subsequence_begins,
                                                         block_indices,
                                                         block_indices_begins,
                                                         buffer,
                                                         sub_seq_id,
                                                         h);
        });
    } else {
        parallel_for3d(B, L1, H, [&](size_t b, size_t m, size_t h) {
            auto slot = slot_mapping.ptr<int32_t>(b)[m];
            if (slot < 0) {
                return;
            }
            auto block_number = slot / block_size;
            auto block_offset = slot % block_size;
            quantize_block_by_dims<T,
                                   VALUE_DST_PREC>(v_src, v_dst, b, h, m, block_number, block_offset, value_group_size);
        });
    }
}

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  float* temp_buffer,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp,
                  const size_t L0,
                  const bool quant_k_by_channel,
                  const size_t k_group_size,
                  const size_t v_group_size) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<float, uint8_t>(k_src,
                                      v_src,
                                      k_dst,
                                      v_dst,
                                      L0,
                                      temp_buffer,
                                      k_scale_zp,
                                      v_scale_zp,
                                      quant_k_by_channel,
                                      k_group_size,
                                      v_group_size);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::bfloat16, uint8_t>(k_src,
                                             v_src,
                                             k_dst,
                                             v_dst,
                                             L0,
                                             temp_buffer,
                                             k_scale_zp,
                                             v_scale_zp,
                                             quant_k_by_channel,
                                             k_group_size,
                                             v_group_size);
    } else if (k_src.get_precision() == ov::element::f16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::float16, uint8_t>(k_src,
                                            v_src,
                                            k_dst,
                                            v_dst,
                                            L0,
                                            temp_buffer,
                                            k_scale_zp,
                                            v_scale_zp,
                                            quant_k_by_channel,
                                            k_group_size,
                                            v_group_size);
    } else {
        OPENVINO_THROW("unsupport src type: ",
                       k_src.get_precision(),
                       ", dst type: ",
                       k_dst.get_precision(),
                       " in attn_quantkv");
    }
}

void paged_attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                        const ov::intel_cpu::PlainTensor& v_src,
                        const ov::intel_cpu::PlainTensor& k_dst,
                        const ov::intel_cpu::PlainTensor& v_dst,
                        const ov::intel_cpu::PlainTensor& past_lens,
                        const ov::intel_cpu::PlainTensor& subsequence_begins,
                        const ov::intel_cpu::PlainTensor& block_indices,
                        const ov::intel_cpu::PlainTensor& block_indices_begins,
                        const ov::intel_cpu::PlainTensor& slot_mapping,
                        ov::intel_cpu::PlainTensor& temp_buffer,
                        const bool quant_key_by_channel,
                        const bool quant_value_by_channel,
                        const size_t key_group_size,
                        const size_t value_group_size) {
    using function_type = void (*)(const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   const ov::intel_cpu::PlainTensor&,
                                   ov::intel_cpu::PlainTensor&,
                                   const bool,
                                   const bool,
                                   const size_t,
                                   const size_t);
    static constexpr function_type funcs_fp32[] = {
        paged_attn_quant_mt<float, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<float, ov::element::u8, ov::element::u4>,
        paged_attn_quant_mt<float, ov::element::u4, ov::element::u8>,
        paged_attn_quant_mt<float, ov::element::u4, ov::element::u4>,
    };
    static constexpr function_type funcs_bf16[] = {
        paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<ov::bfloat16, ov::element::u8, ov::element::u4>,
        paged_attn_quant_mt<ov::bfloat16, ov::element::u4, ov::element::u8>,
        paged_attn_quant_mt<ov::bfloat16, ov::element::u4, ov::element::u4>,
    };
    static constexpr function_type funcs_f16[] = {
        paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u8>,
        paged_attn_quant_mt<ov::float16, ov::element::u8, ov::element::u4>,
        paged_attn_quant_mt<ov::float16, ov::element::u4, ov::element::u8>,
        paged_attn_quant_mt<ov::float16, ov::element::u4, ov::element::u4>,
    };
    size_t dispatch = 0;
    if (k_dst.get_precision() == ov::element::u4) {
        dispatch |= 0x02;
    }
    if (v_dst.get_precision() == ov::element::u4) {
        dispatch |= 0x01;
    }
    if (k_src.get_precision() == ov::element::f32) {
        funcs_fp32[dispatch](k_src,
                             v_src,
                             k_dst,
                             v_dst,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             slot_mapping,
                             temp_buffer,
                             quant_key_by_channel,
                             quant_value_by_channel,
                             key_group_size,
                             value_group_size);
    } else if (k_src.get_precision() == ov::element::bf16) {
        funcs_bf16[dispatch](k_src,
                             v_src,
                             k_dst,
                             v_dst,
                             past_lens,
                             subsequence_begins,
                             block_indices,
                             block_indices_begins,
                             slot_mapping,
                             temp_buffer,
                             quant_key_by_channel,
                             quant_value_by_channel,
                             key_group_size,
                             value_group_size);
    } else if (k_src.get_precision() == ov::element::f16) {
        funcs_f16[dispatch](k_src,
                            v_src,
                            k_dst,
                            v_dst,
                            past_lens,
                            subsequence_begins,
                            block_indices,
                            block_indices_begins,
                            slot_mapping,
                            temp_buffer,
                            quant_key_by_channel,
                            quant_value_by_channel,
                            key_group_size,
                            value_group_size);
    }
}

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    quant_u8(src, dst, n, scale, zp);
}

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
    attn_dequant_kernel<float, ov::element::u8>(src, dst, n, scale, zp);
}

void attn_quant_by_channel_u8(const float* src,
                              uint8_t* dst,
                              size_t seq_dim,
                              size_t hidden_dims,
                              size_t src_stride,
                              size_t dst_stride,
                              float* scale,
                              float* zp) {
    quantize_by_channel<float, ov::element::u8>(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

void attn_dequant_by_channel_u8(const uint8_t* src,
                                float* dst,
                                size_t seq_dim,
                                size_t hidden_dims,
                                size_t src_stride,
                                size_t dst_stride,
                                float* scale,
                                float* zp) {
    attn_dequant_by_channel_kernel<float,
                                   ov::element::u8>(src, dst, seq_dim, hidden_dims, src_stride, dst_stride, scale, zp);
}

}  // namespace ov::Extensions::Cpu::XARCH
