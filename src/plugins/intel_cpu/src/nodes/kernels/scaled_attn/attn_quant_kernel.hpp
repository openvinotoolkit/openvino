// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"
#include "openvino/core/type/element_type.hpp"
#include "utils/plain_tensor.hpp"
#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>
#if defined(HAVE_SVE)
#    include "arm_sve.h"
#endif

namespace ov::Extensions::Cpu::XARCH {

template <typename T>
void find_minmax(const T* src, size_t n, float& min, float& max) {
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
void quant_u8(const T* src, uint8_t* dst, size_t n, float& scale, float& zp) {
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
void quant_i8(const T* src, int8_t* dst, size_t n, float& scale) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
    find_minmax(src, n, min, max);
    float max_abs = std::max(std::abs(min), std::abs(max));
    scale = max_abs / ((1 << 7) - 1);
    if (scale == 0) {
        scale = 0.0001f;
    }
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(1 / scale);
    auto v_upper = _mm512_set1_epi32(127);
    auto v_lower = _mm512_set1_epi32(-128);
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v = mm512_uni_loadu_ps(src + i);
        v = _mm512_mul_ps(v, v_scale);
        auto v_i32 = _mm512_cvt_roundps_epi32(v, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        v_i32 = _mm512_max_epi32(v_i32, v_lower);
        v_i32 = _mm512_min_epi32(v_i32, v_upper);
        _mm512_mask_cvtepi32_storeu_epi8(dst + i, 0xffff, v_i32);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(1 / scale);
    auto v_upper = _mm256_set1_epi32(127);
    auto v_lower = _mm256_set1_epi32(-128);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        v = _mm256_mul_ps(v, v_scale);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);
        v_i32 = _mm256_max_epi32(v_i32, v_lower);
        v_i32 = _mm256_min_epi32(v_i32, v_upper);
        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packs_epi16(packed, packed);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        tmp = std::round(tmp / scale);
        tmp = std::max(tmp, -128.0f);
        tmp = std::min(tmp, 127.0f);
        dst[i] = static_cast<int8_t>(tmp);
    }
}

template <typename T>
void find_params_by_channel(const T* src,
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
void quantize_by_channel(const T* src,
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
void quantize_by_channel(const T* src,
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
void quant_u4(const T* src, void* dst, size_t n, float& scale, float& zp) {
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
void quantize(const T* src, uint8_t* dst, size_t n, float* scale_zp) {
    quant_u8(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::i8, bool> = true>
void quantize(const T* src, uint8_t* dst, size_t n, float* scale_zp) {
    quant_i8(src, reinterpret_cast<int8_t*>(dst), n, *scale_zp);
}

template <typename T, ov::element::Type_t DST_PREC, std::enable_if_t<DST_PREC == ov::element::u4, bool> = true>
void quantize(const T* src, void* dst, size_t n, float* scale_zp) {
    quant_u4(src, dst, n, *scale_zp, *(scale_zp + 1));
}

template <typename T, ov::element::Type_t DST_PREC>
void quantize_q_by_dims(const ov::intel_cpu::PlainTensor& src,
                        const ov::intel_cpu::PlainTensor& dst,
                        size_t b,
                        size_t h,
                        size_t group_size) {
    // The cache layout is [scale0, zp0]|[group0]|[scale1, zp1]|[group1].....
    // dst_offset is the offset among groups. The addition of 2 * sizeof(float) aims to shift to next group
    // base pointer points to the base address of next group.
    // base +  2 * sizeof(float) aims to skip the scale/zp within the group.
    constexpr size_t sub_byte_multiplier = DST_PREC == ov::element::u4 ? 2 : 1;
    size_t S = src.m_dims[3];
    constexpr size_t param_size = sizeof(float) * (DST_PREC == ov::element::i8 ? 1 : 2);
    size_t m = 0;
    for (size_t src_offset = 0, dst_offset = 0; src_offset < S;
         src_offset += group_size, dst_offset += group_size / sub_byte_multiplier + param_size) {
        auto base = dst.ptr<uint8_t, DST_PREC>(b, h, 0);
        base += dst_offset;
        auto p = reinterpret_cast<float*>(base);
        uint8_t* ptr = base + param_size;
        quantize<T, DST_PREC>(src.ptr<T>(b, h, m, src_offset), ptr, group_size, p);
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<SRC_PREC == ov::element::u8, bool>::type = true>
void attn_dequant_kernel(const void* src, TDST* dst, size_t n, float* params) {
    size_t i = 0;
    // loadu_si128/epi64 does not support const qualifier
    uint8_t* src_nc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));
    float scale = params[0];
    float zp = params[1];
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
          typename std::enable_if<SRC_PREC == ov::element::i8, bool>::type = true>
void attn_dequant_kernel(const void* src, TDST* dst, size_t n, float* params) {
    size_t i = 0;
    // loadu_si128/epi64 does not support const qualifier
    int8_t* src_nc = const_cast<int8_t*>(reinterpret_cast<const int8_t*>(src));
    float scale = params[0];
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(scale);
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_512 = _mm512_cvtepi8_epi32(v0_128);
        auto v0_value = _mm512_cvtepi32_ps(v0_512);
        auto v0_out = _mm512_mul_ps(v0_value, v_scale);
        mm512_uni_storeu_ps(dst + i, v0_out);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(scale);
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src_nc + i));
        auto v0_256 = _mm256_cvtepi8_epi32(v0_128);
        auto v0_value = _mm256_cvtepi32_ps(v0_256);
        auto v0_out = _mm256_mul_ps(v0_value, v_scale);
        mm256_uni_storeu_ps(dst + i, v0_out);
    }
#endif
    for (; i < n; ++i) {
        float tmp = src_nc[i];
        tmp *= scale;
        dst[i] = tmp;
    }
}

template <typename TDST,
          ov::element::Type_t SRC_PREC,
          typename std::enable_if<SRC_PREC == ov::element::u4, bool>::type = true>
void attn_dequant_kernel(const void* src, TDST* dst, size_t n, float* params) {
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
    uint8_t* src_nc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));
    float scale = params[0];
    float zp = params[1];
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
    for (; i < n; ++i) {
        float tmp = extract_half_byte(src_nc[i / 2], static_cast<uint8_t>(i % 2));
        tmp = (tmp - zp) * scale;
        dst[i] = tmp;
    }
}

template <typename TDST, ov::element::Type_t PREC, std::enable_if_t<PREC == ov::element::u8, bool> = true>
void attn_dequant_by_channel_kernel(const void* src,
                                    TDST* dst,
                                    size_t seq_dim,
                                    size_t hidden_dims,
                                    size_t src_stride,
                                    size_t dst_stride,
                                    float* scale,
                                    float* zp) {
    uint8_t* src_nc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));

    for (size_t i = 0; i < seq_dim; ++i) {
        size_t j = 0;
#if defined(HAVE_AVX512F)
        while (j + vec_len_f32_avx512 <= hidden_dims) {
            auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(src_nc + j + i * src_stride));
            auto v0_512 = _mm512_cvtepu8_epi32(v0_128);
            auto v0_value = _mm512_cvtepi32_ps(v0_512);
            v0_value = _mm512_sub_ps(v0_value, _mm512_loadu_ps(zp + j));
            auto v0_out = _mm512_mul_ps(v0_value, _mm512_loadu_ps(scale + j));
            mm512_uni_storeu_ps(dst + i * dst_stride + j, v0_out);
            j += vec_len_f32_avx512;
        }
#endif
#if defined(HAVE_AVX2)
        while (j + vec_len_f32_avx2 <= hidden_dims) {
            auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(src_nc + j + i * src_stride));
            auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
            auto v0_value = _mm256_cvtepi32_ps(v0_256);
            v0_value = _mm256_sub_ps(v0_value, _mm256_loadu_ps(zp + j));
            auto v0_out = _mm256_mul_ps(v0_value, _mm256_loadu_ps(scale + j));
            mm256_uni_storeu_ps(dst + i * dst_stride + j, v0_out);
            j += vec_len_f32_avx2;
        }
#endif
        while (j < hidden_dims) {
            float tmp = src_nc[i * src_stride + j];
            tmp = (tmp - zp[j]) * scale[j];
            dst[i * dst_stride + j] = tmp;
            j += 1;
        }
    }
}

template <typename TDST, ov::element::Type_t PREC, std::enable_if_t<PREC == ov::element::u4, bool> = true>
void attn_dequant_by_channel_kernel(const void* src,
                                    TDST* dst,
                                    size_t seq_dim,
                                    size_t hidden_dims,
                                    size_t src_stride,
                                    size_t dst_stride,
                                    float* scale,
                                    float* zp) {
    uint8_t* src_nc = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(src));
    size_t j = 0;
#if defined(HAVE_AVX512F)
    for (; j + vec_len_f32_avx512 * 2 <= hidden_dims; j += vec_len_f32_avx512 * 2) {
        auto v_scale0 = _mm512_loadu_ps(scale + j);
        auto v_zp0 = _mm512_loadu_ps(zp + j);
        auto v_scale1 = _mm512_loadu_ps(scale + j + vec_len_f32_avx512);
        auto v_zp1 = _mm512_loadu_ps(zp + j + vec_len_f32_avx512);
        __m512 first_half, second_half;
        for (size_t i = 0; i < seq_dim; i++) {
            mm512_loadu_u4_to_f32(src_nc + i * src_stride + j / 2, first_half, second_half);
            first_half = _mm512_sub_ps(first_half, v_zp0);
            first_half = _mm512_mul_ps(first_half, v_scale0);
            second_half = _mm512_sub_ps(second_half, v_zp1);
            second_half = _mm512_mul_ps(second_half, v_scale1);
            mm512_uni_storeu_ps(dst + i * dst_stride + j, first_half);
            mm512_uni_storeu_ps(dst + i * dst_stride + j + vec_len_f32_avx512, second_half);
        }
    }
#elif defined(HAVE_AVX2)
    for (; j + vec_len_f32_avx2 * 2 <= hidden_dims; j += vec_len_f32_avx2 * 2) {
        auto v_scale0 = _mm256_loadu_ps(scale + j);
        auto v_zp0 = _mm256_loadu_ps(zp + j);
        auto v_scale1 = _mm256_loadu_ps(scale + j + vec_len_f32_avx2);
        auto v_zp1 = _mm256_loadu_ps(zp + j + vec_len_f32_avx2);
        __m256 first_half, second_half;
        for (size_t i = 0; i < seq_dim; i++) {
            mm256_loadu_u4_to_f32(src_nc + i * src_stride + j / 2, first_half, second_half);
            first_half = _mm256_sub_ps(first_half, v_zp0);
            first_half = _mm256_mul_ps(first_half, v_scale0);
            second_half = _mm256_sub_ps(second_half, v_zp1);
            second_half = _mm256_mul_ps(second_half, v_scale1);
            mm256_uni_storeu_ps(dst + i * dst_stride + j, first_half);
            mm256_uni_storeu_ps(dst + i * dst_stride + j + vec_len_f32_avx2, second_half);
        }
    }
#endif
    for (; j < hidden_dims; j += 2) {
        for (size_t i = 0; i < seq_dim; ++i) {
            uint8_t data = src_nc[i * src_stride + j / 2];
            float tmp0 = extract_half_byte(data, static_cast<bool>(j % 2));
            float tmp1 = extract_half_byte(data, static_cast<bool>((j + 1) % 2));
            dst[i * dst_stride + j] = (tmp0 - zp[j]) * scale[j];
            dst[i * dst_stride + j + 1] = (tmp1 - zp[j + 1]) * scale[j + 1];
        }
    }
}

#if defined(HAVE_SVE)
void inline attn_dequant_u8_kernel(const uint8_t* src, float* dst, size_t n, float* params) {
    size_t i = 0;
    uint8_t* src_nc = const_cast<uint8_t*>(src);
    float scale = params[0];
    float zp = params[1];
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

}  // namespace ov::Extensions::Cpu::XARCH