// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <float.h>

#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/parallel.hpp"
#include "common.hpp"
#include "attn_quant.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;

template<typename T>
static void quant_u8(const T* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    size_t i = 0;
    float max = -FLT_MAX;
    float min = FLT_MAX;
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
    scale = (max - min) / 255;
    zp = -min / scale;

    i = 0;
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
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto v = mm256_uni_loadu_ps(src + i);
        v = _mm256_fmadd_ps(v, v_scale, v_zp);
        v = _mm256_round_ps(v, _MM_ROUND_NEAREST);
        auto v_i32 = _mm256_cvtps_epi32(v);

        auto high4 = _mm256_extractf128_si256(v_i32, 1);
        auto low4 = _mm256_castsi256_si128(v_i32);
        auto packed = _mm_packs_epi32(low4, high4);
        packed = _mm_packus_epi16(packed, packed);
        _mm_storel_epi64(reinterpret_cast<__m128i*>(dst + i), packed);
    }
#endif
    for (; i < n; i++) {
        float tmp = src[i];
        dst[i] = static_cast<uint8_t>(std::round(tmp / scale + zp));
    }
}

template <typename T, typename T2>
static void attn_quant_mt(const ov::intel_cpu::PlainTensor& k_src,
                          const ov::intel_cpu::PlainTensor& v_src,
                          const ov::intel_cpu::PlainTensor& k_dst,
                          const ov::intel_cpu::PlainTensor& v_dst,
                          const ov::intel_cpu::PlainTensor& k_scale_zp,
                          const ov::intel_cpu::PlainTensor& v_scale_zp) {
    size_t B = k_src.m_dims[0], H = k_src.m_dims[1], L1 = k_src.m_dims[2], S = k_src.m_dims[3];
    parallel_for3d(B, H, L1, [&](size_t b, size_t h, size_t m) {
        auto p_k = k_scale_zp.ptr<float>(b, h, m);
        auto p_v = v_scale_zp.ptr<float>(b, h, m);
        quant_u8(k_src.ptr<T>(b, h, m),
                 k_dst.ptr<T2>(b, h, m),
                 S,
                 p_k[0],
                 p_k[1]);
        quant_u8(v_src.ptr<T>(b, h, m),
                 v_dst.ptr<T2>(b, h, m),
                 S,
                 p_v[0],
                 p_v[1]);
    });
}

void attn_quantkv(const ov::intel_cpu::PlainTensor& k_src,
                  const ov::intel_cpu::PlainTensor& v_src,
                  const ov::intel_cpu::PlainTensor& k_dst,
                  const ov::intel_cpu::PlainTensor& v_dst,
                  const ov::intel_cpu::PlainTensor& k_scale_zp,
                  const ov::intel_cpu::PlainTensor& v_scale_zp) {
    if (k_src.get_precision() == ov::element::f32 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<float, uint8_t>(k_src, v_src, k_dst, v_dst, k_scale_zp, v_scale_zp);
    } else if (k_src.get_precision() == ov::element::bf16 && k_dst.get_precision() == ov::element::u8) {
        attn_quant_mt<ov::bfloat16, uint8_t>(k_src, v_src, k_dst, v_dst, k_scale_zp, v_scale_zp);
    } else {
        OPENVINO_THROW("unsupport src type: ", k_src.get_precision(), ", dst type: ", k_dst.get_precision(), " in attn_quantkv");
    }
}

void attn_quant_u8(const float* src, uint8_t* dst, size_t n, float& scale, float& zp) {
    quant_u8(src, dst, n, scale, zp);
}

void attn_dequant_u8(const uint8_t* src, float* dst, size_t n, float scale, float zp) {
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

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov