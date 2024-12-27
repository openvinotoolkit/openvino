// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "nodes/kernels/scaled_attn/common.hpp"

#if defined(HAVE_SSE) || defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include <cstddef>
#include <cstdint>

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

template <typename TDST>
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

template <typename TDST>
void attn_dequant_u8_by_channel_kernel(const uint8_t* src,
                                       TDST* dst,
                                       size_t seq_dim,
                                       size_t hidden_dims,
                                       size_t src_stride,
                                       size_t dst_stride,
                                       float* scale,
                                       float* zp) {
    uint8_t* src_nc = const_cast<uint8_t*>(src);

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

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
