// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdint.h>
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#include <immintrin.h>
#endif
#include "common/bf16.hpp"
#include "llm_types.hpp"
#include "utility_avx512.hpp"

namespace llmdnn {
    template<typename S, typename D>
    void memcpy2d_stride_avx512(D* dst, S* src, size_t height, size_t width, size_t src_stride, size_t dst_stride, float* quant=nullptr);

    template<typename D>
    void memcpy2d_stride_avx512(D* dst, float* src, size_t height, size_t width, size_t src_stride, size_t dst_stride, float* quant=nullptr) {
        static_assert(std::is_same<D, ov::bfloat16>::value || std::is_same<D, float>::value ||
                      std::is_same<D, int8_t>::value || std::is_same<D, uint8_t>::value,
                      "memcpy2d_stride_avx512 only support output data types ov::bfloat16/uint8_t/int8_t/float");

        auto tail = width % 16;
        __mmask16 x_mask = _cvtu32_mask16(0xFFFFu >> (16 - tail));

        for (size_t j = 0; j < height; j++) {
            int i;
            if (std::is_same<D, float>::value) {
                for(i = 0; i < width - tail; i += 16) {
                    auto x = _mm512_loadu_ps(src + i);
                    _mm512_storeu_ps(reinterpret_cast<float*>(dst) + i, x);
                }
                // handle tails
                if (tail) {
                    auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                    _mm512_mask_storeu_ps(reinterpret_cast<float*>(dst) + i, x_mask, x);
                }
            }

            if (std::is_same<D, ov::bfloat16>::value) {
                for(i = 0; i < width / 32 * 32; i += 32) {
                    auto x0 = _mm512_loadu_ps(src + i);
                    auto x1 = _mm512_loadu_ps(src + i + 16);
                    auto out = _mm512_cvtne2ps_pbh(x1, x0);
                    _mm512_storeu_epi32(reinterpret_cast<ov::bfloat16*>(dst) + i, (__m512i)out);
                }
                if (i < width - tail) {
                    auto x = _mm512_loadu_ps(src + i);
                    auto out = _mm512_cvtne2ps_pbh(x, x);
                    _mm256_storeu_si256(reinterpret_cast<__m256i*>(reinterpret_cast<ov::bfloat16*>(dst) + i),
                        _mm512_extracti64x4_epi64(out, 0));
                    i += 16;
                }
                // handle tails
                if (tail) {
                    auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                    auto out = _mm512_cvtne2ps_pbh(x, x);
                    _mm256_mask_storeu_epi16(reinterpret_cast<__m256i*>(reinterpret_cast<ov::bfloat16*>(dst) + i),
                        x_mask, _mm512_extracti64x4_epi64(out, 0));
                }
            }

            if (std::is_same<D, int8_t>::value) {
                for(i = 0; i < width - tail; i += 16) {
                    auto x = _mm512_loadu_ps(src + i);
                    auto q = _mm512_loadu_ps(quant + i);
                    x = _mm512_mul_ps(x, q);
                    auto x_i = _mm512_cvtps_epi32(x);
                    _mm512_mask_cvtsepi32_storeu_epi8(reinterpret_cast<int8_t*>(dst) + i, 0xFFFF, x_i);
                }
                // handle tails
                if (tail) {
                    auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                    auto q = _mm512_maskz_loadu_ps(x_mask, quant + i);
                    x = _mm512_mul_ps(x, q);
                    auto x_i = _mm512_cvtps_epi32(x);
                    _mm512_mask_cvtsepi32_storeu_epi8(reinterpret_cast<int8_t*>(dst) + i, x_mask, x_i);
                }
            }

            if (std::is_same<D, uint8_t>::value) {
                auto zero = _mm512_setzero_epi32();
                for(i = 0; i < width - tail; i += 16) {
                    auto x = _mm512_loadu_ps(src + i);
                    auto q = _mm512_loadu_ps(quant + i);
                    x = _mm512_mul_ps(x, q);
                    auto x_i = _mm512_cvtps_epi32(x);
                    x_i = _mm512_max_epi32(x_i, zero);
                    _mm512_mask_cvtusepi32_storeu_epi8(reinterpret_cast<int8_t*>(dst) + i, 0xFFFF, x_i);
                }
                // handle tails
                if (tail) {
                    auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                    auto q = _mm512_maskz_loadu_ps(x_mask, quant + i);
                    x = _mm512_mul_ps(x, q);
                    auto x_i = _mm512_cvtps_epi32(x);
                    x_i = _mm512_max_epi32(x_i, zero);
                    _mm512_mask_cvtusepi32_storeu_epi8(reinterpret_cast<int8_t*>(dst) + i, x_mask, x_i);
                }
            }

            src = reinterpret_cast<float*>(reinterpret_cast<int8_t*>(src) + src_stride);
            dst = reinterpret_cast<D*>(reinterpret_cast<int8_t*>(dst) + dst_stride);
        }
    }
}