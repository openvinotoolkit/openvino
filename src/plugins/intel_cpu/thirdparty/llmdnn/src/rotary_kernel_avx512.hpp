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
#include "utility_kernel_avx512.hpp"

namespace llmdnn {
    template<typename T>
    void rotary_avx512(size_t N, float* cos, float* sin, T* q_src, T* k_src, T* q_dst, T* k_dst) {
        static_assert(std::is_same<T, ov::bfloat16>::value,
                      "rotary_avx512 only support output data types ov::bfloat16/int8_t");
        auto half = N / 2;
        // for (size_t i = 0; i < half; i++) {
        //     q_dst[i] = q_src[i] * cos[i] - q_src[i + half] * sin[i];
        //     k_dst[i] = k_src[i] * cos[i] - k_src[i + half] * sin[i];
        // }
        // for (size_t i = half; i < N; i++) {
        //     q_dst[i] = q_src[i] * cos[i] + q_src[i - half] * sin[i];
        //     k_dst[i] = k_src[i] * cos[i] + k_src[i - half] * sin[i];
        // }
        size_t tail = half % 16;
        __mmask16 x_mask = _cvtu32_mask16(0xFFFFu >> (16 - tail));
        size_t i;
        for (i = 0; i < half - tail; i += 16) {
            auto q = _mm256_loadu_epi16(q_src + i + half);
            auto q_f = _mm512_cvtpbh_ps((__m256bh)q);
            auto k = _mm256_loadu_epi16(k_src + i + half);
            auto k_f = _mm512_cvtpbh_ps((__m256bh)k);
            auto cos_f = _mm512_loadu_ps(cos + i);
            auto sin_f = _mm512_loadu_ps(sin + i);
            auto q_dst_f = _mm512_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm512_mul_ps(k_f, sin_f);

            q = _mm256_loadu_epi16(q_src + i);
            q_f = _mm512_cvtpbh_ps((__m256bh)q);
            k = _mm256_loadu_epi16(k_src + i);
            k_f = _mm512_cvtpbh_ps((__m256bh)k);

            q_dst_f = _mm512_fmsub_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm512_fmsub_ps(k_f, cos_f, k_dst_f);

            auto out = _mm512_cvtne2ps_pbh(q_dst_f, q_dst_f);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q_dst + i), _mm512_extracti64x4_epi64((__m512i)out, 0));
            out = _mm512_cvtne2ps_pbh(k_dst_f, k_dst_f);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(k_dst + i), _mm512_extracti64x4_epi64((__m512i)out, 0));
        }
        if (tail) {
            auto q = _mm256_maskz_loadu_epi16(x_mask, q_src + i + half);
            auto q_f = _mm512_cvtpbh_ps((__m256bh)q);
            auto k = _mm256_maskz_loadu_epi16(x_mask, k_src + i + half);
            auto k_f = _mm512_cvtpbh_ps((__m256bh)k);
            auto cos_f = _mm512_maskz_loadu_ps(x_mask, cos + i);
            auto sin_f = _mm512_maskz_loadu_ps(x_mask, sin + i);
            auto q_dst_f = _mm512_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm512_mul_ps(k_f, sin_f);

            q = _mm256_maskz_loadu_epi16(x_mask, q_src + i);
            q_f = _mm512_cvtpbh_ps((__m256bh)q);
            k = _mm256_maskz_loadu_epi16(x_mask, k_src + i);
            k_f = _mm512_cvtpbh_ps((__m256bh)k);

            q_dst_f = _mm512_fmsub_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm512_fmsub_ps(k_f, cos_f, k_dst_f);

            auto out = _mm512_cvtne2ps_pbh(q_dst_f, q_dst_f);
            _mm256_mask_storeu_epi16(q_dst + i, x_mask, _mm512_extracti64x4_epi64((__m512i)out, 0));
            out = _mm512_cvtne2ps_pbh(k_dst_f, k_dst_f);
            _mm256_mask_storeu_epi16(k_dst + i, x_mask, _mm512_extracti64x4_epi64((__m512i)out, 0));
        }
        // second half
        q_src += half;
        k_src += half;
        cos += half;
        sin += half;
        q_dst += half;
        k_dst += half;
        for (i = 0; i < half - tail; i += 16) {
            auto q = _mm256_loadu_epi16(q_src + i - half);
            auto q_f = _mm512_cvtpbh_ps((__m256bh)q);
            auto k = _mm256_loadu_epi16(k_src + i - half);
            auto k_f = _mm512_cvtpbh_ps((__m256bh)k);
            auto cos_f = _mm512_loadu_ps(cos + i);
            auto sin_f = _mm512_loadu_ps(sin + i);
            auto q_dst_f = _mm512_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm512_mul_ps(k_f, sin_f);

            q = _mm256_loadu_epi16(q_src + i);
            q_f = _mm512_cvtpbh_ps((__m256bh)q);
            k = _mm256_loadu_epi16(k_src + i);
            k_f = _mm512_cvtpbh_ps((__m256bh)k);

            q_dst_f = _mm512_fmadd_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm512_fmadd_ps(k_f, cos_f, k_dst_f);

            auto out = _mm512_cvtne2ps_pbh(q_dst_f, q_dst_f);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(q_dst + i), _mm512_extracti64x4_epi64((__m512i)out, 0));
            out = _mm512_cvtne2ps_pbh(k_dst_f, k_dst_f);
            _mm256_storeu_si256(reinterpret_cast<__m256i *>(k_dst + i), _mm512_extracti64x4_epi64((__m512i)out, 0));
        }
        if (tail) {
            auto q = _mm256_maskz_loadu_epi16(x_mask, q_src + i - half);
            auto q_f = _mm512_cvtpbh_ps((__m256bh)q);
            auto k = _mm256_maskz_loadu_epi16(x_mask, k_src + i - half);
            auto k_f = _mm512_cvtpbh_ps((__m256bh)k);
            auto cos_f = _mm512_maskz_loadu_ps(x_mask, cos + i);
            auto sin_f = _mm512_maskz_loadu_ps(x_mask, sin + i);
            auto q_dst_f = _mm512_mul_ps(q_f, sin_f);
            auto k_dst_f = _mm512_mul_ps(k_f, sin_f);

            q = _mm256_maskz_loadu_epi16(x_mask, q_src + i);
            q_f = _mm512_cvtpbh_ps((__m256bh)q);
            k = _mm256_maskz_loadu_epi16(x_mask, k_src + i);
            k_f = _mm512_cvtpbh_ps((__m256bh)k);

            q_dst_f = _mm512_fmadd_ps(q_f, cos_f, q_dst_f);
            k_dst_f = _mm512_fmadd_ps(k_f, cos_f, k_dst_f);

            auto out = _mm512_cvtne2ps_pbh(q_dst_f, q_dst_f);
            _mm256_mask_storeu_epi16(q_dst + i, x_mask, _mm512_extracti64x4_epi64((__m512i)out, 0));
            out = _mm512_cvtne2ps_pbh(k_dst_f, k_dst_f);
            _mm256_mask_storeu_epi16(k_dst + i, x_mask, _mm512_extracti64x4_epi64((__m512i)out, 0));
        }
    }
}