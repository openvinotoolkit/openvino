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
    inline void exp_ps_avx512(__m512 & src) {
        static __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));    // log(FLT_MIN)
        static __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));    // log(FLT_MAX)
        static __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));          // log2(e)
        static __m512 half = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));                // 0.5f
        static __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));                // ln(2)
        static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));                 // 1.0f
        static __m512i exponent_bias = _mm512_set1_epi32(0x0000007f);                           // 127
        static constexpr int n_mantissa_bits = 23;
        static __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7ffffb));            // p1 = 0.999999701f
        static __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3efffee3));            // p2 = 0.499991506f
        static __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3e2aad40));            // p3 = 0.166676521f
        static __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3d2b9d0d));            // p4 = 0.0418978221f
        static __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3c07cfce));            // p5 = 0.00828929059f
        static __m512 two = _mm512_castsi512_ps(_mm512_set1_epi32(0x40000000));                 // 2
        // exp(x) =
        // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
        // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

        // get mask of values lower than log(FLT_MIN) to zero them in the output
        auto zero_mask = _mm512_cmp_ps_mask(src, exp_ln_flt_min_f, _CMP_LT_OS);

        // clip src
        src = _mm512_min_ps(src, exp_ln_flt_max_f);
        src = _mm512_max_ps(src, exp_ln_flt_min_f);

        // aux1 : r
        auto aux1 = src;

        // calculate exp(x)
        // fx = x * log2(e) + 0.5
        src = _mm512_mul_ps(src, exp_log2ef);
        src = _mm512_add_ps(src, half);

        // tmp = floorf(fx)
        src = _mm512_floor_ps(src);

        // aux1 = x - fx * ln2
        aux1 = _mm512_fnmadd_ps(src, ln2f, aux1);
        
        // We do not count 2^n here, because n can reach 128 and 2^128 is not
        // representable by fp32, so to get around this problem, instead of computing
        // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
        // and 2 are numbers representable in fp32.

        // compute 2^(n-1)
        src = _mm512_sub_ps(src, one);
        auto aux2_i = _mm512_cvtps_epi32(src);
        aux2_i = _mm512_add_epi32(aux2_i, exponent_bias);
        aux2_i = _mm512_slli_epi32 (aux2_i, n_mantissa_bits);

        // set zeroes at those points which were < log(FLT_MIN)
        auto zero = _mm512_setzero_ps();
        auto aux2 = _mm512_mask_blend_ps(zero_mask, _mm512_castsi512_ps(aux2_i), zero);

        // compute polynomial
        src = exp_pol5;
        src = _mm512_fmadd_ps(src, aux1, exp_pol4);
        src = _mm512_fmadd_ps(src, aux1, exp_pol3);
        src = _mm512_fmadd_ps(src, aux1, exp_pol2);
        src = _mm512_fmadd_ps(src, aux1, exp_pol1);
        src = _mm512_fmadd_ps(src, aux1, one);

        // y = y * 2^n
        src = _mm512_mul_ps(src, aux2);
        src = _mm512_mul_ps(src, two);
    }

    template<typename D>
    void softmax_avx512(D* dst, float* src, int N, float* s_max=nullptr, float* s_sum=nullptr, float* quant=nullptr) {
        static_assert(std::is_same<D, ov::bfloat16>::value || std::is_same<D, float>::value ||
                      std::is_same<D, int8_t>::value || std::is_same<D, uint8_t>::value,
                      "softmax_avx512 only support output data types ov::bfloat16/uint8_t/int8_t/float");

        static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));                 // 1.0f
        auto tail = N % 16;
        __mmask16 x_mask = _cvtu32_mask16(0xFFFFu >> (16 - tail));

        // get max
        auto x_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
        int i;
        for (i = 0; i < N - tail; i += 16) {
            auto x = _mm512_loadu_ps(src + i);
            x_max = _mm512_max_ps(x_max, x);
        }
        // tails
        if (tail) {
            auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
            x_max = _mm512_mask_max_ps(x_max, x_mask, x_max, x);
        }
        auto max = _mm512_reduce_max_ps(x_max);
        if (s_max) *s_max = max;
        x_max = _mm512_set1_ps(max);

        // softmax
        auto sum_exp = _mm512_setzero_ps();
        for(i = 0; i < N - tail; i += 16) {
            auto x = _mm512_loadu_ps(src + i);
            x = _mm512_sub_ps(x, x_max);
            exp_ps_avx512(x);                             // exp(x-x_max)
            sum_exp = _mm512_add_ps(sum_exp, x);   // sum(exp(x-x_max))
            _mm512_storeu_ps(src + i, x);          // save exp(x-x_max)
        }

        // handle tails
        if (tail) {
            auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
            x = _mm512_sub_ps(x, x_max);
            exp_ps_avx512(x);
            x = _mm512_mask_blend_ps(x_mask, _mm512_setzero_ps(), x);
            sum_exp = _mm512_add_ps(sum_exp, x);
            _mm512_mask_storeu_ps(src + i, x_mask, x);
        }

        auto sum = _mm512_reduce_add_ps(sum_exp);
        if (s_sum) *s_sum = sum;
        sum_exp = _mm512_set1_ps(sum);
        auto reciprocal_sum_exp = _mm512_div_ps(one, sum_exp);     // 1/sum_exp

        // divide
        if (std::is_same<D, float>::value) {
            for(i = 0; i < N - tail; i += 16) {
                auto x = _mm512_loadu_ps(src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                _mm512_storeu_ps(dst + i, x);
            }
            // handle tails
            if (tail) {
                auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                _mm512_mask_storeu_ps(dst + i, x_mask, x);
            }
        }
        if (std::is_same<D, ov::bfloat16>::value) {
            for(i = 0; i < N / 32 * 32; i += 32) {
                auto x0 = _mm512_loadu_ps(src + i);
                auto x1 = _mm512_loadu_ps(src + i + 16);
                x0 = _mm512_mul_ps(x0, reciprocal_sum_exp);
                x1 = _mm512_mul_ps(x1, reciprocal_sum_exp);
                auto out = _mm512_cvtne2ps_pbh(x1, x0);
                _mm512_storeu_epi32(dst + i, (__m512i)out);
            }
            if (i < N - tail) {
                auto x = _mm512_loadu_ps(src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                auto out = _mm512_cvtne2ps_pbh(x, x);
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(dst + i), _mm512_extracti64x4_epi64(out, 0));
                i += 16;
            }
            // handle tails
            if (tail) {
                auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                auto out = _mm512_cvtne2ps_pbh(x, x);
                _mm256_mask_storeu_epi16(dst + i, x_mask, _mm512_extracti64x4_epi64(out, 0));
            }
        }
        if (std::is_same<D, int8_t>::value) {
            for(i = 0; i < N - tail; i += 16) {
                auto q = _mm512_loadu_ps(quant + i);
                auto x = _mm512_loadu_ps(src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                x = _mm512_mul_ps(x, q);
                auto x_i = _mm512_cvtps_epi32(x);
                _mm512_mask_cvtsepi32_storeu_epi8(dst + i, 0xFFFF, x_i);
            }
            // handle tails
            if (tail) {
                auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                auto q = _mm512_maskz_loadu_ps(x_mask, quant + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                x = _mm512_mul_ps(x, q);
                auto x_i = _mm512_cvtps_epi32(x);
                _mm512_mask_cvtsepi32_storeu_epi8(dst + i, x_mask, x_i);
            }
        }
        if (std::is_same<D, uint8_t>::value) {
            auto zero = _mm512_setzero_epi32();
            for(i = 0; i < N - tail; i += 16) {
                auto q = _mm512_loadu_ps(quant + i);
                auto x = _mm512_loadu_ps(src + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                x = _mm512_mul_ps(x, q);
                auto x_i = _mm512_cvtps_epi32(x);
                x_i = _mm512_max_epi32(x_i, zero);
                _mm512_mask_cvtusepi32_storeu_epi8(dst + i, 0xFFFF, x_i);
            }
            // handle tails
            if (tail) {
                auto x = _mm512_maskz_loadu_ps(x_mask, src + i);
                auto q = _mm512_maskz_loadu_ps(x_mask, quant + i);
                x = _mm512_mul_ps(x, reciprocal_sum_exp);
                x = _mm512_mul_ps(x, q);
                auto x_i = _mm512_cvtps_epi32(x);
                x_i = _mm512_max_epi32(x_i, zero);
                _mm512_mask_cvtusepi32_storeu_epi8(dst + i, x_mask, x_i);
            }
        }
    }
}