// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "defs.h"

#define EXP_HI  88.3762626647949f
#define EXP_LO -88.3762626647949f

#define LOG2EF 1.44269504088896341f

#define EXP_C1  0.693359375f
#define EXP_C2 -2.12194440e-4f

#define EXP_P0 1.9875691500e-4f
#define EXP_P1 1.3981999507e-3f
#define EXP_P2 8.3334519073e-3f
#define EXP_P3 4.1665795894e-2f
#define EXP_P4 1.6666665459e-1f
#define EXP_P5 5.0000001201e-1f

#if defined(HAVE_AVX2)
static inline __m256 _avx_opt_exp_ps(__m256 vsrc) {
    const __m256 vc_one    = _mm256_set1_ps(1.0f);
    const __m256 vc_half   = _mm256_set1_ps(0.5f);

    const __m256 vc_exp_hi = _mm256_set1_ps(EXP_HI);
    const __m256 vc_exp_lo = _mm256_set1_ps(EXP_LO);

    const __m256 vc_log2e  = _mm256_set1_ps(LOG2EF);

    const __m256 vc_exp_c1 = _mm256_set1_ps(EXP_C1);
    const __m256 vc_exp_c2 = _mm256_set1_ps(EXP_C2);

    const __m256 vc_exp_p0 = _mm256_set1_ps(EXP_P0);
    const __m256 vc_exp_p1 = _mm256_set1_ps(EXP_P1);
    const __m256 vc_exp_p2 = _mm256_set1_ps(EXP_P2);
    const __m256 vc_exp_p3 = _mm256_set1_ps(EXP_P3);
    const __m256 vc_exp_p4 = _mm256_set1_ps(EXP_P4);
    const __m256 vc_exp_p5 = _mm256_set1_ps(EXP_P5);

    // 1: shrink to a meaningful range
    __m256 vsrc0 = _mm256_max_ps(_mm256_min_ps(vsrc, vc_exp_hi), vc_exp_lo);

    // 2. express exp(i) as exp(g + n*log(2))
    // Generally speaking, it is to split exp and significand
#if defined(HAVE_FMA)
    __m256 fx = _mm256_fmadd_ps(vsrc0, vc_log2e, vc_half);
#else
    __m256 fx = _mm256_add_ps(_mm256_mul_ps(vsrc0, vc_log2e), vc_half);
#endif

    // 3. get the significand
    __m256 fx_ = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(fx));
    __m256 mask = _mm256_cmp_ps(fx_, fx, _CMP_GT_OS);
    mask = _mm256_and_ps(mask, vc_one);
    fx = _mm256_sub_ps(fx_, mask);

    __m256 q = _mm256_mul_ps(fx, vc_exp_c1);
    __m256 z = _mm256_mul_ps(fx, vc_exp_c2);
    __m256 x_ = _mm256_sub_ps(_mm256_sub_ps(vsrc0, q), z);

    // 4. rational approximation for exponential of the fractional part:
    z = _mm256_mul_ps(x_, x_);

#if defined(HAVE_FMA)
    __m256 y = _mm256_fmadd_ps(vc_exp_p0, x_, vc_exp_p1);
    y = _mm256_fmadd_ps(vc_exp_p0, x_, vc_exp_p1);
    y = _mm256_fmadd_ps(y, x_, vc_exp_p2);
    y = _mm256_fmadd_ps(y, x_, vc_exp_p3);
    y = _mm256_fmadd_ps(y, x_, vc_exp_p4);
    y = _mm256_fmadd_ps(y, x_, vc_exp_p5);
    y = _mm256_fmadd_ps(y, z, x_);
    y = _mm256_add_ps(y, vc_one);
#else
    __m256 y = _mm256_mul_ps(vc_exp_p0, x_);
    z = _mm256_mul_ps(x_, x_);
    y = _mm256_add_ps(y, vc_exp_p1);
    y = _mm256_mul_ps(y, x_);
    y = _mm256_add_ps(y, vc_exp_p2);
    y = _mm256_mul_ps(y, x_);
    y = _mm256_add_ps(y, vc_exp_p3);
    y = _mm256_mul_ps(y, x_);
    y = _mm256_add_ps(y, vc_exp_p4);
    y = _mm256_mul_ps(y, x_);
    y = _mm256_add_ps(y, vc_exp_p5);

    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x_);
    y = _mm256_add_ps(y, vc_one);
#endif

    // 5. multiply by power of 2
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(0x7f)), 23);

    __m256 vdst = _mm256_mul_ps(y, _mm256_castsi256_ps(pow2n));
    return vdst;
}
#endif

#if defined(HAVE_SSE)
static inline __m128 _sse_opt_exp_ps(__m128 vsrc) {
    const __m128 vc_one    = _mm_set1_ps(1.0f);
    const __m128 vc_half   = _mm_set1_ps(0.5f);

    const __m128 vc_exp_hi = _mm_set1_ps(EXP_HI);
    const __m128 vc_exp_lo = _mm_set1_ps(EXP_LO);

    const __m128 vc_log2e  = _mm_set1_ps(LOG2EF);

    const __m128 vc_exp_c1 = _mm_set1_ps(EXP_C1);
    const __m128 vc_exp_c2 = _mm_set1_ps(EXP_C2);

    const __m128 vc_exp_p0 = _mm_set1_ps(EXP_P0);
    const __m128 vc_exp_p1 = _mm_set1_ps(EXP_P1);
    const __m128 vc_exp_p2 = _mm_set1_ps(EXP_P2);
    const __m128 vc_exp_p3 = _mm_set1_ps(EXP_P3);
    const __m128 vc_exp_p4 = _mm_set1_ps(EXP_P4);
    const __m128 vc_exp_p5 = _mm_set1_ps(EXP_P5);

    // 1: shrink to a meaningful range
    __m128 vsrc0 = _mm_max_ps(_mm_min_ps(vsrc, vc_exp_hi), vc_exp_lo);

    // 2. express exp(i) as exp(g + n*log(2))
    // Generally speaking, it is to split exp and significand
    __m128 fx = _mm_add_ps(_mm_mul_ps(vsrc0, vc_log2e), vc_half);

    // 3. get the significand
    __m128 fx_ = _mm_cvtepi32_ps(_mm_cvtps_epi32(fx));
    __m128 mask = _mm_cmpgt_ps(fx_, fx);
    mask = _mm_and_ps(mask, vc_one);
    fx = _mm_sub_ps(fx_, mask);

    __m128 q = _mm_mul_ps(fx, vc_exp_c1);
    __m128 z = _mm_mul_ps(fx, vc_exp_c2);
    __m128 x_ = _mm_sub_ps(vsrc0, q);
    x_ = _mm_sub_ps(x_, z);

    // 4. rational approximation for exponential of the fractional part:
    __m128 y = _mm_mul_ps(vc_exp_p0, x_);
    z = _mm_mul_ps(x_, x_);
    y = _mm_add_ps(y, vc_exp_p1);
    y = _mm_mul_ps(y, x_);
    y = _mm_add_ps(y, vc_exp_p2);
    y = _mm_mul_ps(y, x_);
    y = _mm_add_ps(y, vc_exp_p3);
    y = _mm_mul_ps(y, x_);
    y = _mm_add_ps(y, vc_exp_p4);
    y = _mm_mul_ps(y, x_);
    y = _mm_add_ps(y, vc_exp_p5);

    y = _mm_mul_ps(y, z);
    y = _mm_add_ps(y, x_);
    y = _mm_add_ps(y, vc_one);

    // 5. multiply by power of 2
    __m128i pow2n = _mm_slli_epi32(_mm_add_epi32(_mm_cvtps_epi32(fx), _mm_set1_epi32(0x7f)), 23);

    __m128 vdst = _mm_mul_ps(y, _mm_castsi128_ps(pow2n));
    return vdst;
}
#endif