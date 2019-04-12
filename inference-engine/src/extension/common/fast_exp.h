// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "defs.h"

#define FAST_EXP_HI   87.3365402f
#define FAST_EXP_LO  -87.3365402f

#define LOG2EF 1.44269504088896341f
#define LOG2  0.693147181f

#define FAST_EXP_C1  12582912.0f
#define FAST_EXP_C2  0.00829171948f

#define FAST_EXP_P0 1.42860677e-06f
#define FAST_EXP_P1 0.0418735221f
#define FAST_EXP_P2 0.166674316f
#define FAST_EXP_P3 0.49999392f
#define FAST_EXP_P4 0.999999881f
#define FAST_EXP_P5 1.0f

#if defined(HAVE_AVX2)
static inline __m256 _avx_fast_exp_ps(__m256 vsrc) {
    __m256 vc_exp_c1 = _mm256_set1_ps(FAST_EXP_C1);
    __m256 vc_exp_c2 = _mm256_set1_ps(FAST_EXP_C2);
    __m256 vc_log2e  = _mm256_set1_ps(LOG2EF);
    __m256 vc_log2   = _mm256_set1_ps(LOG2);

    __m256 vc_exp_p0 = _mm256_set1_ps(FAST_EXP_P0);
    __m256 vc_exp_p1 = _mm256_set1_ps(FAST_EXP_P1);
    __m256 vc_exp_p2 = _mm256_set1_ps(FAST_EXP_P2);
    __m256 vc_exp_p3 = _mm256_set1_ps(FAST_EXP_P3);
    __m256 vc_exp_p4 = _mm256_set1_ps(FAST_EXP_P4);
    __m256 cv_exp_p5 = _mm256_set1_ps(FAST_EXP_P5);

    __m256 vc_exp_hi = _mm256_set1_ps(FAST_EXP_HI);
    __m256 vc_exp_lo = _mm256_set1_ps(FAST_EXP_LO);

    vsrc = _mm256_max_ps(_mm256_min_ps(vsrc, vc_exp_hi), vc_exp_lo);
#if defined(HAVE_FMA)
    __m256 fx = _mm256_fmadd_ps(vsrc, vc_log2e, vc_exp_c1);
#else
    __m256 fx = _mm256_add_ps(_mm256_mul_ps(vsrc, vc_log2e), vc_exp_c1);
#endif
    __m256 fx_ = _mm256_sub_ps(fx, vc_exp_c1);
    __m256i msk = _mm256_slli_epi32(_mm256_castps_si256(fx), 23);

#if defined(HAVE_FMA)
    __m256 q = _mm256_fnmadd_ps(fx_, vc_log2, vsrc);
    __m256 y = _mm256_fnmadd_ps(fx_, vc_exp_p0, q);
           q = _mm256_fmadd_ps(vc_exp_c2, y, vc_exp_p1);
           q = _mm256_fmadd_ps(y, q, vc_exp_p2);
           q = _mm256_fmadd_ps(y, q, vc_exp_p3);
           q = _mm256_fmadd_ps(y, q, vc_exp_p4);
           q = _mm256_fmadd_ps(y, q, cv_exp_p5);
#else
    __m256 q = _mm256_sub_ps(vsrc, _mm256_mul_ps(fx_, vc_log2));
    __m256 y = _mm256_sub_ps(q, _mm256_mul_ps(fx_, vc_exp_p0));
           q = _mm256_add_ps(_mm256_mul_ps(vc_exp_c2, y), vc_exp_p1);
           q = _mm256_add_ps(_mm256_mul_ps(y, q), vc_exp_p2);
           q = _mm256_add_ps(_mm256_mul_ps(y, q), vc_exp_p3);
           q = _mm256_add_ps(_mm256_mul_ps(y, q), vc_exp_p4);
           q = _mm256_add_ps(_mm256_mul_ps(y, q), cv_exp_p5);
#endif

    __m256 vexp = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_castps_si256(q), msk));
    return vexp;
}
#endif

#if defined(HAVE_SSE)
static inline __m128 _sse_fast_exp_ps(__m128 vsrc) {
    __m128 vc_exp_c1 = _mm_set1_ps(FAST_EXP_C1);
    __m128 vc_exp_c2 = _mm_set1_ps(FAST_EXP_C2);
    __m128 vc_log2e  = _mm_set1_ps(LOG2EF);
    __m128 vc_log2   = _mm_set1_ps(LOG2);

    __m128 vc_exp_p0 = _mm_set1_ps(FAST_EXP_P0);
    __m128 vc_exp_p1 = _mm_set1_ps(FAST_EXP_P1);
    __m128 vc_exp_p2 = _mm_set1_ps(FAST_EXP_P2);
    __m128 vc_exp_p3 = _mm_set1_ps(FAST_EXP_P3);
    __m128 vc_exp_p4 = _mm_set1_ps(FAST_EXP_P4);
    __m128 cv_exp_p5 = _mm_set1_ps(FAST_EXP_P5);

    __m128 vc_exp_hi = _mm_set1_ps(FAST_EXP_HI);
    __m128 vc_exp_lo = _mm_set1_ps(FAST_EXP_LO);

    vsrc = _mm_max_ps(_mm_min_ps(vsrc, vc_exp_hi), vc_exp_lo);

    __m128 fx = _mm_fmadd_ps(vsrc, vc_log2e, vc_exp_c1);
    __m128 fx_ = _mm_sub_ps(fx, vc_exp_c1);
    __m128i msk = _mm_slli_epi32(_mm_castps_si128(fx), 23);

    __m128 q = _mm_fnmadd_ps(fx_, vc_log2, vsrc);
    __m128 y = _mm_fnmadd_ps(fx_, vc_exp_p0, q);
           q = _mm_fmadd_ps(vc_exp_c2, y, vc_exp_p1);
           q = _mm_fmadd_ps(y, q, vc_exp_p2);
           q = _mm_fmadd_ps(y, q, vc_exp_p3);
           q = _mm_fmadd_ps(y, q, vc_exp_p4);
           q = _mm_fmadd_ps(y, q, cv_exp_p5);

    __m128 vexp = _mm_castsi128_ps(_mm_add_epi32(_mm_castps_si128(q), msk));

    return vexp;
}
#endif
