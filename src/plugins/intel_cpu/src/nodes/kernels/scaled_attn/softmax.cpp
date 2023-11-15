// Copyright (C) 2018-2023 Intel Corporation
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
#include "softmax.hpp"
#include "common.hpp"

namespace InferenceEngine {
namespace Extensions {
namespace Cpu {
namespace XARCH {
#if defined(HAVE_AVX2)
inline __m256i get_mask(int N7) {
    static __m256i mask[] = {
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, 0),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, 0, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, 0, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, 0, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, 0, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(0, -1, -1, -1, -1, -1, -1, -1),
        _mm256_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1),
    };
    return _mm256_loadu_si256(&mask[N7]);
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

inline void exp_ps_avx2(__m256& src) {
    static __m256 exp_ln_flt_min_f = _mm256_castsi256_ps(_mm256_set1_epi32(0xc2aeac50));  // log(FLT_MIN)
    static __m256 exp_ln_flt_max_f = _mm256_castsi256_ps(_mm256_set1_epi32(0x42b17218));  // log(FLT_MAX)
    static __m256 exp_log2ef = _mm256_castsi256_ps(_mm256_set1_epi32(0x3fb8aa3b));        // log2(e)
    static __m256 half = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f000000));              // 0.5f
    static __m256 ln2f = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f317218));              // ln(2)
    static __m256 one = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f800000));               // 1.0f
    static __m256i exponent_bias = _mm256_set1_epi32(0x0000007f);                         // 127
    static constexpr int n_mantissa_bits = 23;
    static __m256 exp_pol1 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3f7ffffb));  // p1 = 0.999999701f
    static __m256 exp_pol2 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3efffee3));  // p2 = 0.499991506f
    static __m256 exp_pol3 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3e2aad40));  // p3 = 0.166676521f
    static __m256 exp_pol4 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3d2b9d0d));  // p4 = 0.0418978221f
    static __m256 exp_pol5 = _mm256_castsi256_ps(_mm256_set1_epi32(0x3c07cfce));  // p5 = 0.00828929059f
    static __m256 two = _mm256_castsi256_ps(_mm256_set1_epi32(0x40000000));       // 2
    // exp(x) =
    // = exp(n * ln(2) + r) // divide x by ln(2) and get quot and rem
    // = 2^n * exp(r)       // simplify the exp(n*ln(2)) expression

    // get mask of values lower than log(FLT_MIN) to zero them in the output
    auto zero_mask = _mm256_cmp_ps(src, exp_ln_flt_min_f, _CMP_LT_OS);

    // clip src
    src = _mm256_min_ps(src, exp_ln_flt_max_f);
    src = _mm256_max_ps(src, exp_ln_flt_min_f);

    // aux1 : r
    auto aux1 = src;

    // calculate exp(x)
    // fx = x * log2(e) + 0.5
    src = _mm256_mul_ps(src, exp_log2ef);
    src = _mm256_add_ps(src, half);

    // tmp = floorf(fx)
    src = _mm256_floor_ps(src);

    // aux1 = x - fx * ln2
    aux1 = _mm256_fnmadd_ps(src, ln2f, aux1);

    // We do not count 2^n here, because n can reach 128 and 2^128 is not
    // representable by fp32, so to get around this problem, instead of computing
    // 2^n * exp(r) will be counted 2*2^(n-1)*exp(r), because 2^127
    // and 2 are numbers representable in fp32.

    // compute 2^(n-1)
    src = _mm256_sub_ps(src, one);
    auto aux2_i = _mm256_cvtps_epi32(src);
    aux2_i = _mm256_add_epi32(aux2_i, exponent_bias);
    aux2_i = _mm256_slli_epi32(aux2_i, n_mantissa_bits);

    // set zeroes at those points which were < log(FLT_MIN)
    auto zero = _mm256_setzero_ps();
    auto aux2 = _mm256_blendv_ps(_mm256_castsi256_ps(aux2_i), zero, zero_mask);

    // compute polynomial
    src = exp_pol5;
    src = _mm256_fmadd_ps(src, aux1, exp_pol4);
    src = _mm256_fmadd_ps(src, aux1, exp_pol3);
    src = _mm256_fmadd_ps(src, aux1, exp_pol2);
    src = _mm256_fmadd_ps(src, aux1, exp_pol1);
    src = _mm256_fmadd_ps(src, aux1, one);

    // y = y * 2^n
    src = _mm256_mul_ps(src, aux2);
    src = _mm256_mul_ps(src, two);
}
#endif

inline void scale_add_reduce_max(float* a, const float scale, const float* b, const size_t size, float& max) {
#if defined(HAVE_AVX512F)
    auto v_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm512_set1_ps(scale);
    auto v_a = v_max;
    auto v_b = v_max;
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_b = _mm512_loadu_ps(b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_max_ps(v_max, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    // process tails
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_b = _mm512_maskz_loadu_ps(mask, b + i);
        v_a = _mm512_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm512_mask_max_ps(v_max, mask, v_a, v_max);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }

    max = _mm512_reduce_max_ps(v_max);
#elif defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm256_set1_ps(scale);
    auto v_a = v_max;
    auto v_b = v_max;
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_b = _mm256_loadu_ps(b + i);
        v_a = _mm256_fmadd_ps(v_a, v_scale, v_b);
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    // process tails
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_b = _mm256_maskload_ps(b + i, mask);
        v_a = _mm256_fmadd_ps(v_a, v_scale, v_b);
        v_a = _mm256_blendv_ps(v_max, v_a, _mm256_castsi256_ps(mask));
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hmax(v_max);
    max = _mm256_cvtss_f32(v_max);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= scale;
        a[i] += b[i];
        max = a[i] > max ? a[i] : max;
    }
#endif
}
template <bool has_alibi, bool has_attn_mask, bool has_causal_mask>
inline void scale_add2_reduce_max(float* a,
                                  float scale,
                                  const float* alibi,
                                  const float* attn_mask,
                                  const uint8_t* causal_mask,
                                  bool select_nfltmax_at_0,  // true:  0 in mask set -FLT_MAX
                                  size_t size,
                                  float& max) {
#if defined(HAVE_AVX512F)
    auto v_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm512_set1_ps(scale);
    auto v_a = v_max;
    size_t i = 0;
    auto v_zeroi32 = _mm512_setzero_epi32();
    auto v_nfltmax = _mm512_set1_ps(-FLT_MAX);
    auto kmask_xor = _cvtu32_mask16(select_nfltmax_at_0 ? 0xFFFF : 0);
    // process vector body
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_mask = _mm512_loadu_ps(alibi + i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_attn_mask) {
            auto v_mask = _mm512_loadu_ps(attn_mask + i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max = _mm512_max_ps(v_max, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    // process tails
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_mask = _mm512_maskz_loadu_ps(mask, alibi + i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_attn_mask) {
            auto v_mask = _mm512_maskz_loadu_ps(mask, attn_mask + i);
            v_a = _mm512_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max = _mm512_mask_max_ps(v_max, mask, v_a, v_max);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }

    max = _mm512_reduce_max_ps(v_max);
#elif defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm256_set1_ps(scale);
    auto v_a = v_max;
    auto v_zeroi32 = _mm256_setzero_si256();
    auto v_mask_xor = _mm256_set1_epi32(select_nfltmax_at_0 ? -1 : 0);
    auto v_nfltmax = _mm256_set1_ps(-FLT_MAX);
    size_t i = 0;
    // process vector body
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_mask = _mm256_loadu_ps(alibi + i);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_attn_mask) {
            auto v_mask = _mm256_loadu_ps(attn_mask + i);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    // process tails
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_mask = _mm256_maskload_ps(alibi + i, mask);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_attn_mask) {
            auto v_mask = _mm256_maskload_ps(attn_mask + i, mask);
            v_a = _mm256_add_ps(v_a, v_mask);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_a = _mm256_blendv_ps(v_max, v_a, _mm256_castsi256_ps(mask));
        v_max = _mm256_max_ps(v_max, v_a);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hmax(v_max);
    max = _mm256_cvtss_f32(v_max);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] *= scale;
        if (has_alibi)
            a[i] += alibi[i];

        if (has_attn_mask)
            a[i] += attn_mask[i];

        if (has_causal_mask) {
            if (select_nfltmax_at_0) {
                if (causal_mask[i] == 0)
                    a[i] = -FLT_MAX;
            } else {
                if (causal_mask[i] != 0)
                    a[i] = -FLT_MAX;
            }
        }

        max = a[i] > max ? a[i] : max;
    }
#endif
}

#if defined(HAVE_AVX512F)
inline void exp_ps_avx512(__m512& src) {
    static __m512 exp_ln_flt_min_f = _mm512_castsi512_ps(_mm512_set1_epi32(0xc2aeac50));  // log(FLT_MIN)
    static __m512 exp_ln_flt_max_f = _mm512_castsi512_ps(_mm512_set1_epi32(0x42b17218));  // log(FLT_MAX)
    static __m512 exp_log2ef = _mm512_castsi512_ps(_mm512_set1_epi32(0x3fb8aa3b));        // log2(e)
    static __m512 half = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f000000));              // 0.5f
    static __m512 ln2f = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f317218));              // ln(2)
    static __m512 one = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f800000));               // 1.0f
    static __m512i exponent_bias = _mm512_set1_epi32(0x0000007f);                         // 127
    static constexpr int n_mantissa_bits = 23;
    static __m512 exp_pol1 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3f7ffffb));  // p1 = 0.999999701f
    static __m512 exp_pol2 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3efffee3));  // p2 = 0.499991506f
    static __m512 exp_pol3 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3e2aad40));  // p3 = 0.166676521f
    static __m512 exp_pol4 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3d2b9d0d));  // p4 = 0.0418978221f
    static __m512 exp_pol5 = _mm512_castsi512_ps(_mm512_set1_epi32(0x3c07cfce));  // p5 = 0.00828929059f
    static __m512 two = _mm512_castsi512_ps(_mm512_set1_epi32(0x40000000));       // 2
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
    aux2_i = _mm512_slli_epi32(aux2_i, n_mantissa_bits);

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
#endif

inline void exp_reduce_sum(float* a, const float max, const size_t size, float& sum) {
#if defined(HAVE_AVX512F)
    size_t i = 0;
    __m512 v_a;
    auto v_max = _mm512_set1_ps(max);
    auto v_sum = _mm512_set1_ps(0.0f);
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_sub_ps(v_a, v_max);
        exp_ps_avx512(v_a);
        v_sum = _mm512_add_ps(v_sum, v_a);
        _mm512_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx512;
    }

    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_sub_ps(v_a, v_max);
        exp_ps_avx512(v_a);
        v_sum = _mm512_mask_add_ps(v_sum, mask, v_a, v_sum);
        _mm512_mask_storeu_ps(a + i, mask, v_a);
    }
    sum = _mm512_reduce_add_ps(v_sum);
#elif defined(HAVE_AVX2)
    size_t i = 0;
    __m256 v_a;
    auto v_max = _mm256_set1_ps(max);
    auto v_sum = _mm256_set1_ps(0.0f);
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_sub_ps(v_a, v_max);
        exp_ps_avx2(v_a);
        v_sum = _mm256_add_ps(v_sum, v_a);
        _mm256_storeu_ps(a + i, v_a);
        i += vec_len_f32_avx2;
    }

    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_sub_ps(v_a, v_max);
        exp_ps_avx2(v_a);
        v_a = _mm256_blendv_ps(_mm256_setzero_ps(), v_a, _mm256_castsi256_ps(mask));
        v_sum = _mm256_add_ps(v_a, v_sum);
        _mm256_maskstore_ps(a + i, mask, v_a);
    }
    hsum(v_sum);
    sum = _mm256_cvtss_f32(v_sum);
#else
    for (size_t i = 0; i < size; i++) {
        a[i] = exp(a[i] - max);
        sum += a[i];
    }
#endif
}

inline void multiply_scalar(float* a, float* a_dst, const float val, const size_t size) {
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    size_t i = 0;
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        _mm512_mask_storeu_ps(a_dst + i, mask, v_a);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(val);
    __m256 v_a = {0};
    size_t i = 0;
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);
        _mm256_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx2;
    }
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);
        _mm256_maskstore_ps(a_dst + i, mask, v_a);
    }
#else
    for (size_t i = 0; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

inline void multiply_scalar(float* a, ov::bfloat16* a_dst, const float val, const size_t size) {
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
    size_t i = 0;
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
        mm512_uni_mask_storeu_ps(a_dst + i, mask, v_a);
    }
#else
    for (size_t i = 0; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

void attn_softmax(float* a,
                  void* a_dst,
                  float scale,
                  float* alibi,
                  float* attn_mask,
                  uint8_t* causal_mask,
                  bool select_nfltmax_at_0,
                  size_t len,
                  size_t total_size,
                  Precision dst_precision) {
    using func_type = void (*)(float*, float, const float*, const float*, const uint8_t*, bool, size_t, float&);
    static func_type funcs[] = {
        scale_add2_reduce_max<false, false, false>,
        scale_add2_reduce_max<false, false, true>,
        scale_add2_reduce_max<false, true, false>,
        scale_add2_reduce_max<false, true, true>,
        scale_add2_reduce_max<true, false, false>,
        scale_add2_reduce_max<true, false, true>,
        scale_add2_reduce_max<true, true, false>,
        scale_add2_reduce_max<true, true, true>
    };
    int dispatch = (alibi ? 0b100 : 0) | (attn_mask ? 0b010 : 0) | (causal_mask ? 0b001 : 0);
    float max = std::numeric_limits<float>::lowest();
    funcs[dispatch](a, scale, alibi, attn_mask, causal_mask, select_nfltmax_at_0, len, max);

    float sum = 0.0f;
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    // divide sum
    float scalar = 1.0f / sum;
    if (dst_precision == Precision::FP32) {
        multiply_scalar(a, static_cast<float*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<float*>(a_dst) + len, 0, sizeof(float) * (total_size - len));
    } else {
        multiply_scalar(a, static_cast<ov::bfloat16*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<ov::bfloat16*>(a_dst) + len, 0, sizeof(ov::bfloat16) * (total_size - len));
    }
}
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace InferenceEngine