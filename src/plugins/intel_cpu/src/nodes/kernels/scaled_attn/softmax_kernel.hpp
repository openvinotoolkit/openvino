// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common.hpp"
#include "openvino/core/type/element_type.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#if defined(OPENVINO_ARCH_ARM64)
#include "arm_neon.h"
#endif

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

#if defined(HAVE_AVX2)
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

template <bool has_alibi, bool has_attn_mask, bool has_causal_mask, typename T>
inline void scale_add2_reduce_max(float* a,
                                  float scale,
                                  const float* alibi_lookup,
                                  const T* attn_mask,
                                  const uint8_t* causal_mask,
                                  bool select_nfltmax_at_0,  // true:  0 in mask set -FLT_MAX
                                  size_t size,
                                  float alibi_slope,
                                  float& max) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_max = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm512_set1_ps(scale);
    auto v_a = v_max;
    auto v_zeroi32 = _mm512_setzero_epi32();
    auto v_nfltmax = _mm512_set1_ps(-FLT_MAX);
    auto kmask_xor = _cvtu32_mask16(select_nfltmax_at_0 ? 0xFFFF : 0);
    auto v_alibi_slope = _mm512_set1_ps(alibi_slope);
    // process vector body
    while (i + vec_len_f32_avx512 <= size) {
        v_a = _mm512_loadu_ps(a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm512_loadu_ps(alibi_lookup + i);
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm512_uni_loadu_ps(attn_mask + i);
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
            auto v_lookup = _mm512_maskz_loadu_ps(mask, alibi_lookup + i);
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm512_uni_loadu_tail_ps(attn_mask + i, size - i);
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

        i += (size - i);
    }

    max = _mm512_reduce_max_ps(v_max);
#elif defined(HAVE_AVX2)
    auto v_max = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_scale = _mm256_set1_ps(scale);
    auto v_a = v_max;
    auto v_zeroi32 = _mm256_setzero_si256();
    auto v_mask_xor = _mm256_set1_epi32(select_nfltmax_at_0 ? -1 : 0);
    auto v_nfltmax = _mm256_set1_ps(-FLT_MAX);
    auto v_alibi_slope = _mm256_set1_ps(alibi_slope);
    // process vector body
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);

        if (has_alibi) {
            auto v_lookup = _mm256_loadu_ps(alibi_lookup + i);
            v_a = _mm256_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm256_uni_loadu_ps(attn_mask + i);
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
            auto v_lookup = _mm256_maskload_ps(alibi_lookup + i, mask);
            v_a = _mm256_fmadd_ps(v_lookup, v_alibi_slope, v_a);
        }

        if (has_attn_mask) {
            auto v_mask = mm256_uni_loadu_tail_ps(attn_mask + i, size - i);
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

        i += (size - i);
    }
    hmax(v_max);
    max = _mm256_cvtss_f32(v_max);
#elif defined(OPENVINO_ARCH_ARM64)
    auto v_max = vdupq_n_f32(std::numeric_limits<float>::lowest());
    auto v_scale = vdupq_n_f32(scale);
    auto v_nfltmax = vdupq_n_f32(-FLT_MAX);
    auto v_alibi_slope = vdupq_n_f32(alibi_slope);
    uint32x4_t v_zeroi32 = vdupq_n_u32(0);

    // process vector body
    while (i + vec_len_f32_neon <= size) {
        float32x4_t v_a = vld1q_f32(a + i);
        v_a = vmulq_f32(v_a, v_scale);

        if (has_alibi) {
            float32x4_t v_lookup = vld1q_f32(alibi_lookup + i);
            v_a = vmlaq_f32(v_a, v_lookup, v_alibi_slope);
        }

        if (has_attn_mask) {
            float32x4_t v_mask = __vld1q_f32(attn_mask + i);
            v_a = vaddq_f32(v_a, v_mask);
        }

        if (has_causal_mask) {
            uint8x16_t v_maski8 = vld1q_u8(causal_mask + i);
            uint16x8_t v_maski16 = vmovl_u8(vget_low_u8(v_maski8));
            uint32x4_t v_maski32_low = vmovl_u16(vget_low_u16(v_maski16));
            uint32x4_t v_maski32_high = vmovl_u16(vget_high_u16(v_maski16));
            uint32x4_t v_maski32[2] = {v_maski32_low, v_maski32_high};
            for (int j = 0; j < 2; ++j) {
                uint32x4_t kmask = vceqq_u32(v_maski32[j], v_zeroi32);  // ==0
                v_a = vbslq_f32(kmask, v_nfltmax, v_a);  // mask => -FLT_MAX
            }
        }

        v_max = vmaxq_f32(v_max, v_a);
        vst1q_f32(a + i, v_a);
        i += vec_len_f32_neon;
    }
    max = vmaxvq_f32(v_max);

#endif
    for (; i < size; i++) {
        a[i] *= scale;
        if (has_alibi) {
            a[i] += alibi_lookup[i] * alibi_slope;
        }

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
}

#if defined(OPENVINO_ARCH_ARM64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <bool has_alibi, bool has_attn_mask, bool has_causal_mask, typename T>
inline void scale_add2_reduce_max(ov::float16* a,
                                  float scale,
                                  const ov::float16* alibi_lookup,
                                  const T* attn_mask,
                                  const uint8_t* causal_mask,
                                  bool select_nfltmax_at_0,  // true:  0 in mask set -FLT_MAX
                                  size_t size,
                                  float alibi_slope,
                                  ov::float16& max) {
    float16x8_t v_max = vdupq_n_f16(static_cast<float16_t>(-FLT_MAX));
    float16x8_t v_scale = vdupq_n_f16(static_cast<float16_t>(scale));
    float16x8_t v_a;
    size_t i = 0;
    uint16x8_t v_zeroi16 = vdupq_n_u16(0);
    float16x8_t v_nfltmax = vdupq_n_f16(static_cast<float16_t>(-FLT_MAX));
    uint16x8_t mask_xor = vdupq_n_u16(select_nfltmax_at_0 ? 0xFFFF : 0);
    float16x8_t v_alibi_slope = vdupq_n_f16(static_cast<float16_t>(alibi_slope));

    // process vector body
    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));
        v_a = vmulq_f16(v_a, v_scale);

        if (has_alibi) {
            float16x8_t v_lookup = vld1q_f16(reinterpret_cast<const float16_t*>(alibi_lookup + i));
            v_a = vfmaq_f16(v_a, v_lookup, v_alibi_slope);
        }

        if (has_attn_mask) {
            float16x8_t v_mask = vld1q_f16(reinterpret_cast<const float16_t*>(attn_mask + i));
            v_a = vaddq_f16(v_a, v_mask);
        }

        if (has_causal_mask) {
            uint8x8_t v_maski8 = vld1_u8(causal_mask + i);
            uint16x8_t v_maski16 = vmovl_u8(v_maski8);
            uint16x8_t kmask = vceqq_u16(v_maski16, v_zeroi16);
            kmask = veorq_u16(kmask, mask_xor);
            v_a = vbslq_f16(kmask, v_nfltmax, v_a);
        }

        v_max = vmaxq_f16(v_max, v_a);
        vst1q_f16(reinterpret_cast<float16_t*>(a + i), v_a);
    }
    // process tails
    for (; i < size; i++) {
        a[i] *= scale;
        if (has_alibi) {
            a[i] += alibi_lookup[i] * alibi_slope;
        }

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
    max = vmaxvq_f16(v_max);
}
#endif

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
    size_t i = 0;
#if defined(HAVE_AVX512F)
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

        i += (size - i);
    }
    sum = _mm512_reduce_add_ps(v_sum);
#elif defined(HAVE_AVX2)
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

        i += (size - i);
    }
    hsum(v_sum);
    sum = _mm256_cvtss_f32(v_sum);
#elif defined(OPENVINO_ARCH_ARM64)
    float32x4_t v_a;
    float32x4_t v_max = vdupq_n_f32(max);
    float32x4_t v_sum = vdupq_n_f32(0.0f);

    while (i + vec_len_f32_neon <= size) {
        v_a = vld1q_f32(a + i);
        v_a = vsubq_f32(v_a, v_max);
        v_a = exp_ps_neon_f32(v_a);
        v_sum = vaddq_f32(v_sum, v_a);
        vst1q_f32(a + i, v_a);
        i += vec_len_f32_neon;
    }
    sum = vaddvq_f32(v_sum);

#endif
    for (; i < size; i++) {
        a[i] = exp(a[i] - max);
        sum += a[i];
    }
}

#if defined(OPENVINO_ARCH_ARM64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void exp_reduce_sum(ov::float16* a, const ov::float16 max, const size_t size, ov::float16& sum) {
    const size_t vec_len_f16_neon = 8;
    float16x8_t v_a;
    float16x8_t v_max = vdupq_n_f16(max);
    float16x8_t v_sum = vdupq_n_f16(0.0f);
    size_t i = 0;

    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));
        v_a = vsubq_f16(v_a, v_max);
        v_a = exp_ps_neon_f16(v_a);
        v_sum = vaddq_f16(v_sum, v_a);
        vst1q_f16(reinterpret_cast<float16_t*>(a + i), v_a);
    }

    float16x4_t v_sum_low = vadd_f16(vget_low_f16(v_sum), vget_high_f16(v_sum));
    float16x4_t v_sum_pair = vpadd_f16(v_sum_low, v_sum_low);
    float16x4_t v_sum_final = vpadd_f16(v_sum_pair, v_sum_pair);

    sum += vget_lane_f16(v_sum_final, 0);

    for (; i < size; ++i) {
        a[i] = static_cast<ov::float16>(exp(static_cast<float>(a[i] - max)));
        sum += a[i];
    }
}
#endif

inline void multiply_scalar(float* a, float* a_dst, const float val, const size_t size) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
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

        i += (size - i);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(val);
    __m256 v_a = {0};
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

        i += (size - i);
    }
#elif defined(OPENVINO_ARCH_ARM64)
    float32x4_t v_scale = vdupq_n_f32(val);
    while (i + vec_len_f32_neon <= size) {
        float32x4_t v_a = vld1q_f32(a + i);
        v_a = vmulq_f32(v_a, v_scale);
        vst1q_f32(a_dst + i, v_a);
        i += vec_len_f32_neon;
    }
#endif
    for (; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
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

#if defined(OPENVINO_ARCH_ARM64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void multiply_scalar(ov::float16* a, ov::float16* a_dst, const ov::float16 val, const size_t size) {
    const size_t vec_len_f16_neon = 8;
    float16x8_t v_a, v_res;
    float16x8_t v_val = vdupq_n_f16(val);
    size_t i = 0;
    // Process vectorized portion
    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));
        v_res = vmulq_f16(v_a, v_val);
        vst1q_f16(reinterpret_cast<float16_t*>(a_dst + i), v_res);
    }
    // Process the remaining elements
    for (; i < size; ++i) {
        a_dst[i] = a[i] * val;
    }
}
#endif

inline void attn_softmax_kernel(float* a,
                                void* a_dst,
                                float scale,
                                float* alibi,
                                void* attn_mask,
                                uint8_t* causal_mask,
                                bool select_nfltmax_at_0,
                                size_t len,
                                size_t total_size,
                                ov::element::Type attn_mask_prec,
                                ov::element::Type dst_precision,
                                float alibi_slope = 0) {
    using func_fp32_type = void (*)(float*, float, const float*, const float*, const uint8_t*, bool, size_t, float, float&);
    using func_bf16_type = void (*)(float*, float, const float*, const ov::bfloat16*, const uint8_t*, bool, size_t, float, float&);
    static constexpr func_fp32_type funcs_fp32[] = {
        scale_add2_reduce_max<false, false, false>,
        scale_add2_reduce_max<false, false, true>,
        scale_add2_reduce_max<false, true, false>,
        scale_add2_reduce_max<false, true, true>,
        scale_add2_reduce_max<true, false, false>,
        scale_add2_reduce_max<true, false, true>,
        scale_add2_reduce_max<true, true, false>,
        scale_add2_reduce_max<true, true, true>
    };
    static constexpr func_bf16_type funcs_bf16[] = {
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
    if (attn_mask_prec == ov::element::f32) {
        funcs_fp32[dispatch](a, scale, alibi, static_cast<const float*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);
    } else {
        funcs_bf16[dispatch](a, scale, alibi, static_cast<const ov::bfloat16*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);
    }

    float sum = 0.0f;
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    // divide sum
    float scalar = 1.0f / sum;
    if (dst_precision == ov::element::f32) {
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
#if defined(OPENVINO_ARCH_ARM64) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void attn_softmax_kernel(ov::float16* a,
                                void* a_dst,
                                float scale,
                                ov::float16* alibi,
                                void* attn_mask,
                                uint8_t* causal_mask,
                                bool select_nfltmax_at_0,
                                size_t len,
                                size_t total_size,
                                ov::element::Type attn_mask_prec,
                                ov::element::Type dst_precision,
                                float alibi_slope = 0) {
    using func_fp32_type = void (*)(ov::float16*, float, const ov::float16*, const float*, const uint8_t*, bool, size_t, float, ov::float16&);
    using func_bf16_type = void (*)(ov::float16*, float, const ov::float16*, const ov::bfloat16*, const uint8_t*, bool, size_t, float, ov::float16&);
    using func_fp16_type = void (*)(ov::float16*, float, const ov::float16*, const ov::float16*, const uint8_t*, bool, size_t, float, ov::float16&);
    static constexpr func_fp32_type funcs_fp32[] = {
            scale_add2_reduce_max<false, false, false>,
            scale_add2_reduce_max<false, false, true>,
            scale_add2_reduce_max<false, true, false>,
            scale_add2_reduce_max<false, true, true>,
            scale_add2_reduce_max<true, false, false>,
            scale_add2_reduce_max<true, false, true>,
            scale_add2_reduce_max<true, true, false>,
            scale_add2_reduce_max<true, true, true>
    };
    static constexpr func_bf16_type funcs_bf16[] = {
            scale_add2_reduce_max<false, false, false>,
            scale_add2_reduce_max<false, false, true>,
            scale_add2_reduce_max<false, true, false>,
            scale_add2_reduce_max<false, true, true>,
            scale_add2_reduce_max<true, false, false>,
            scale_add2_reduce_max<true, false, true>,
            scale_add2_reduce_max<true, true, false>,
            scale_add2_reduce_max<true, true, true>
    };
    static constexpr func_fp16_type funcs_fp16[] = {
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
    ov::float16 max = std::numeric_limits<ov::float16>::lowest();
    if (attn_mask_prec == ov::element::f32) {
        funcs_fp32[dispatch](a, scale, alibi, static_cast<const float*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);
    } else if (attn_mask_prec == ov::element::f16) {
        funcs_fp16[dispatch](a, scale, alibi, static_cast<const ov::float16*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);
    } else {
        funcs_bf16[dispatch](a, scale, alibi, static_cast<const ov::bfloat16*>(attn_mask), causal_mask, select_nfltmax_at_0, len, alibi_slope, max);
    }

    ov::float16 sum = 0.0f;
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    // divide sum
    ov::float16 scalar = 1.0f / sum;
    multiply_scalar(a, static_cast<ov::float16*>(a_dst), scalar, len);
    // apply causual mask to final result instead of attn_score
    if (total_size > len)
        memset(static_cast<ov::float16*>(a_dst) + len, 0, sizeof(ov::float16) * (total_size - len));
}
#endif

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov