// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include <cassert>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#include "arm_neon.h"
#endif

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

// avx512/avx2 register length in byte
static constexpr size_t vec_len_avx512 = 64lu;
static constexpr size_t vec_len_avx2 = 32lu;
static constexpr size_t vec_len_neon = 16lu;
// avx512/avx2 register length in float
static constexpr size_t vec_len_f32_avx512 = vec_len_avx512 / sizeof(float);
static constexpr size_t vec_len_f32_avx2 = vec_len_avx2 / sizeof(float);
static constexpr size_t vec_len_f32_neon = vec_len_neon / sizeof(float);
static constexpr size_t vec_len_f16_neon = vec_len_neon / sizeof(ov::float16);

#ifdef HAVE_AVX512F
    inline __m512 cvt_bf16_to_fp32(const __m256i src) {
        __m512i y = _mm512_cvtepu16_epi32(src);
        return _mm512_castsi512_ps(_mm512_slli_epi32(y, 16));
    }

    // load addr to __m512 reg
    inline __m512 mm512_uni_loadu_ps(const float* a) {
        return _mm512_loadu_ps(a);
    }

    inline __m512 mm512_uni_loadu_ps(const ov::bfloat16* a) {
        auto vec_bf16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        return cvt_bf16_to_fp32(vec_bf16);
    }

    inline __m512 mm512_uni_loadu_ps(const ov::float16* a) {
        auto vec_f16 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a));
        return _mm512_cvtph_ps(vec_f16);
    }

    // load addr to __m512 reg
    inline __m512 mm512_uni_loadu_tail_ps(const float* a, size_t count) {
        __mmask16 mask = (1 << count) - 1;
        return _mm512_maskz_loadu_ps(mask, a);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const ov::bfloat16* a, size_t count) {
        auto mask = (1 << count) - 1;
        auto bf16_vec = _mm256_maskz_loadu_epi16(mask, a);
        return cvt_bf16_to_fp32(bf16_vec);
    }

    inline __m512 mm512_uni_loadu_tail_ps(const ov::float16* a, size_t count) {
        auto mask = (1 << count) - 1;
        auto f16_vec = _mm256_maskz_loadu_epi16(mask, a);
        return _mm512_cvtph_ps(f16_vec);
    }

    // store __m512 reg to addr
    inline void mm512_uni_storeu_ps(float* a,  __m512 v) {
        _mm512_storeu_ps(a, v);
    }
    inline void mm512_uni_storeu_ps(ov::bfloat16 *addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(addr), _mm512_cvtepi32_epi16(x));
    }

    inline void mm512_uni_storeu_ps(ov::float16* addr,  __m512 v) {
        __m256i vec_f16 = _mm512_cvtps_ph(v, 0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(addr), vec_f16);
    }

    // store __m512 reg to addr
    inline void mm512_uni_mask_storeu_ps(ov::bfloat16 *addr, __mmask16 mask_addr, __m512 xps) {
        __m512i xpi32 = _mm512_castps_si512(xps);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(xps, xps, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
    }

    inline void mm512_uni_storeu_tail_ps(float *addr, __m512 v, size_t count) {
        __mmask16 mask_addr = (1 << count) - 1;
        _mm512_mask_storeu_ps(addr, mask_addr, v);
    }

    inline void mm512_uni_storeu_tail_ps(ov::bfloat16 *addr, __m512 v, size_t count) {
        __mmask16 mask_addr = (1 << count) - 1;
        __m512i xpi32 = _mm512_castps_si512(v);
        __m512i nan = _mm512_set1_epi32(0xffff);
        auto mask = _mm512_cmp_ps_mask(v, v, _CMP_ORD_Q);
        __m512i ones = _mm512_set1_epi32(0x1);
        __m512i vec_bias = _mm512_set1_epi32(0x7fff);
        auto x = _mm512_and_si512(_mm512_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm512_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm512_srli_epi32(_mm512_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm512_mask_blend_epi32(mask, nan, x);                     // Check NaN before converting back to bf16
        _mm512_mask_cvtepi32_storeu_epi16(addr, mask_addr, x);
    }

    inline void mm512_uni_storeu_tail_ps(ov::float16 *addr, __m512 v, size_t count) {
        __mmask16 mask_addr = (1 << count) - 1;
        __m256i vec_f16 = _mm512_cvtps_ph(v, 0);
        _mm256_mask_storeu_epi16(reinterpret_cast<__m256i *>(addr), mask_addr, vec_f16);
    }
#endif

#ifdef HAVE_AVX2
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

    // load addr to __m256 reg
    inline __m256 mm256_uni_loadu_ps(const float* a) {
        return _mm256_loadu_ps(a);
    }

    inline __m256 mm256_uni_loadu_ps(const ov::bfloat16* a) {
        auto vec_bf16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
        auto o = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vec_bf16), 16));
        return o;
    }

    inline __m256 mm256_uni_loadu_ps(const ov::float16* a) {
        auto vec_f16 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a));
        auto o = _mm256_cvtph_ps(vec_f16);
        return o;
    }

    // load addr tail to __m256 reg
    inline __m256 mm256_uni_loadu_tail_ps(const float* a, const size_t count) {
        auto mask = get_mask(count);
        return _mm256_maskload_ps(a, mask);
    }

    inline __m256 mm256_uni_loadu_tail_ps(const ov::bfloat16* a, const size_t count) {
        assert("AVX2 version of bfloat16 tail load is just for compilation pass");
        ov::bfloat16 tmp_values[8] = {0};
        std::memcpy(tmp_values, a, count * sizeof(ov::bfloat16));
        return mm256_uni_loadu_ps(tmp_values);
    }

    inline __m256 mm256_uni_loadu_tail_ps(const ov::float16* a, const size_t count) {
        ov::float16 tmp_values[8] = {0};
        std::memcpy(tmp_values, a, count * sizeof(ov::float16));
        return mm256_uni_loadu_ps(tmp_values);
    }

    // store __m256 reg to addr
    inline void mm256_uni_storeu_ps(float* a,  __m256 v) {
        _mm256_storeu_ps(a, v);
    }

    inline void mm256_uni_storeu_ps(ov::bfloat16 *addr, __m256 xps) {
        __m256i xpi32 = _mm256_castps_si256(xps);
        __m256i nan = _mm256_set1_epi32(0xffff);
        __m256i mask = _mm256_castps_si256(_mm256_cmp_ps(xps, xps, _CMP_ORD_Q));
        __m256i ones = _mm256_set1_epi32(0x1);
        __m256i vec_bias = _mm256_set1_epi32(0x7fff);
        auto x = _mm256_and_si256(_mm256_srli_epi32(xpi32, 16), ones); // LSB = x[16]
        x = _mm256_add_epi32(x, vec_bias);                             // rounding_bias = 0x7fff + LSB
        x = _mm256_srli_epi32(_mm256_add_epi32(x, xpi32), 16);         // x = (x + rounding_bias) >> 16;
        x = _mm256_blendv_epi8(nan, x, mask);                          // Check NaN before converting back to bf16
        x = _mm256_packus_epi32(x, x);
        x = _mm256_permute4x64_epi64(x, 0xd8);
        __m128i bf16_o = _mm256_extractf128_si256(x, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(addr), bf16_o);
    }

    inline void mm256_uni_storeu_ps(ov::float16* a,  __m256 v) {
        __m128i vec_f16 = _mm256_cvtps_ph(v, 0);
        _mm_storeu_si128(reinterpret_cast<__m128i *>(a), vec_f16);
    }

    // store __m256 to addr
    inline void mm256_uni_storeu_tail_ps(float *addr, __m256 v, size_t count) {
        const auto mask = get_mask(count);
        return _mm256_maskstore_ps(addr, mask, v);
    }

    inline void hsum(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_add_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_add_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_add_ps(x, y);              // x: 01234567 x x x x x x x
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
    inline void hmin(__m256& x) {
        __m256 y;                             // x:  0 1 2 3   4 5 6 7
        y = _mm256_permute_ps(x, 0x39);       // y:  1 2 3 0   5 6 7 4
        x = _mm256_min_ps(x, y);              // X:  01 12 23 30  45 56 67 74
        y = _mm256_permute_ps(x, 0x4e);       // y:  23 30 01 12  67 74 45 56
        x = _mm256_min_ps(x, y);              // x: 0123 x x x   4567 x x x
        y = _mm256_permute2f128_ps(x, x, 1);  // y: 4567 x x x  0123 x x x
        x = _mm256_min_ps(x, y);              // x: 01234567 x x x x x x x
    }
#endif

#ifdef OPENVINO_ARCH_ARM64
    inline float32x4_t exp_ps_neon_f32(const float32x4_t& src) {
        const auto c1 = vreinterpretq_f32_u32(vdupq_n_u32(0x3f7ffff6));
        const auto c2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3efffedb));
        const auto c3 = vreinterpretq_f32_u32(vdupq_n_u32(0x3e2aaf33));
        const auto c4 = vreinterpretq_f32_u32(vdupq_n_u32(0x3d2b9f17));
        const auto c5 = vreinterpretq_f32_u32(vdupq_n_u32(0x3c072010));

        const auto shift   = vreinterpretq_f32_u32(vdupq_n_u32(0x4b00007f)); // 2^23 + 127 = 0x1.0000fep23f
        const auto one   = vdupq_n_f32(1.0f); // 1
        const auto two   = vdupq_n_f32(2.0f); // 2
        const auto inv_ln2 = vreinterpretq_f32_u32(vdupq_n_u32(0x3fb8aa3b));
        const auto neg_ln2_hi = vreinterpretq_f32_u32(vdupq_n_u32(0xbf317200));
        const auto neg_ln2_lo = vreinterpretq_f32_u32(vdupq_n_u32(0xb5bfbe8e));

        const auto inf       = vdupq_n_f32(std::numeric_limits<float>::infinity());
        const auto max_input = vdupq_n_f32(88.37f); // Approximately ln(2^127.5)
        const auto zero      = vdupq_n_f32(0.f);
        const auto min_input = vdupq_n_f32(-86.64f); // Approximately ln(2^-125)

        const auto z     = vmlaq_f32(shift, src, inv_ln2);
        auto n     = z - shift;
        n = vsubq_f32(n, one);
        const auto scale = vreinterpretq_f32_u32(vreinterpretq_u32_f32(z) << 23); // 2^n

        const auto r_hi = vfmaq_f32(src, n, neg_ln2_hi);
        const auto r    = vfmaq_f32(r_hi, n, neg_ln2_lo);

        const auto r2 = r * r;

        const auto p1     = c1 * r;
        const auto p23    = vfmaq_f32(c2, c3, r);
        const auto p45    = vfmaq_f32(c4, c5, r);
        const auto p2345  = vfmaq_f32(p23, p45, r2);
        const auto p12345 = vfmaq_f32(p1, p2345, r2);

        auto poly = vfmaq_f32(scale, p12345, scale);
        poly = vmulq_f32(poly, two);

        poly = vbslq_f32(vcltq_f32(src, min_input), zero, poly);
        poly = vbslq_f32(vcgtq_f32(src, max_input), inf, poly);

        return poly;
    }
    inline float32x4_t __vld1q_f32(const ov::bfloat16* a) {
        uint16x4_t vec_bf16 = vld1_u16(reinterpret_cast<const uint16_t*>(a));

        float32x4_t vec_f32 = vcvtq_f32_u32(vmovl_u16(vec_bf16));
        return vec_f32;
    }
    inline float32x4_t __vld1q_f32(const float* a) {
        return vld1q_f32(a);
    }
    inline float32x4_t __vld1q_f32(const ov::float16* a) {
        auto _a = reinterpret_cast<const float16_t*>(a);
        return vcvt_f32_f16(vld1_f16(_a));
    }
    inline void __vst1q_f32(float* a, float32x4_t b) {
        vst1q_f32(a, b);
    }
    inline void __vst1q_f32(ov::float16* a, float32x4_t b) {
        float16x4_t v_f16 = vcvt_f16_f32(b);
        vst1_f16(reinterpret_cast<float16_t*>(a), v_f16);
    }
    inline void __vst1q_f32(ov::bfloat16* a, float32x4_t b) {
        uint32x4_t v_int32 = vreinterpretq_u32_f32(b);
        uint16x4_t v_bf16 = vshrn_n_u32(v_int32, 16);

        vst1_u16(reinterpret_cast<uint16_t*>(a), v_bf16);
    }

#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    inline float16x8_t exp_ps_neon_f16(float16x8_t x) {
        const float32x4_t x_high = vcvt_f32_f16(vget_high_f16(x));
        const float32x4_t x_low  = vcvt_f32_f16(vget_low_f16(x));

        // We use f32 to maintain accuracy
        const float16x8_t res = vcombine_f16(vcvt_f16_f32(exp_ps_neon_f32(x_low)), vcvt_f16_f32(exp_ps_neon_f32(x_high)));
        return res;
    }
    inline float16_t hsum(float16x8_t vec) {
        float16x4_t sum1 = vpadd_f16(vget_low_f16(vec), vget_high_f16(vec));
        float16x4_t sum2 = vpadd_f16(sum1, sum1);
        float16x4_t sum3 = vpadd_f16(sum2, sum2);
        return vget_lane_f16(sum3, 0);
    }
#endif
}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov
