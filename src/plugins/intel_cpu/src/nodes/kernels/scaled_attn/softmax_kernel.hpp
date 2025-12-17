// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "openvino/core/except.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/core/type/float16.hpp"
#include "utils/general_utils.h"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>

#    include "common.hpp"
#endif

#if defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
#        include "arm_sve.h"
#    endif
#    include "arm_neon.h"
#    include "common.hpp"
#endif

namespace ov::Extensions::Cpu::XARCH {

#if defined(OPENVINO_ARCH_ARM64)
namespace detail {
inline void zero_out_dst(void* a_dst, ov::element::Type dst_precision, size_t total_size) {
    if (total_size == 0) {
        return;
    }
    if (dst_precision == ov::element::f32) {
        memset(static_cast<float*>(a_dst), 0, sizeof(float) * total_size);
    } else {
        memset(static_cast<ov::float16*>(a_dst), 0, sizeof(ov::float16) * total_size);
    }
}

inline bool handle_empty_len(size_t len, void* a_dst, ov::element::Type dst_precision, size_t total_size) {
    if (len != 0) {
        return false;
    }
    zero_out_dst(a_dst, dst_precision, total_size);
    return true;
}

template <typename T>
inline float to_float(T value) {
    return static_cast<float>(value);
}

template <>
inline float to_float<float>(float value) {
    return value;
}

template <typename T>
inline bool handle_inf_logits(const T* a,
                              void* a_dst,
                              ov::element::Type dst_precision,
                              size_t len,
                              size_t total_size,
                              const float* sink) {
    size_t inf_count = 0;
    if (sink != nullptr && std::isinf(*sink) && *sink > 0.0F) {
        inf_count++;
    }
    for (size_t i = 0; i < len; i++) {
        const float aval = to_float(a[i]);
        if (std::isinf(aval) && aval > 0.0F) {
            inf_count++;
        }
    }
    const float inv = inf_count ? (1.0F / static_cast<float>(inf_count)) : 0.0F;
    if (dst_precision == ov::element::f32) {
        auto* dst = static_cast<float*>(a_dst);
        for (size_t i = 0; i < len; i++) {
            const float aval = to_float(a[i]);
            dst[i] = (inf_count && std::isinf(aval) && aval > 0.0F) ? inv : 0.0F;
        }
        if (total_size > len) {
            memset(dst + len, 0, sizeof(float) * (total_size - len));
        }
    } else {
        auto* dst = static_cast<ov::float16*>(a_dst);
        for (size_t i = 0; i < len; i++) {
            const float aval = to_float(a[i]);
            dst[i] = (inf_count && std::isinf(aval) && aval > 0.0F) ? ov::float16(inv) : ov::float16(0.0F);
        }
        if (total_size > len) {
            memset(dst + len, 0, sizeof(ov::float16) * (total_size - len));
        }
    }
    return true;
}
}  // namespace detail
#endif

#if defined(HAVE_AVX2)
inline void exp_ps_avx2(__m256& src) {
#    define REPEAT8(x) x, x, x, x, x, x, x, x
    static const uint32_t c_min[] = {REPEAT8(0xc2aeac50)};
    static const uint32_t c_max[] = {REPEAT8(0x42b17218)};
    static const uint32_t c_e[] = {REPEAT8(0x3fb8aa3b)};
    static const uint32_t c_half[] = {REPEAT8(0x3f000000)};
    static const uint32_t c_ln2[] = {REPEAT8(0x3f317218)};
    static const uint32_t c_1[] = {REPEAT8(0x3f800000)};
    static const uint32_t c_bias[] = {REPEAT8(0x0000007f)};
    static const uint32_t c_p1[] = {REPEAT8(0x3f7ffffb)};
    static const uint32_t c_p2[] = {REPEAT8(0x3efffee3)};
    static const uint32_t c_p3[] = {REPEAT8(0x3e2aad40)};
    static const uint32_t c_p4[] = {REPEAT8(0x3d2b9d0d)};
    static const uint32_t c_p5[] = {REPEAT8(0x3c07cfce)};
    static const uint32_t c_2[] = {REPEAT8(0x40000000)};
#    undef REPEAT8
    static constexpr int n_mantissa_bits = 23;
    __m256 exp_ln_flt_min_f = _mm256_loadu_ps(reinterpret_cast<const float*>(c_min));      // log(FLT_MIN)
    __m256 exp_ln_flt_max_f = _mm256_loadu_ps(reinterpret_cast<const float*>(c_max));      // log(FLT_MAX)
    __m256 exp_log2ef = _mm256_loadu_ps(reinterpret_cast<const float*>(c_e));              // log2(e)
    __m256 half = _mm256_loadu_ps(reinterpret_cast<const float*>(c_half));                 // 0.5f
    __m256 ln2f = _mm256_loadu_ps(reinterpret_cast<const float*>(c_ln2));                  // ln(2)
    __m256 one = _mm256_loadu_ps(reinterpret_cast<const float*>(c_1));                     // 1.0F
    __m256i exponent_bias = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(c_bias));  // 127
    __m256 exp_pol1 = _mm256_loadu_ps(reinterpret_cast<const float*>(c_p1));               // p1 = 0.999999701f
    __m256 exp_pol2 = _mm256_loadu_ps(reinterpret_cast<const float*>(c_p2));               // p2 = 0.499991506f
    __m256 exp_pol3 = _mm256_loadu_ps(reinterpret_cast<const float*>(c_p3));               // p3 = 0.166676521f
    __m256 exp_pol4 = _mm256_loadu_ps(reinterpret_cast<const float*>(c_p4));               // p4 = 0.0418978221f
    __m256 exp_pol5 = _mm256_loadu_ps(reinterpret_cast<const float*>(c_p5));               // p5 = 0.00828929059f
    __m256 two = _mm256_loadu_ps(reinterpret_cast<const float*>(c_2));                     // 2
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

template <bool has_alibi, bool has_attn_mask, bool has_causal_mask, bool has_sparse_mask, typename T>
inline void scale_add2_reduce_max(float* a,
                                  float scale,
                                  const float* alibi_lookup,
                                  const T* attn_mask,
                                  const uint8_t* causal_mask,
                                  bool select_nfltmax_at_0,  // true:  0 in mask set -FLT_MAX
                                  size_t size,
                                  float alibi_slope,
                                  float& max,
                                  const uint8_t* sparse_mask,
                                  size_t sparse_block_size) {
    // Check sparse_block_size is a multiple of vector length for correct sparse_mask indexing
#if defined(HAVE_AVX512F)
    constexpr size_t vec_len = vec_len_f32_avx512;
#elif defined(HAVE_AVX2)
    constexpr size_t vec_len = vec_len_f32_avx2;
#elif defined(OPENVINO_ARCH_ARM64)
    constexpr size_t vec_len = vec_len_f32_neon;
#else
    constexpr size_t vec_len = 1;
#endif
    if (has_sparse_mask && (sparse_block_size % vec_len != 0)) {
        OPENVINO_THROW("sparse_block_size must be a multiple of vector length (",
                       vec_len,
                       ") for correct sparse_mask indexing");
    }
    OPENVINO_ASSERT(ov::intel_cpu::implication(has_alibi, alibi_lookup), "CPU: alibi_lookup should not be nullptr.");
    OPENVINO_ASSERT(ov::intel_cpu::implication(has_attn_mask, attn_mask), "CPU: attn_mask should not be nullptr.");
    OPENVINO_ASSERT(ov::intel_cpu::implication(has_causal_mask, causal_mask),
                    "CPU: causal_mask should not be nullptr.");
    OPENVINO_ASSERT(ov::intel_cpu::implication(has_sparse_mask, sparse_mask),
                    "CPU: sparse_mask should not be nullptr.");
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_max0 = _mm512_set1_ps(std::numeric_limits<float>::lowest());
    auto v_max1 = v_max0;
    auto v_max2 = v_max0;
    auto v_max3 = v_max0;
    auto v_scale = _mm512_set1_ps(scale);
    auto v_zeroi32 = _mm512_setzero_epi32();
    auto v_nfltmax = _mm512_set1_ps(-FLT_MAX);
    auto kmask_xor = _cvtu32_mask16(select_nfltmax_at_0 ? 0xFFFF : 0);
    auto v_alibi_slope = _mm512_set1_ps(alibi_slope);
    __m512 v_a;
    // process vector body
    // unroll to avoid dependency caused by _mm256_max_ps
    for (; i + 4 * vec_len_f32_avx512 <= size; i += 4 * vec_len_f32_avx512) {
#    define ITEM(n)                                                                                          \
        v_a = _mm512_loadu_ps(a + i + n * vec_len_f32_avx512);                                               \
        v_a = _mm512_mul_ps(v_a, v_scale);                                                                   \
        if (has_alibi) {                                                                                     \
            auto v_lookup = _mm512_loadu_ps(alibi_lookup + i + n * vec_len_f32_avx512);                      \
            v_a = _mm512_fmadd_ps(v_lookup, v_alibi_slope, v_a);                                             \
        }                                                                                                    \
        if (has_attn_mask) {                                                                                 \
            auto v_mask = mm512_uni_loadu_ps(attn_mask + i + n * vec_len_f32_avx512);                        \
            v_a = _mm512_add_ps(v_a, v_mask);                                                                \
        }                                                                                                    \
        if (has_sparse_mask) {                                                                               \
            size_t mask_idx = (i + n * vec_len_f32_avx512) / sparse_block_size;                              \
            uint8_t mask_val = sparse_mask[mask_idx];                                                        \
            __m512 v_mask_block = _mm512_set1_ps(mask_val ? 0.f : -FLT_MAX);                                 \
            v_a = _mm512_add_ps(v_a, v_mask_block);                                                          \
        }                                                                                                    \
        if (has_causal_mask) {                                                                               \
            auto v_maski8 =                                                                                  \
                _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i + n * vec_len_f32_avx512)); \
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);                                                 \
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);                         \
            kmask = _kxor_mask16(kmask, kmask_xor);                                                          \
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                                               \
        }                                                                                                    \
        v_max##n = _mm512_max_ps(v_max##n, v_a);                                                             \
        _mm512_storeu_ps(a + i + n * vec_len_f32_avx512, v_a);

        ITEM(0);
        ITEM(1);
        ITEM(2);
        ITEM(3);
#    undef ITEM
    }
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

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            __m512 v_mask_block = _mm512_set1_ps(mask_val ? 0.f : -FLT_MAX);
            v_a = _mm512_add_ps(v_a, v_mask_block);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max0 = _mm512_max_ps(v_max0, v_a);
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

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            __m512 v_mask_block = _mm512_set1_ps(mask_val ? 0.f : -FLT_MAX);
            v_a = _mm512_add_ps(v_a, v_mask_block);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(causal_mask + i));
            auto v_maski32 = _mm512_cvtepi8_epi32(v_maski8);
            auto kmask = _mm512_cmp_epi32_mask(v_maski32, v_zeroi32, _MM_CMPINT_NE);  // !=0
            kmask = _kxor_mask16(kmask, kmask_xor);                                   // reverse, mask at ==0
            v_a = _mm512_mask_blend_ps(kmask, v_a, v_nfltmax);                        // mask => -FLT_MAX
        }
        v_max0 = _mm512_mask_max_ps(v_max0, mask, v_a, v_max0);
        _mm512_mask_storeu_ps(a + i, mask, v_a);

        i += (size - i);
    }

    v_max0 = _mm512_max_ps(v_max0, v_max1);
    v_max2 = _mm512_max_ps(v_max2, v_max3);
    v_max0 = _mm512_max_ps(v_max0, v_max2);
    max = _mm512_reduce_max_ps(v_max0);
#elif defined(HAVE_AVX2)
    auto v_max0 = _mm256_set1_ps(std::numeric_limits<float>::lowest());
    auto v_max1 = v_max0;
    auto v_max2 = v_max0;
    auto v_max3 = v_max0;
    __m256 v_a;
    auto v_scale = _mm256_set1_ps(scale);
    auto v_zeroi32 = _mm256_setzero_si256();
    auto v_mask_xor = _mm256_set1_epi32(select_nfltmax_at_0 ? -1 : 0);
    auto v_nfltmax = _mm256_set1_ps(-FLT_MAX);
    auto v_alibi_slope = _mm256_set1_ps(alibi_slope);
    // process vector body
    // unroll to avoid dependency caused by _mm512_max_ps
    for (; i + 4 * vec_len_f32_avx2 <= size; i += 4 * vec_len_f32_avx2) {
#    define ITEM(n)                                                                                                    \
        v_a = _mm256_loadu_ps(a + i + n * vec_len_f32_avx2);                                                           \
        v_a = _mm256_mul_ps(v_a, v_scale);                                                                             \
        if (has_alibi) {                                                                                               \
            auto v_lookup = _mm256_loadu_ps(alibi_lookup + i + n * vec_len_f32_avx2);                                  \
            v_a = _mm256_fmadd_ps(v_lookup, v_alibi_slope, v_a);                                                       \
        }                                                                                                              \
        if (has_attn_mask) {                                                                                           \
            auto v_mask = mm256_uni_loadu_ps(attn_mask + i + n * vec_len_f32_avx2);                                    \
            v_a = _mm256_add_ps(v_a, v_mask);                                                                          \
        }                                                                                                              \
        if (has_sparse_mask) {                                                                                         \
            size_t mask_idx = (i + n * vec_len_f32_avx2) / sparse_block_size;                                          \
            uint8_t mask_val = sparse_mask[mask_idx];                                                                  \
            __m256 v_mask_block = _mm256_set1_ps(mask_val ? 0.f : -FLT_MAX);                                           \
            v_a = _mm256_add_ps(v_a, v_mask_block);                                                                    \
        }                                                                                                              \
        if (has_causal_mask) {                                                                                         \
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<__m128i const*>(causal_mask + i + n * vec_len_f32_avx2)); \
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);                                                           \
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                                                      \
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                                                       \
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));                                    \
        }                                                                                                              \
        v_max##n = _mm256_max_ps(v_max##n, v_a);                                                                       \
        _mm256_storeu_ps(a + i + n * vec_len_f32_avx2, v_a);

        ITEM(0);
        ITEM(1);
        ITEM(2);
        ITEM(3);
#    undef ITEM
    }

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

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            __m256 v_mask_block = _mm256_set1_ps(mask_val ? 0.f : -FLT_MAX);
            v_a = _mm256_add_ps(v_a, v_mask_block);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_max0 = _mm256_max_ps(v_max0, v_a);
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

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            __m256 v_mask_block = _mm256_set1_ps(mask_val ? 0.f : -FLT_MAX);
            v_a = _mm256_add_ps(v_a, v_mask_block);
        }

        if (has_causal_mask) {
            auto v_maski8 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(causal_mask + i));
            auto v_maski32 = _mm256_cvtepi8_epi32(v_maski8);
            v_maski32 = _mm256_cmpeq_epi32(v_maski32, v_zeroi32);                    // ==0
            v_maski32 = _mm256_xor_si256(v_maski32, v_mask_xor);                     // reverse, mask at ==0
            v_a = _mm256_blendv_ps(v_nfltmax, v_a, _mm256_castsi256_ps(v_maski32));  // mask => -FLT_MAX
        }

        v_a = _mm256_blendv_ps(v_max0, v_a, _mm256_castsi256_ps(mask));
        v_max0 = _mm256_max_ps(v_max0, v_a);
        _mm256_maskstore_ps(a + i, mask, v_a);

        i += (size - i);
    }
    v_max0 = _mm256_max_ps(v_max0, v_max1);
    v_max2 = _mm256_max_ps(v_max2, v_max3);
    v_max0 = _mm256_max_ps(v_max0, v_max2);
    hmax(v_max0);
    max = _mm256_cvtss_f32(v_max0);
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

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            float32x4_t v_mask_block = vdupq_n_f32(mask_val ? 0.0F : -FLT_MAX);
            v_a = vaddq_f32(v_a, v_mask_block);
        }

        if (has_causal_mask) {
            uint8x16_t v_maski8 = vld1q_u8(causal_mask + i);
            uint16x8_t v_maski16 = vmovl_u8(vget_low_u8(v_maski8));
            uint32x4_t v_maski32_low = vmovl_u16(vget_low_u16(v_maski16));
            uint32x4_t v_maski32_high = vmovl_u16(vget_high_u16(v_maski16));
            uint32x4_t v_maski32[2] = {v_maski32_low, v_maski32_high};
            for (const auto& mask_vec : v_maski32) {
                uint32x4_t kmask = vceqq_u32(mask_vec, v_zeroi32);  // ==0
                v_a = vbslq_f32(kmask, v_nfltmax, v_a);             // mask => -FLT_MAX
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

        if (has_attn_mask) {
            a[i] += attn_mask[i];
        }

        if (has_sparse_mask) {
            size_t mask_idx = i / sparse_block_size;
            uint8_t mask_val = sparse_mask[mask_idx];
            a[i] += (mask_val ? 0.0F : -FLT_MAX);
        }

        if (has_causal_mask) {
            if (select_nfltmax_at_0) {
                if (causal_mask[i] == 0) {
                    a[i] = -FLT_MAX;
                }
            } else {
                if (causal_mask[i] != 0) {
                    a[i] = -FLT_MAX;
                }
            }
        }

        max = a[i] > max ? a[i] : max;
    }
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
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
    size_t i = 0;
    constexpr float16_t min_f16 = std::numeric_limits<float16_t>::lowest();
#    if defined(HAVE_SVE)
    svfloat16_t v_max = svdup_n_f16(min_f16);
    svfloat16_t v_scale = svdup_n_f16(static_cast<float16_t>(scale));
    svfloat16_t v_a;
    svuint16_t v_zeroi16 = svdup_n_u16(0);
    svfloat16_t v_nfltmax = svdup_n_f16(min_f16);
    svfloat16_t v_alibi_slope = svdup_n_f16(static_cast<float16_t>(alibi_slope));

    svbool_t mask_xor = svptrue_b16();
    if (!select_nfltmax_at_0)
        mask_xor = svnot_z(svptrue_b16(), mask_xor);

    svbool_t pg_f16 = svptrue_b16();
    svbool_t pg_u8 = svptrue_b8();
    svbool_t pg_u16 = svptrue_b16();
    size_t vec_len = svcnth();

    for (; i + vec_len <= size; i += vec_len) {
        v_a = svld1_f16(pg_f16, reinterpret_cast<const float16_t*>(a + i));
        v_a = svmul_f16_z(pg_f16, v_a, v_scale);

        if (has_alibi) {
            svfloat16_t v_lookup = svld1_f16(pg_f16, reinterpret_cast<const float16_t*>(alibi_lookup + i));
            v_a = svmla_f16_z(pg_f16, v_a, v_lookup, v_alibi_slope);
        }

        if (has_attn_mask) {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, ov::float16>,
                          "attn_mask must be float or float16 type.");
            if constexpr (std::is_same_v<T, float>) {
                svfloat16_t zero = svdup_n_f16(0.0F);
                size_t inc_low = (vec_len + 1) / 2;
                size_t inc_high = vec_len / 2;
                svbool_t pg_f32_low = svwhilelt_b32(0, static_cast<int>(inc_low));
                svbool_t pg_f32_high = svwhilelt_b32(0, static_cast<int>(inc_high));

                svfloat16_t low_f16 = svzip1_f16(v_a, zero);
                svfloat16_t high_f16 = svzip2_f16(v_a, zero);
                svfloat32_t low_f32 = svcvt_f32_f16_x(pg_f32_low, low_f16);
                svfloat32_t high_f32 = svcvt_f32_f16_x(pg_f32_high, high_f16);

                svuint32_t idx_even = svindex_u32(0, 2);
                svuint32_t idx_odd = svindex_u32(1, 2);
                svfloat32_t mask_low = svld1_gather_u32index_f32(pg_f32_low, attn_mask + i, idx_even);
                svfloat32_t mask_high = svld1_gather_u32index_f32(pg_f32_high, attn_mask + i, idx_odd);
                low_f32 = svadd_f32_x(pg_f32_low, low_f32, mask_low);
                high_f32 = svadd_f32_x(pg_f32_high, high_f32, mask_high);

                svfloat16_t low_f16_out = svcvt_f16_f32_x(pg_f32_low, low_f32);
                svfloat16_t high_f16_out = svcvt_f16_f32_x(pg_f32_high, high_f32);
                v_a = svuzp1(low_f16_out, high_f16_out);
            } else if constexpr (std::is_same_v<T, ov::float16>) {
                svfloat16_t v_mask = svld1_f16(pg_f16, reinterpret_cast<const float16_t*>(attn_mask + i));
                v_a = svadd_f16_z(pg_f16, v_a, v_mask);
            }
        }

        if (has_causal_mask) {
            svuint8_t v_maski8 = svld1_u8(pg_u8, causal_mask + i);
            svuint16_t v_maski16 = svtrn1_u16(svreinterpret_u16_u8(v_maski8), svdup_n_u16(0));
            svbool_t kmask = svcmpeq_u16(pg_u16, v_maski16, v_zeroi16);
            kmask = sveor_z(pg_u16, kmask, mask_xor);
            v_a = svsel_f16(kmask, v_nfltmax, v_a);
        }

        v_max = svmax_f16_z(pg_f16, v_max, v_a);
        svst1_f16(pg_f16, reinterpret_cast<float16_t*>(a + i), v_a);
    }
    max = svmaxv_f16(svptrue_b16(), v_max);
#    elif defined(HAVE_NEON_FP16)
    float16x8_t v_max = vdupq_n_f16(min_f16);
    float16x8_t v_scale = vdupq_n_f16(static_cast<float16_t>(scale));
    float16x8_t v_a;
    uint16x8_t v_zeroi16 = vdupq_n_u16(0);
    float16x8_t v_nfltmax = vdupq_n_f16(min_f16);
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
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, ov::float16>,
                          "attn_mask must be float or float16 type.");
            float16x8_t v_mask;
            if constexpr (std::is_same_v<T, float>) {
                float32x4_t m0 = vld1q_f32(attn_mask + i);
                float32x4_t m1 = vld1q_f32(attn_mask + i + vec_len_f32_neon);
                v_mask = vcombine_f16(vcvt_f16_f32(m0), vcvt_f16_f32(m1));
            } else if constexpr (std::is_same_v<T, ov::float16>) {
                v_mask = vld1q_f16(reinterpret_cast<const float16_t*>(attn_mask + i));
            }
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
    max = vmaxvq_f16(v_max);
#    endif
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
                    a[i] = min_f16;
            } else {
                if (causal_mask[i] != 0)
                    a[i] = min_f16;
            }
        }
        max = a[i] > max ? a[i] : max;
    }
}
#endif

#if defined(HAVE_AVX512F)
static inline void exp_ps_avx512(__m512& src) {
#    define REPEAT16(x) x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x
    static const uint32_t c_min[] = {REPEAT16(0xc2aeac50)};
    static const uint32_t c_max[] = {REPEAT16(0x42b17218)};
    static const uint32_t c_e[] = {REPEAT16(0x3fb8aa3b)};
    static const uint32_t c_half[] = {REPEAT16(0x3f000000)};
    static const uint32_t c_ln2[] = {REPEAT16(0x3f317218)};
    static const uint32_t c_1[] = {REPEAT16(0x3f800000)};
    static const uint32_t c_bias[] = {REPEAT16(0x0000007f)};
    static const uint32_t c_p1[] = {REPEAT16(0x3f7ffffb)};
    static const uint32_t c_p2[] = {REPEAT16(0x3efffee3)};
    static const uint32_t c_p3[] = {REPEAT16(0x3e2aad40)};
    static const uint32_t c_p4[] = {REPEAT16(0x3d2b9d0d)};
    static const uint32_t c_p5[] = {REPEAT16(0x3c07cfce)};
    static const uint32_t c_2[] = {REPEAT16(0x40000000)};
#    undef REPEAT16
    static constexpr int n_mantissa_bits = 23;
    __m512 exp_ln_flt_min_f = _mm512_loadu_ps(reinterpret_cast<const float*>(c_min));  // log(FLT_MIN)
    __m512 exp_ln_flt_max_f = _mm512_loadu_ps(reinterpret_cast<const float*>(c_max));  // log(FLT_MAX)
    __m512 exp_log2ef = _mm512_loadu_ps(reinterpret_cast<const float*>(c_e));          // log2(e)
    __m512 half = _mm512_loadu_ps(reinterpret_cast<const float*>(c_half));             // 0.5f
    __m512 ln2f = _mm512_loadu_ps(reinterpret_cast<const float*>(c_ln2));              // ln(2)
    __m512 one = _mm512_loadu_ps(reinterpret_cast<const float*>(c_1));                 // 1.0F
    __m512i exponent_bias = _mm512_loadu_si512(c_bias);                                // 127
    __m512 exp_pol1 = _mm512_loadu_ps(reinterpret_cast<const float*>(c_p1));           // p1 = 0.999999701f
    __m512 exp_pol2 = _mm512_loadu_ps(reinterpret_cast<const float*>(c_p2));           // p2 = 0.499991506f
    __m512 exp_pol3 = _mm512_loadu_ps(reinterpret_cast<const float*>(c_p3));           // p3 = 0.166676521f
    __m512 exp_pol4 = _mm512_loadu_ps(reinterpret_cast<const float*>(c_p4));           // p4 = 0.0418978221f
    __m512 exp_pol5 = _mm512_loadu_ps(reinterpret_cast<const float*>(c_p5));           // p5 = 0.00828929059f
    __m512 two = _mm512_loadu_ps(reinterpret_cast<const float*>(c_2));                 // 2
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
    auto v_sum = _mm512_set1_ps(0.0F);
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
    auto v_sum = _mm256_set1_ps(0.0F);
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
#    if defined(HAVE_SVE)
    svfloat32_t v_a;
    svfloat32_t v_max = svdup_n_f32(max);
    svfloat32_t v_sum = svdup_n_f32(0.0F);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < size) {
        if (size - i < vec_len_f32_sve()) {
            inc = size - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        v_a = svld1_f32(pg, a + i);
        v_a = svsub_f32_z(pg, v_a, v_max);
        v_a = exp_ps_sve(pg, v_a);
        v_sum = svadd_f32_m(pg, v_sum, v_a);
        svst1_f32(pg, a + i, v_a);
        i += inc;
    }
    sum = svaddv_f32(svptrue_b32(), v_sum);
#    else
    float32x4_t v_a;
    float32x4_t v_max = vdupq_n_f32(max);
    float32x4_t v_sum = vdupq_n_f32(0.0F);

    while (i + vec_len_f32_neon <= size) {
        v_a = vld1q_f32(a + i);
        v_a = vsubq_f32(v_a, v_max);
        v_a = exp_ps_neon_f32(v_a);
        v_sum = vaddq_f32(v_sum, v_a);
        vst1q_f32(a + i, v_a);
        i += vec_len_f32_neon;
    }
    sum = vaddvq_f32(v_sum);
#    endif
#endif
    for (; i < size; i++) {
        a[i] = std::exp(a[i] - max);
        sum += a[i];
    }
}

#if defined(OPENVINO_ARCH_ARM64)
inline void exp_reduce_sum_f32(ov::float16* a, const ov::float16 max, const size_t size, ov::float16& sum) {
    size_t i = 0;
#    if defined(HAVE_SVE)
    svfloat32_t v_max = svdup_n_f32(static_cast<float>(max));
    svfloat32_t v_sum = svdup_n_f32(0.0F);

    svbool_t pg_f32 = svptrue_b32();
    svbool_t pg_f16 = svptrue_b16();
    svfloat16_t zero = svdup_n_f16(0.0);

    for (; i + svcnth() <= size; i += svcnth()) {
        svfloat16_t v_a_f16 = svld1_f16(pg_f16, reinterpret_cast<const float16_t*>(a + i));
        auto v_a_f16_low = svzip1_f16(v_a_f16, zero);
        auto v_a_f16_high = svzip2_f16(v_a_f16, zero);
        auto v_a_low = svcvt_f32_f16_x(pg_f16, v_a_f16_low);
        auto v_a_high = svcvt_f32_f16_x(pg_f16, v_a_f16_high);

        v_a_low = svsub_f32_x(pg_f32, v_a_low, v_max);
        v_a_high = svsub_f32_x(pg_f32, v_a_high, v_max);
        v_a_low = exp_ps_sve(pg_f32, v_a_low);
        v_a_high = exp_ps_sve(pg_f32, v_a_high);
        v_sum = svadd_f32_x(pg_f32, v_sum, v_a_low);
        v_sum = svadd_f32_x(pg_f32, v_sum, v_a_high);

        svfloat16_t v_result_low = svcvt_f16_f32_x(pg_f32, v_a_low);
        svfloat16_t v_result_high = svcvt_f16_f32_x(pg_f32, v_a_high);
        auto result = svuzp1(v_result_low, v_result_high);
        svst1_f16(pg_f16, reinterpret_cast<float16_t*>(a + i), result);
    }
    float total_sum = svaddv_f32(svptrue_b32(), v_sum);
#    else
    float32x4_t v_a;
    float32x4_t v_max = vdupq_n_f32(static_cast<float>(max));
    float32x4_t v_sum = vdupq_n_f32(0.0F);

    // Process 4 FP32 elements at a time
    for (; i + vec_len_f32_neon <= size; i += vec_len_f32_neon) {
        // Load FP16 and convert to FP32
        float16x4_t v_a_f16 = vld1_f16(reinterpret_cast<const float16_t*>(a + i));
        v_a = vcvt_f32_f16(v_a_f16);

        // Compute in FP32
        v_a = vsubq_f32(v_a, v_max);
        v_a = exp_ps_neon_f32(v_a);
        v_sum = vaddq_f32(v_sum, v_a);

        // Convert back to FP16 and store
        float16x4_t v_result_f16 = vcvt_f16_f32(v_a);
        vst1_f16(reinterpret_cast<float16_t*>(a + i), v_result_f16);
    }

    // Reduce sum
    float total_sum = vaddvq_f32(v_sum);
#    endif
    // Handle remaining elements
    for (; i < size; ++i) {
        const float val = std::exp(static_cast<float>(a[i] - max));
        a[i] = static_cast<ov::float16>(val);
        total_sum += val;
    }

    sum += static_cast<ov::float16>(total_sum);
}
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void exp_reduce_sum(ov::float16* a, const ov::float16 max, const size_t size, ov::float16& sum) {
    size_t i = 0;
#    if defined(HAVE_SVE)
    svfloat16_t v_a;
    svfloat16_t v_max = svdup_n_f16(max);
    svfloat16_t v_sum = svdup_n_f16(0.0F);
    svbool_t pg = svptrue_b16();
    size_t inc = vec_len_f16_sve();

    while (i < size) {
        if (size - i < vec_len_f16_sve()) {
            inc = size - i;
            pg = svwhilelt_b16(0, static_cast<int>(inc));
        }
        v_a = svld1_f16(pg, reinterpret_cast<float16_t*>(a + i));
        v_a = svsub_f16_z(pg, v_a, v_max);
        v_a = exp_ps_sve_f16(pg, v_a);
        v_sum = svadd_f16_m(pg, v_sum, v_a);
        svst1_f16(pg, reinterpret_cast<float16_t*>(a + i), v_a);
        i += inc;
    }
    sum = svaddv_f16(svptrue_b16(), v_sum);
#    else
    const size_t vec_len_f16_neon = 8;
    float16x8_t v_a;
    float16x8_t v_max = vdupq_n_f16(max);
    float16x8_t v_sum = vdupq_n_f16(0.0F);

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
#    endif
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
        mm512_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx512;
    }
    if (i < size) {
        __mmask16 mask = (1 << (size - i)) - 1;
        v_a = _mm512_maskz_loadu_ps(mask, a + i);
        v_a = _mm512_mul_ps(v_a, v_scale);
        mm512_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#elif defined(HAVE_AVX2)
    auto v_scale = _mm256_set1_ps(val);
    __m256 v_a = {0};
    while (i + vec_len_f32_avx2 <= size) {
        v_a = _mm256_loadu_ps(a + i);
        v_a = _mm256_mul_ps(v_a, v_scale);
        mm256_uni_storeu_ps(a_dst + i, v_a);
        i += vec_len_f32_avx2;
    }
    if (i < size) {
        auto mask = get_mask(size - i);
        v_a = _mm256_maskload_ps(a + i, mask);
        v_a = _mm256_mul_ps(v_a, v_scale);
        mm256_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
    svfloat32_t v_scale = svdup_n_f32(val);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < size) {
        if (size - i < vec_len_f32_sve()) {
            inc = size - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        svfloat32_t v_a = svld1_f32(pg, a + i);
        v_a = svmul_f32_z(pg, v_a, v_scale);
        svst1_f32(pg, a_dst + i, v_a);
        i += inc;
    }
#    else
    float32x4_t v_scale = vdupq_n_f32(val);
    while (i + vec_len_f32_neon <= size) {
        float32x4_t v_a = vld1q_f32(a + i);
        v_a = vmulq_f32(v_a, v_scale);
        vst1q_f32(a_dst + i, v_a);
        i += vec_len_f32_neon;
    }
#    endif
#endif
    for (; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
}

template <typename T, typename = std::enable_if_t<ov::intel_cpu::any_of_v<T, ov::bfloat16, ov::float16>>>
inline void multiply_scalar(const float* a, T* a_dst, const float val, const size_t size) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v_scale = _mm512_set1_ps(val);
    __m512 v_a = {0};
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
        mm512_uni_storeu_tail_ps(a_dst + i, v_a, size - i);

        i += (size - i);
    }
#else
    for (; i < size; i++) {
        a_dst[i] = a[i] * val;
    }
#endif
}

#if defined(OPENVINO_ARCH_ARM64)
inline void multiply_scalar(ov::float16* a, float* a_dst, const ov::float16 val, const size_t size) {
    float16x4_t v_a_f16;
    float32x4_t v_a, v_res;
    float32x4_t v_val = vdupq_n_f32(static_cast<float>(val));
    size_t i = 0;

    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a_f16 = vld1_f16(reinterpret_cast<const float16_t*>(a + i));
        v_a = vcvt_f32_f16(v_a_f16);

        v_res = vmulq_f32(v_a, v_val);

        vst1q_f32(reinterpret_cast<float*>(a_dst + i), v_res);
    }

    for (; i < size; ++i) {
        const auto a_f32 = static_cast<float>(a[i]);
        a_dst[i] = a_f32 * static_cast<float>(val);
    }
}
inline void multiply_scalar_f32(ov::float16* a, ov::float16* a_dst, const ov::float16 val, const size_t size) {
    float32x4_t v_a, v_res;
    float32x4_t v_val = vdupq_n_f32(val);
    size_t i = 0;
    for (; i + vec_len_f32_neon <= size; i += vec_len_f32_neon) {
        v_a = __vld1q_f32((a + i));
        v_res = vmulq_f32(v_a, v_val);
        __vst1q_f32(a_dst + i, v_res);
    }
    auto val_f32 = static_cast<float>(val);
    for (; i < size; ++i) {
        auto _a_f32 = static_cast<float>(a_dst[i]);
        auto _a_dst_f32 = val_f32 * _a_f32;
        a_dst[i] = static_cast<ov::float16>(_a_dst_f32);
    }
}
#endif

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline void multiply_scalar(ov::float16* a, ov::float16* a_dst, const ov::float16 val, const size_t size) {
    size_t i = 0;
#    if defined(HAVE_SVE)
    svfloat16_t v_scale = svdup_n_f16(val);
    size_t inc = vec_len_f16_sve();
    svbool_t pg = svptrue_b16();

    while (i < size) {
        if (size - i < vec_len_f16_sve()) {
            inc = size - i;
            pg = svwhilelt_b16(0, static_cast<int>(inc));
        }
        svfloat16_t v_a = svld1_f16(pg, reinterpret_cast<float16_t*>(a + i));
        v_a = svmul_f16_z(pg, v_a, v_scale);
        svst1_f16(pg, reinterpret_cast<float16_t*>(a_dst + i), v_a);
        i += inc;
    }
#    else
    float16x8_t v_a, v_res;
    float16x8_t v_val = vdupq_n_f16(val);
    for (; i + vec_len_f16_neon <= size; i += vec_len_f16_neon) {
        v_a = vld1q_f16(reinterpret_cast<const float16_t*>(a + i));
        v_res = vmulq_f16(v_a, v_val);
        vst1q_f16(reinterpret_cast<float16_t*>(a_dst + i), v_res);
    }
#    endif
    for (; i < size; ++i) {
        a_dst[i] = a[i] * val;
    }
}
#endif

template <typename T>
inline void attn_softmax_kernel(T* a,
                                void* a_dst,
                                float scale,
                                T* alibi,
                                void* attn_mask,
                                uint8_t* causal_mask,
                                bool select_nfltmax_at_0,
                                size_t len,
                                size_t total_size,
                                ov::element::Type attn_mask_prec,
                                ov::element::Type dst_precision,
                                const float* sink,
                                float alibi_slope = 0,
                                uint8_t* sparse_mask = nullptr,
                                size_t sparse_block_size = 1);
template <>
inline void attn_softmax_kernel<float>(float* a,
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
                                       const float* sink,
                                       float alibi_slope,
                                       uint8_t* sparse_mask,
                                       size_t sparse_block_size) {
    using func_fp32_type = void (*)(float*,
                                    float,
                                    const float*,
                                    const float*,
                                    const uint8_t*,
                                    bool,
                                    size_t,
                                    float,
                                    float&,
                                    const uint8_t*,
                                    size_t);
    using func_bf16_type = void (*)(float*,
                                    float,
                                    const float*,
                                    const ov::bfloat16*,
                                    const uint8_t*,
                                    bool,
                                    size_t,
                                    float,
                                    float&,
                                    const uint8_t*,
                                    size_t);
#if defined(OPENVINO_ARCH_ARM64)
    if (detail::handle_empty_len(len, a_dst, dst_precision, total_size)) {
        return;
    }
#endif
    using func_f16_type = void (*)(float*,
                                   float,
                                   const float*,
                                   const ov::float16*,
                                   const uint8_t*,
                                   bool,
                                   size_t,
                                   float,
                                   float&,
                                   const uint8_t*,
                                   size_t);
    static constexpr func_fp32_type funcs_fp32[] = {scale_add2_reduce_max<false, false, false, false>,
                                                    scale_add2_reduce_max<false, false, true, false>,
                                                    scale_add2_reduce_max<false, true, false, false>,
                                                    scale_add2_reduce_max<false, true, true, false>,
                                                    scale_add2_reduce_max<true, false, false, false>,
                                                    scale_add2_reduce_max<true, false, true, false>,
                                                    scale_add2_reduce_max<true, true, false, false>,
                                                    scale_add2_reduce_max<true, true, true, false>,
                                                    scale_add2_reduce_max<false, false, false, true>,
                                                    scale_add2_reduce_max<false, false, true, true>,
                                                    scale_add2_reduce_max<false, true, false, true>,
                                                    scale_add2_reduce_max<false, true, true, true>,
                                                    scale_add2_reduce_max<true, false, false, true>,
                                                    scale_add2_reduce_max<true, false, true, true>,
                                                    scale_add2_reduce_max<true, true, false, true>,
                                                    scale_add2_reduce_max<true, true, true, true>};
    static constexpr func_bf16_type funcs_bf16[] = {scale_add2_reduce_max<false, false, false, false>,
                                                    scale_add2_reduce_max<false, false, true, false>,
                                                    scale_add2_reduce_max<false, true, false, false>,
                                                    scale_add2_reduce_max<false, true, true, false>,
                                                    scale_add2_reduce_max<true, false, false, false>,
                                                    scale_add2_reduce_max<true, false, true, false>,
                                                    scale_add2_reduce_max<true, true, false, false>,
                                                    scale_add2_reduce_max<true, true, true, false>,
                                                    scale_add2_reduce_max<false, false, false, true>,
                                                    scale_add2_reduce_max<false, false, true, true>,
                                                    scale_add2_reduce_max<false, true, false, true>,
                                                    scale_add2_reduce_max<false, true, true, true>,
                                                    scale_add2_reduce_max<true, false, false, true>,
                                                    scale_add2_reduce_max<true, false, true, true>,
                                                    scale_add2_reduce_max<true, true, false, true>,
                                                    scale_add2_reduce_max<true, true, true, true>};
    static constexpr func_f16_type funcs_f16[] = {scale_add2_reduce_max<false, false, false, false>,
                                                  scale_add2_reduce_max<false, false, true, false>,
                                                  scale_add2_reduce_max<false, true, false, false>,
                                                  scale_add2_reduce_max<false, true, true, false>,
                                                  scale_add2_reduce_max<true, false, false, false>,
                                                  scale_add2_reduce_max<true, false, true, false>,
                                                  scale_add2_reduce_max<true, true, false, false>,
                                                  scale_add2_reduce_max<true, true, true, false>,
                                                  scale_add2_reduce_max<false, false, false, true>,
                                                  scale_add2_reduce_max<false, false, true, true>,
                                                  scale_add2_reduce_max<false, true, false, true>,
                                                  scale_add2_reduce_max<false, true, true, true>,
                                                  scale_add2_reduce_max<true, false, false, true>,
                                                  scale_add2_reduce_max<true, false, true, true>,
                                                  scale_add2_reduce_max<true, true, false, true>,
                                                  scale_add2_reduce_max<true, true, true, true>};
    int dispatch =
        (alibi ? 0b100 : 0) | (attn_mask ? 0b010 : 0) | (causal_mask ? 0b001 : 0) | (sparse_mask ? 0b1000 : 0);
    float max = std::numeric_limits<float>::lowest();
    if (attn_mask_prec == ov::element::f32) {
        funcs_fp32[dispatch](a,
                             scale,
                             alibi,
                             static_cast<const float*>(attn_mask),
                             causal_mask,
                             select_nfltmax_at_0,
                             len,
                             alibi_slope,
                             max,
                             sparse_mask,
                             sparse_block_size);
    } else if (attn_mask_prec == ov::element::bf16) {
        funcs_bf16[dispatch](a,
                             scale,
                             alibi,
                             static_cast<const ov::bfloat16*>(attn_mask),
                             causal_mask,
                             select_nfltmax_at_0,
                             len,
                             alibi_slope,
                             max,
                             sparse_mask,
                             sparse_block_size);
    } else {
        funcs_f16[dispatch](a,
                            scale,
                            alibi,
                            static_cast<const ov::float16*>(attn_mask),
                            causal_mask,
                            select_nfltmax_at_0,
                            len,
                            alibi_slope,
                            max,
                            sparse_mask,
                            sparse_block_size);
    }

    float sum = 0.0F;
    if (sink != nullptr) {
        max = max > (*sink) ? max : (*sink);
    }
#if defined(OPENVINO_ARCH_ARM64)
    if (std::isinf(max) && max > 0.0F) {
        detail::handle_inf_logits(a, a_dst, dst_precision, len, total_size, sink);
        return;
    }
#endif
    // exp sum
    exp_reduce_sum(a, max, len, sum);
    if (sink != nullptr) {
        sum += std::exp(*sink - max);
    }
    // divide sum
    float scalar = 1.0F / sum;
    if (dst_precision == ov::element::f32) {
        multiply_scalar(a, reinterpret_cast<float*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len) {
            memset(static_cast<float*>(a_dst) + len, 0, sizeof(float) * (total_size - len));
        }
    } else if (dst_precision == ov::element::bf16) {
        multiply_scalar(a, static_cast<ov::bfloat16*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len) {
            memset(static_cast<ov::bfloat16*>(a_dst) + len, 0, sizeof(ov::bfloat16) * (total_size - len));
        }
    } else {
        multiply_scalar(a, static_cast<ov::float16*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len) {
            memset(static_cast<ov::float16*>(a_dst) + len, 0, sizeof(ov::float16) * (total_size - len));
        }
    }
}
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <>
inline void attn_softmax_kernel<ov::float16>(ov::float16* a,
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
                                             const float* sink,
                                             float alibi_slope,
                                             [[maybe_unused]] uint8_t* sparse_mask,
                                             [[maybe_unused]] size_t sparse_block_size) {
    using func_fp32_type = void (*)(ov::float16*,
                                    float,
                                    const ov::float16*,
                                    const float*,
                                    const uint8_t*,
                                    bool,
                                    size_t,
                                    float,
                                    ov::float16&);
#    if !defined(OPENVINO_ARCH_ARM64)
    using func_bf16_type = void (*)(ov::float16*,
                                    float,
                                    const ov::float16*,
                                    const ov::bfloat16*,
                                    const uint8_t*,
                                    bool,
                                    size_t,
                                    float,
                                    ov::float16&);
#    endif
    using func_fp16_type = void (*)(ov::float16*,
                                    float,
                                    const ov::float16*,
                                    const ov::float16*,
                                    const uint8_t*,
                                    bool,
                                    size_t,
                                    float,
                                    ov::float16&);
    static constexpr func_fp32_type funcs_fp32[] = {scale_add2_reduce_max<false, false, false>,
                                                    scale_add2_reduce_max<false, false, true>,
                                                    scale_add2_reduce_max<false, true, false>,
                                                    scale_add2_reduce_max<false, true, true>,
                                                    scale_add2_reduce_max<true, false, false>,
                                                    scale_add2_reduce_max<true, false, true>,
                                                    scale_add2_reduce_max<true, true, false>,
                                                    scale_add2_reduce_max<true, true, true>};
#    if !defined(OPENVINO_ARCH_ARM64)
    static constexpr func_bf16_type funcs_bf16[] = {scale_add2_reduce_max<false, false, false>,
                                                    scale_add2_reduce_max<false, false, true>,
                                                    scale_add2_reduce_max<false, true, false>,
                                                    scale_add2_reduce_max<false, true, true>,
                                                    scale_add2_reduce_max<true, false, false>,
                                                    scale_add2_reduce_max<true, false, true>,
                                                    scale_add2_reduce_max<true, true, false>,
                                                    scale_add2_reduce_max<true, true, true>};
#    endif
    static constexpr func_fp16_type funcs_fp16[] = {scale_add2_reduce_max<false, false, false>,
                                                    scale_add2_reduce_max<false, false, true>,
                                                    scale_add2_reduce_max<false, true, false>,
                                                    scale_add2_reduce_max<false, true, true>,
                                                    scale_add2_reduce_max<true, false, false>,
                                                    scale_add2_reduce_max<true, false, true>,
                                                    scale_add2_reduce_max<true, true, false>,
                                                    scale_add2_reduce_max<true, true, true>};
    int dispatch = (alibi ? 0b100 : 0) | (attn_mask ? 0b010 : 0) | (causal_mask ? 0b001 : 0);
#    if defined(OPENVINO_ARCH_ARM64)
    if (detail::handle_empty_len(len, a_dst, dst_precision, total_size)) {
        return;
    }
#    endif
    ov::float16 max = std::numeric_limits<ov::float16>::lowest();
    if (attn_mask_prec == ov::element::f16) {
        funcs_fp16[dispatch](a,
                             scale,
                             alibi,
                             static_cast<const ov::float16*>(attn_mask),
                             causal_mask,
                             select_nfltmax_at_0,
                             len,
                             alibi_slope,
                             max);
#    if !defined(OPENVINO_ARCH_ARM64)
    } else if (attn_mask_prec == ov::element::bf16) {
        funcs_bf16[dispatch](a,
                             scale,
                             alibi,
                             static_cast<const ov::bfloat16*>(attn_mask),
                             causal_mask,
                             select_nfltmax_at_0,
                             len,
                             alibi_slope,
                             max);
#    endif
    } else {
        funcs_fp32[dispatch](a,
                             scale,
                             alibi,
                             static_cast<const float*>(attn_mask),
                             causal_mask,
                             select_nfltmax_at_0,
                             len,
                             alibi_slope,
                             max);
    }

    ov::float16 sum = 0.0F;
    if (sink != nullptr) {
        max = std::max(max, static_cast<const ov::float16>(*sink));
    }
#    if defined(OPENVINO_ARCH_ARM64)
    const float max_f = static_cast<float>(max);
    if (std::isinf(max_f) && max_f > 0.0F) {
        detail::handle_inf_logits(a, a_dst, dst_precision, len, total_size, sink);
        return;
    }
#    endif
    exp_reduce_sum_f32(a, max, len, sum);
    if (sink != nullptr) {
        sum += std::exp(*sink - max);
    }
    ov::float16 scalar = 1.0F / sum;
    if (dst_precision == ov::element::f32) {
        multiply_scalar(a, static_cast<float*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<float*>(a_dst) + len, 0, sizeof(float) * (total_size - len));
    } else {
        multiply_scalar_f32(a, static_cast<ov::float16*>(a_dst), scalar, len);
        // apply causual mask to final result instead of attn_score
        if (total_size > len)
            memset(static_cast<ov::float16*>(a_dst) + len, 0, sizeof(ov::float16) * (total_size - len));
    }
}
#endif

}  // namespace ov::Extensions::Cpu::XARCH