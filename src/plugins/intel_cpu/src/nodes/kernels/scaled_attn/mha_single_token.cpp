// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <cfloat>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <type_traits>

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#include "attn_quant_kernel.hpp"
#include "common.hpp"
#include "mha_single_token.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "softmax_kernel.hpp"

#if defined(OPENVINO_ARCH_ARM64)
#    include <arm_neon.h>
#endif

namespace ov::Extensions::Cpu::XARCH {

using namespace ov;

#if defined(HAVE_AVX2)

#    define prefetch_bytes(bytes, sel, advance, src) \
        {                                            \
            auto* p = reinterpret_cast<char*>(src);  \
            for (size_t i = 0; i < bytes; i += 64)   \
                _mm_prefetch(p + i + advance, sel);  \
        }

#else

#    define prefetch_bytes(bytes, sel, advance, src)

#endif

template <typename TA, typename TB>
static void cvt_copy(TA* dst, TB* src, size_t n) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
        auto vb = mm512_uni_loadu_ps(src + i);
        mm512_uni_storeu_ps(dst + i, vb);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
        auto vb = mm256_uni_loadu_ps(src + i);
        mm256_uni_storeu_ps(dst + i, vb);
    }
#endif
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}

#if defined(OPENVINO_ARCH_ARM64)
#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#        if defined(HAVE_SVE)
template <>
void cvt_copy(ov::float16* dst, ov::float16* src, size_t n) {
    size_t i = 0;
    size_t inc = vec_len_f16_sve();
    svbool_t pg = svptrue_b16();

    while (i < n) {
        if (n - i < vec_len_f16_sve()) {
            inc = n - i;
            pg = svwhilelt_b16(0, static_cast<int>(inc));
        }
        svfloat16_t b1 = svld1_f16(pg, reinterpret_cast<const float16_t*>(src + i));
        svst1_f16(pg, reinterpret_cast<float16_t*>(dst + i), b1);
        i += inc;
    }
}
#        else   // NEON
template <>
void cvt_copy(ov::float16* dst, ov::float16* src, size_t n) {
    size_t i = 0;
    for (; i + vec_len_f16_neon <= n; i += vec_len_f16_neon) {
        auto vb1 = vld1q_f16(reinterpret_cast<const float16_t*>(src + i));
        vst1q_f16(reinterpret_cast<float16_t*>(dst + i), vb1);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}
#        endif  // defined(HAVE_SVE)
#    endif      // defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)

#    if defined(HAVE_SVE)
template <>
void cvt_copy(float* dst, float* src, size_t n) {
    size_t i = 0;
    auto _dst = reinterpret_cast<float32_t*>(dst);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < n) {
        if (n - i < vec_len_f32_sve()) {
            inc = n - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        svfloat32_t b1 = svld1_f32(pg, src + i);
        svst1_f32(pg, _dst + i, b1);
        i += inc;
    }
}
#    else   // NEON
template <>
void cvt_copy(float* dst, float* src, size_t n) {
    size_t i = 0;
    for (; i + vec_len_f32_neon <= n; i += vec_len_f32_neon) {
        float32x4_t vb1 = __vld1q_f32(src + i);
        __vst1q_f32(dst + i, vb1);
    }
    for (; i < n; i++) {
        dst[i] = src[i];
    }
}
#    endif  // defined(HAVE_SVE)
#endif      // defined(OPENVINO_ARCH_ARM64)

template <typename T>
static void attn_acc_value(float* out, float weight, T* v, size_t S, float* scale, float* zp, size_t group_size) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto attn_w_vec_fp32 = _mm512_set1_ps(weight);
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto v_value = mm512_uni_loadu_ps(v + i);
        auto v_out = mm512_uni_loadu_ps(out + i);
        v_out = _mm512_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        _mm512_storeu_ps(out + i, v_out);
    }
#elif defined(HAVE_AVX2)
    auto attn_w_vec_fp32 = _mm256_set1_ps(weight);
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto v_value = mm256_uni_loadu_ps(v + i);
        auto v_out = mm256_uni_loadu_ps(out + i);
        v_out = _mm256_fmadd_ps(attn_w_vec_fp32, v_value, v_out);
        mm256_uni_storeu_ps(out + i, v_out);
    }
#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
    auto _v = reinterpret_cast<float32_t*>(v);
    svfloat32_t attn_w_vec_fp32 = svdup_n_f32(weight);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < S) {
        if (S - i < vec_len_f32_sve()) {
            inc = S - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        svfloat32_t v_value = svld1_f32(pg, _v + i);
        svfloat32_t v_out = svld1_f32(pg, out + i);

        // svmla with merging to preserve inactive lane values when there's ...
        // fewer than vec_len elements left
        v_out = svmla_f32_m(pg, v_out, attn_w_vec_fp32, v_value);
        svst1_f32(pg, out + i, v_out);
        i += inc;
    }
#    else
    float32x4_t attn_w_vec_fp32 = vdupq_n_f32(weight);
    for (; i + vec_len_f32_neon <= S; i += vec_len_f32_neon) {
        float32x4_t v_value = __vld1q_f32(v + i);
        float32x4_t v_out = __vld1q_f32(out + i);
        v_out = vmlaq_f32(v_out, attn_w_vec_fp32, v_value);
        __vst1q_f32(out + i, v_out);
    }
#    endif
#endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
template <typename T>
static void
attn_acc_value(ov::float16* out, ov::float16 weight, T* v, size_t S, float* scale, float* zp, size_t group_size) {
    size_t i = 0;
    auto _v = reinterpret_cast<float16_t*>(v);
    auto _out = reinterpret_cast<float16_t*>(out);

#    if defined(HAVE_SVE)
    svfloat16_t attn_w_vec_fp16 = svdup_n_f16(weight);
    svbool_t pg = svptrue_b16();
    size_t inc = vec_len_f16_sve();

    while (i < S) {
        if (S - i < vec_len_f16_sve()) {
            inc = S - i;
            pg = svwhilelt_b16(0, static_cast<int>(inc));
        }
        svfloat16_t v_value = svld1_f16(pg, _v + i);
        svfloat16_t v_out = svld1_f16(pg, _out + i);

        v_out = svmla_f16_m(pg, v_out, attn_w_vec_fp16, v_value);
        svst1_f16(pg, _out + i, v_out);
        i += inc;
    }
#    else
    auto attn_w_vec_fp16 = vdupq_n_f16(weight);
    for (; i + vec_len_f16_neon <= S; i += vec_len_f16_neon) {
        auto v_value = vld1q_f16(_v + i);
        auto v_out = vld1q_f16(_out + i);
        v_out = vfmaq_f16(v_out, attn_w_vec_fp16, v_value);
        vst1q_f16(_out + i, v_out);
    }
#    endif
    for (; i < S; i++) {
        out[i] += weight * v[i];
    }
}
#endif

static void attn_acc_value(float* out, float weight, uint8_t* v, size_t S, float* scale, float* zp, size_t group_size) {
    size_t group_id = 0;
    while (group_id < S / group_size) {
        size_t i = 0;
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
#if defined(HAVE_AVX512F)
        auto attn_w_vec_fp32 = _mm512_set1_ps(weight * group_scale);
        auto v_zp = _mm512_set1_ps(group_zp);
        for (; i + 4 * vec_len_f32_avx512 <= group_size; i += 4 * vec_len_f32_avx512) {
            auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i));
            auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx512));
            auto v2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx512 * 2));
            auto v3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx512 * 3));

            auto v0_out = mm512_uni_loadu_ps(out + offset + i);
            auto v1_out = mm512_uni_loadu_ps(out + offset + i + vec_len_f32_avx512);
            auto v2_out = mm512_uni_loadu_ps(out + offset + i + vec_len_f32_avx512 * 2);
            auto v3_out = mm512_uni_loadu_ps(out + offset + i + vec_len_f32_avx512 * 3);

            auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
            auto v1_256 = _mm512_cvtepu8_epi32(v1_128);
            auto v2_256 = _mm512_cvtepu8_epi32(v2_128);
            auto v3_256 = _mm512_cvtepu8_epi32(v3_128);

            auto v0_value = _mm512_cvtepi32_ps(v0_256);
            auto v1_value = _mm512_cvtepi32_ps(v1_256);
            auto v2_value = _mm512_cvtepi32_ps(v2_256);
            auto v3_value = _mm512_cvtepi32_ps(v3_256);

            v0_value = _mm512_sub_ps(v0_value, v_zp);
            v1_value = _mm512_sub_ps(v1_value, v_zp);
            v2_value = _mm512_sub_ps(v2_value, v_zp);
            v3_value = _mm512_sub_ps(v3_value, v_zp);

            v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
            v2_out = _mm512_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
            v3_out = _mm512_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 0, v0_out);
            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 1, v1_out);
            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 2, v2_out);
            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 3, v3_out);
        }
        if (i + 2 * vec_len_f32_avx512 <= group_size) {
            auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i));
            auto v1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx512));

            auto v0_out = mm512_uni_loadu_ps(out + offset + i);
            auto v1_out = mm512_uni_loadu_ps(out + offset + i + vec_len_f32_avx512);

            auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
            auto v1_256 = _mm512_cvtepu8_epi32(v1_128);

            auto v0_value = _mm512_cvtepi32_ps(v0_256);
            auto v1_value = _mm512_cvtepi32_ps(v1_256);

            v0_value = _mm512_sub_ps(v0_value, v_zp);
            v1_value = _mm512_sub_ps(v1_value, v_zp);

            v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            v1_out = _mm512_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 0, v0_out);
            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 1, v1_out);
            i += 2 * vec_len_f32_avx512;
        }
        if (i + vec_len_f32_avx512 <= group_size) {
            auto v0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(v + offset + i));
            auto v0_out = mm512_uni_loadu_ps(out + offset + i);
            auto v0_256 = _mm512_cvtepu8_epi32(v0_128);
            auto v0_value = _mm512_cvtepi32_ps(v0_256);
            v0_value = _mm512_sub_ps(v0_value, v_zp);
            v0_out = _mm512_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            mm512_uni_storeu_ps(out + offset + i + vec_len_f32_avx512 * 0, v0_out);
            i += vec_len_f32_avx512;
        }
#elif defined(HAVE_AVX2)
        auto attn_w_vec_fp32 = _mm256_set1_ps(weight * group_scale);
        auto v_zp = _mm256_set1_ps(group_zp);
        for (; i + 4 * vec_len_f32_avx2 <= group_size; i += 4 * vec_len_f32_avx2) {
            auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i));
            auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx2));
            auto v2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx2 * 2));
            auto v3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx2 * 3));

            auto v0_out = mm256_uni_loadu_ps(out + offset + i);
            auto v1_out = mm256_uni_loadu_ps(out + offset + i + vec_len_f32_avx2);
            auto v2_out = mm256_uni_loadu_ps(out + offset + i + vec_len_f32_avx2 * 2);
            auto v3_out = mm256_uni_loadu_ps(out + offset + i + vec_len_f32_avx2 * 3);

            auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
            auto v1_256 = _mm256_cvtepu8_epi32(v1_128);
            auto v2_256 = _mm256_cvtepu8_epi32(v2_128);
            auto v3_256 = _mm256_cvtepu8_epi32(v3_128);

            auto v0_value = _mm256_cvtepi32_ps(v0_256);
            auto v1_value = _mm256_cvtepi32_ps(v1_256);
            auto v2_value = _mm256_cvtepi32_ps(v2_256);
            auto v3_value = _mm256_cvtepi32_ps(v3_256);

            v0_value = _mm256_sub_ps(v0_value, v_zp);
            v1_value = _mm256_sub_ps(v1_value, v_zp);
            v2_value = _mm256_sub_ps(v2_value, v_zp);
            v3_value = _mm256_sub_ps(v3_value, v_zp);

            v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);
            v2_out = _mm256_fmadd_ps(attn_w_vec_fp32, v2_value, v2_out);
            v3_out = _mm256_fmadd_ps(attn_w_vec_fp32, v3_value, v3_out);

            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 0, v0_out);
            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 1, v1_out);
            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 2, v2_out);
            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 3, v3_out);
        }
        if (i + 2 * vec_len_f32_avx2 <= group_size) {
            auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i));
            auto v1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i + vec_len_f32_avx2));

            auto v0_out = mm256_uni_loadu_ps(out + offset + i);
            auto v1_out = mm256_uni_loadu_ps(out + offset + i + vec_len_f32_avx2);

            auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
            auto v1_256 = _mm256_cvtepu8_epi32(v1_128);

            auto v0_value = _mm256_cvtepi32_ps(v0_256);
            auto v1_value = _mm256_cvtepi32_ps(v1_256);

            v0_value = _mm256_sub_ps(v0_value, v_zp);
            v1_value = _mm256_sub_ps(v1_value, v_zp);

            v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            v1_out = _mm256_fmadd_ps(attn_w_vec_fp32, v1_value, v1_out);

            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 0, v0_out);
            mm256_uni_storeu_ps(out + offset + i + vec_len_f32_avx2 * 1, v1_out);
            i += 2 * vec_len_f32_avx2;
        }
        if (i + vec_len_f32_avx2 <= group_size) {
            auto v0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(v + offset + i));
            auto v0_out = mm256_uni_loadu_ps(out + offset + i);
            auto v0_256 = _mm256_cvtepu8_epi32(v0_128);
            auto v0_value = _mm256_cvtepi32_ps(v0_256);
            v0_value = _mm256_sub_ps(v0_value, v_zp);
            v0_out = _mm256_fmadd_ps(attn_w_vec_fp32, v0_value, v0_out);
            mm256_uni_storeu_ps(out + offset + i, v0_out);
            i += vec_len_f32_avx2;
        }
#endif
        for (; i < group_size; i++) {
            out[offset + i] += weight * (v[offset + i] - group_zp) * group_scale;
        }
        group_id += 1;
    }
}

template <typename T>
void sum_q_head(T* a, size_t n, size_t group_size, float* out) {
    size_t group_id = 0;
    while (group_id < n / group_size) {
        float group_sum = 0.0f;
        size_t offset = group_id * group_size;
        size_t i = 0;
#if defined(HAVE_AVX512F)
        auto vsum0 = _mm512_set1_ps(0.0f);
        auto vsum1 = _mm512_set1_ps(0.0f);
        auto vsum2 = _mm512_set1_ps(0.0f);
        auto vsum3 = _mm512_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512 * 4) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);
            auto va2 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 2);
            auto va3 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 3);

            vsum0 = _mm512_add_ps(va0, vsum0);
            vsum1 = _mm512_add_ps(va1, vsum1);
            vsum2 = _mm512_add_ps(va2, vsum2);
            vsum3 = _mm512_add_ps(va3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);

            vsum0 = _mm512_add_ps(va0, vsum0);
            vsum1 = _mm512_add_ps(va1, vsum1);
            i += 2 * vec_len_f32_avx512;
        }
        if (i + vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            vsum0 = _mm512_add_ps(va0, vsum0);
            i += vec_len_f32_avx512;
        }
        vsum0 = _mm512_add_ps(vsum0, vsum1);
        vsum2 = _mm512_add_ps(vsum2, vsum3);
        vsum0 = _mm512_add_ps(vsum0, vsum2);
        group_sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2 * 4) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);
            auto va2 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 2);
            auto va3 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 3);

            vsum0 = _mm256_add_ps(va0, vsum0);
            vsum1 = _mm256_add_ps(va1, vsum1);
            vsum2 = _mm256_add_ps(va2, vsum2);
            vsum3 = _mm256_add_ps(va3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);

            vsum0 = _mm256_add_ps(va0, vsum0);
            vsum1 = _mm256_add_ps(va1, vsum1);
            i += 2 * vec_len_f32_avx2;
        }
        if (i + vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            vsum0 = _mm256_add_ps(va0, vsum0);
            i += vec_len_f32_avx2;
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        hsum(vsum0);
        group_sum = _mm256_cvtss_f32(vsum0);
#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
        svfloat32_t sum0 = svdup_n_f32(0.0f);
        svfloat32_t sum1 = svdup_n_f32(0.0f);
        svfloat32_t sum2 = svdup_n_f32(0.0f);
        svfloat32_t sum3 = svdup_n_f32(0.0f);
        svbool_t pg = svptrue_b32();
        auto vec_len = vec_len_f32_sve();

        for (; i + 4 * vec_len <= group_size; i += 4 * vec_len) {
            svfloat32_t a0 = svld1_f32(pg, a + offset + i);
            svfloat32_t a1 = svld1_f32(pg, a + offset + i + vec_len);
            svfloat32_t a2 = svld1_f32(pg, a + offset + i + vec_len * 2);
            svfloat32_t a3 = svld1_f32(pg, a + offset + i + vec_len * 3);

            sum0 = svadd_f32_z(pg, a0, sum0);
            sum1 = svadd_f32_z(pg, a1, sum1);
            sum2 = svadd_f32_z(pg, a2, sum2);
            sum3 = svadd_f32_z(pg, a3, sum3);
        }
        if (i + 2 * vec_len <= group_size) {
            svfloat32_t a0 = svld1_f32(pg, a + offset + i);
            svfloat32_t a1 = svld1_f32(pg, a + offset + i + vec_len);

            sum0 = svadd_f32_z(pg, a0, sum0);
            sum1 = svadd_f32_z(pg, a1, sum1);
            i += 2 * vec_len;
        }
        if (i + vec_len <= group_size) {
            svfloat32_t a0 = svld1_f32(pg, a + offset + i);
            sum0 = svadd_f32_z(pg, a0, sum0);
            i += vec_len;
        }
        // Process tail elements parallely as well (if any)
        if (i != group_size) {
            svbool_t pg_rem = svwhilelt_b32(0, static_cast<int>(group_size - i));
            svfloat32_t a0 = svld1_f32(pg_rem, a + offset + i);
            sum0 = svadd_f32_m(pg_rem, sum0, a0);
            i = group_size;
        }
        float32_t sum_0 = svaddv_f32(pg, sum0);
        float32_t sum_1 = svaddv_f32(pg, sum1);
        float32_t sum_2 = svaddv_f32(pg, sum2);
        float32_t sum_3 = svaddv_f32(pg, sum3);
        group_sum = static_cast<float>(sum_0 + sum_1 + sum_2 + sum_3);
#    else
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        float32x4_t vsum2 = vdupq_n_f32(0.0f);
        float32x4_t vsum3 = vdupq_n_f32(0.0f);

        for (; i + 4 * vec_len_f32_neon <= group_size; i += vec_len_f32_neon * 4) {
            float32x4_t va0 = vld1q_f32(a + offset + i);
            float32x4_t va1 = vld1q_f32(a + offset + i + vec_len_f32_neon);
            float32x4_t va2 = vld1q_f32(a + offset + i + vec_len_f32_neon * 2);
            float32x4_t va3 = vld1q_f32(a + offset + i + vec_len_f32_neon * 3);

            vsum0 = vaddq_f32(va0, vsum0);
            vsum1 = vaddq_f32(va1, vsum1);
            vsum2 = vaddq_f32(va2, vsum2);
            vsum3 = vaddq_f32(va3, vsum3);
        }
        if (i + 2 * vec_len_f32_neon <= group_size) {
            float32x4_t va0 = vld1q_f32(a + offset + i);
            float32x4_t va1 = vld1q_f32(a + offset + i + vec_len_f32_neon);

            vsum0 = vaddq_f32(va0, vsum0);
            vsum1 = vaddq_f32(va1, vsum1);
            i += 2 * vec_len_f32_neon;
        }
        if (i + vec_len_f32_neon <= group_size) {
            float32x4_t va0 = vld1q_f32(a + offset + i);
            vsum0 = vaddq_f32(va0, vsum0);
            i += vec_len_f32_neon;
        }

        vsum0 = vaddq_f32(vsum0, vsum1);
        vsum2 = vaddq_f32(vsum2, vsum3);
        vsum0 = vaddq_f32(vsum0, vsum2);

        float32x2_t sum_low = vget_low_f32(vsum0);
        float32x2_t sum_high = vget_high_f32(vsum0);
        sum_low = vadd_f32(sum_low, sum_high);
        sum_low = vpadd_f32(sum_low, sum_low);
        group_sum = vget_lane_f32(sum_low, 0);
#    endif
#endif
        for (; i < group_size; i++) {
            float tmp = a[i + offset];
            group_sum += tmp;
        }
        out[group_id] = group_sum;
        group_id += 1;
    }
}

template <typename TA, typename TB>
static float dot_product(TA* a, TB* b, size_t n, float* scale, float* zp, float* head_sum, size_t group_size) {
    size_t i = 0;
    float sum = 0.0f;
#if defined(HAVE_AVX512F)
    auto vsum0 = _mm512_setzero_ps();
    auto vsum1 = _mm512_setzero_ps();
    auto vsum2 = _mm512_setzero_ps();
    auto vsum3 = _mm512_setzero_ps();
    for (; i + 4 * vec_len_f32_avx512 <= n; i += 4 * vec_len_f32_avx512) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);
        auto vb2 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 2);
        auto vb3 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512 * 3);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0 = mm512_uni_loadu_ps(b + i);
        auto vb1 = mm512_uni_loadu_ps(b + i + vec_len_f32_avx512);

        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0 = mm512_uni_loadu_ps(b + i);
        vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx512;
    }
    vsum0 = _mm512_add_ps(vsum0, vsum1);
    vsum2 = _mm512_add_ps(vsum2, vsum3);
    vsum0 = _mm512_add_ps(vsum0, vsum2);
    sum = _mm512_reduce_add_ps(vsum0);
#elif defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);
        auto vb2 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 2);
        auto vb3 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2 * 3);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0 = mm256_uni_loadu_ps(b + i);
        auto vb1 = mm256_uni_loadu_ps(b + i + vec_len_f32_avx2);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0 = mm256_uni_loadu_ps(b + i);
        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum = _mm256_cvtss_f32(vsum0);

#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
    svbool_t pg = svptrue_b32();
    svfloat32_t sum0 = svdup_n_f32(0.0f);
    svfloat32_t sum1 = svdup_n_f32(0.0f);
    svfloat32_t sum2 = svdup_n_f32(0.0f);
    svfloat32_t sum3 = svdup_n_f32(0.0f);
    auto vec_len = vec_len_f32_sve();

    auto _a = reinterpret_cast<float32_t*>(a);
    auto _b = reinterpret_cast<float32_t*>(b);

    for (; i + 4 * vec_len <= n; i += 4 * vec_len) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t a1 = svld1_f32(pg, _a + i + vec_len);
        svfloat32_t a2 = svld1_f32(pg, _a + i + vec_len * 2);
        svfloat32_t a3 = svld1_f32(pg, _a + i + vec_len * 3);

        svfloat32_t b0 = svld1_f32(pg, _b + i);
        svfloat32_t b1 = svld1_f32(pg, _b + i + vec_len);
        svfloat32_t b2 = svld1_f32(pg, _b + i + vec_len * 2);
        svfloat32_t b3 = svld1_f32(pg, _b + i + vec_len * 3);

        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        sum1 = svmla_f32_z(pg, sum1, a1, b1);
        sum2 = svmla_f32_z(pg, sum2, a2, b2);
        sum3 = svmla_f32_z(pg, sum3, a3, b3);
    }
    if (i + 2 * vec_len <= n) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t a1 = svld1_f32(pg, _a + i + vec_len);

        svfloat32_t b0 = svld1_f32(pg, _b + i);
        svfloat32_t b1 = svld1_f32(pg, _b + i + vec_len);

        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        sum1 = svmla_f32_z(pg, sum1, a1, b1);
        i += 2 * vec_len;
    }
    if (i + vec_len <= n) {
        svfloat32_t a0 = svld1_f32(pg, _a + i);
        svfloat32_t b0 = svld1_f32(pg, _b + i);
        sum0 = svmla_f32_z(pg, sum0, a0, b0);
        i += vec_len;
    }
    // Process the tail elements parallely as well (if any)
    if (i != n) {
        svbool_t pg_rem = svwhilelt_b32(0, static_cast<int>(n - i));
        svfloat32_t a0 = svld1_f32(pg_rem, _a + i);
        svfloat32_t b0 = svld1_f32(pg_rem, _b + i);
        sum0 = svmla_f32_m(pg_rem, sum0, a0, b0);
        i = n;
    }
    float32_t sum_0 = svaddv_f32(pg, sum0);
    float32_t sum_1 = svaddv_f32(pg, sum1);
    float32_t sum_2 = svaddv_f32(pg, sum2);
    float32_t sum_3 = svaddv_f32(pg, sum3);
    sum = static_cast<float>(sum_0 + sum_1 + sum_2 + sum_3);
#    else
    float32x4_t vsum0 = vdupq_n_f32(0.0f);
    float32x4_t vsum1 = vdupq_n_f32(0.0f);
    float32x4_t vsum2 = vdupq_n_f32(0.0f);
    float32x4_t vsum3 = vdupq_n_f32(0.0f);

    for (; i + 4 * vec_len_f32_neon <= n; i += vec_len_f32_neon * 4) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t va1 = __vld1q_f32(a + i + vec_len_f32_neon);
        float32x4_t va2 = __vld1q_f32(a + i + vec_len_f32_neon * 2);
        float32x4_t va3 = __vld1q_f32(a + i + vec_len_f32_neon * 3);

        float32x4_t vb0 = __vld1q_f32(b + i);
        float32x4_t vb1 = __vld1q_f32(b + i + vec_len_f32_neon);
        float32x4_t vb2 = __vld1q_f32(b + i + vec_len_f32_neon * 2);
        float32x4_t vb3 = __vld1q_f32(b + i + vec_len_f32_neon * 3);

        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        vsum1 = vmlaq_f32(vsum1, va1, vb1);
        vsum2 = vmlaq_f32(vsum2, va2, vb2);
        vsum3 = vmlaq_f32(vsum3, va3, vb3);
    }
    if (i + 2 * vec_len_f32_neon <= n) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t va1 = __vld1q_f32(a + i + vec_len_f32_neon);

        float32x4_t vb0 = __vld1q_f32(b + i);
        float32x4_t vb1 = __vld1q_f32(b + i + vec_len_f32_neon);

        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        vsum1 = vmlaq_f32(vsum1, va1, vb1);
        i += 2 * vec_len_f32_neon;
    }
    if (i + vec_len_f32_neon <= n) {
        float32x4_t va0 = __vld1q_f32(a + i);
        float32x4_t vb0 = __vld1q_f32(b + i);
        vsum0 = vmlaq_f32(vsum0, va0, vb0);
        i += vec_len_f32_neon;
    }

    vsum0 = vaddq_f32(vsum0, vsum1);
    vsum2 = vaddq_f32(vsum2, vsum3);
    vsum0 = vaddq_f32(vsum0, vsum2);

    float32x2_t temp_sum = vadd_f32(vget_low_f32(vsum0), vget_high_f32(vsum0));
    temp_sum = vpadd_f32(temp_sum, temp_sum);
    sum = vget_lane_f32(temp_sum, 0);
#    endif
#endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static ov::float16 dot_product_fp16(ov::float16* a,
                                    ov::float16* b,
                                    size_t n,
                                    float* scale,
                                    float* zp,
                                    float* head_sum) {
    size_t i = 0;
    ov::float16 sum = 0.0f;
    auto _a = reinterpret_cast<float16_t*>(a);
    auto _b = reinterpret_cast<float16_t*>(b);

#    if defined(HAVE_SVE)
    svbool_t pg = svptrue_b16();
    svfloat16_t sum0 = svdup_n_f16(0.0f);
    svfloat16_t sum1 = svdup_n_f16(0.0f);
    svfloat16_t sum2 = svdup_n_f16(0.0f);
    svfloat16_t sum3 = svdup_n_f16(0.0f);
    auto vec_len = vec_len_f16_sve();

    for (; i + 4 * vec_len <= n; i += 4 * vec_len) {
        svfloat16_t a0 = svld1_f16(pg, _a + i);
        svfloat16_t a1 = svld1_f16(pg, _a + i + vec_len);
        svfloat16_t a2 = svld1_f16(pg, _a + i + vec_len * 2);
        svfloat16_t a3 = svld1_f16(pg, _a + i + vec_len * 3);

        svfloat16_t b0 = svld1_f16(pg, _b + i);
        svfloat16_t b1 = svld1_f16(pg, _b + i + vec_len);
        svfloat16_t b2 = svld1_f16(pg, _b + i + vec_len * 2);
        svfloat16_t b3 = svld1_f16(pg, _b + i + vec_len * 3);

        sum0 = svmla_f16_z(pg, sum0, a0, b0);
        sum1 = svmla_f16_z(pg, sum1, a1, b1);
        sum2 = svmla_f16_z(pg, sum2, a2, b2);
        sum3 = svmla_f16_z(pg, sum3, a3, b3);
    }
    if (i + 2 * vec_len <= n) {
        svfloat16_t a0 = svld1_f16(pg, _a + i);
        svfloat16_t a1 = svld1_f16(pg, _a + i + vec_len);

        svfloat16_t b0 = svld1_f16(pg, _b + i);
        svfloat16_t b1 = svld1_f16(pg, _b + i + vec_len);

        sum0 = svmla_f16_z(pg, sum0, a0, b0);
        sum1 = svmla_f16_z(pg, sum1, a1, b1);
        i += 2 * vec_len;
    }
    if (i + vec_len <= n) {
        svfloat16_t a0 = svld1_f16(pg, _a + i);
        svfloat16_t b0 = svld1_f16(pg, _b + i);
        sum0 = svmla_f16_z(pg, sum0, a0, b0);
        i += vec_len;
    }
    // Process the tail elements parallely as well (if any)
    if (i != n) {
        svbool_t pg_rem = svwhilelt_b16(0, static_cast<int>(n - i));
        svfloat16_t a0 = svld1_f16(pg_rem, _a + i);
        svfloat16_t b0 = svld1_f16(pg_rem, _b + i);
        sum0 = svmla_f16_m(pg_rem, sum0, a0, b0);
        i = n;
    }
    float16_t sum_0 = svaddv_f16(pg, sum0);
    float16_t sum_1 = svaddv_f16(pg, sum1);
    float16_t sum_2 = svaddv_f16(pg, sum2);
    float16_t sum_3 = svaddv_f16(pg, sum3);
    sum = static_cast<float>(sum_0 + sum_1 + sum_2 + sum_3);
#    else
    auto vsum0 = vdupq_n_f16(0.0f);
    auto vsum1 = vdupq_n_f16(0.0f);
    auto vsum2 = vdupq_n_f16(0.0f);
    auto vsum3 = vdupq_n_f16(0.0f);

    for (; i + 4 * vec_len_f16_neon <= n; i += vec_len_f16_neon * 4) {
        auto va0 = vld1q_f16(_a + i);
        auto va1 = vld1q_f16(_a + i + vec_len_f16_neon);
        auto va2 = vld1q_f16(_a + i + vec_len_f16_neon * 2);
        auto va3 = vld1q_f16(_a + i + vec_len_f16_neon * 3);

        auto vb0 = vld1q_f16(_b + i);
        auto vb1 = vld1q_f16(_b + i + vec_len_f16_neon);
        auto vb2 = vld1q_f16(_b + i + vec_len_f16_neon * 2);
        auto vb3 = vld1q_f16(_b + i + vec_len_f16_neon * 3);

        vsum0 = vfmaq_f16(vsum0, va0, vb0);
        vsum1 = vfmaq_f16(vsum1, va1, vb1);
        vsum2 = vfmaq_f16(vsum2, va2, vb2);
        vsum3 = vfmaq_f16(vsum3, va3, vb3);
    }
    if (i + 2 * vec_len_f16_neon <= n) {
        auto va0 = vld1q_f16(_a + i);
        auto va1 = vld1q_f16(_a + i + vec_len_f16_neon);

        auto vb0 = vld1q_f16(_b + i);
        auto vb1 = vld1q_f16(_b + i + vec_len_f16_neon);

        vsum0 = vfmaq_f16(vsum0, va0, vb0);
        vsum1 = vfmaq_f16(vsum1, va1, vb1);
        i += 2 * vec_len_f16_neon;
    }
    if (i + vec_len_f16_neon <= n) {
        auto va0 = vld1q_f16(_a + i);
        auto vb0 = vld1q_f16(_b + i);
        vsum0 = vfmaq_f16(vsum0, va0, vb0);
        i += vec_len_f16_neon;
    }

    vsum0 = vaddq_f16(vsum0, vsum1);
    vsum2 = vaddq_f16(vsum2, vsum3);
    vsum0 = vaddq_f16(vsum0, vsum2);

    sum = hsum(vsum0);
#    endif
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}
#endif

template <typename TA>
static float dot_product_by_channel(TA* a, uint8_t* b, size_t n, float* scale, float* zp, size_t group_size) {
    float sum = 0.0f;
    size_t i = 0;
#if defined(HAVE_AVX512F)
    auto v512_sum0 = _mm512_set1_ps(0.0f);
    auto v512_sum1 = _mm512_set1_ps(0.0f);
    auto v512_sum2 = _mm512_set1_ps(0.0f);
    auto v512_sum3 = _mm512_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx512 <= n; i += vec_len_f32_avx512 * 4) {
        auto v0_zp = _mm512_loadu_ps(zp + i);
        auto v1_zp = _mm512_loadu_ps(zp + i + vec_len_f32_avx512);
        auto v2_zp = _mm512_loadu_ps(zp + i + vec_len_f32_avx512 * 2);
        auto v3_zp = _mm512_loadu_ps(zp + i + vec_len_f32_avx512 * 3);
        auto v0_scale = _mm512_loadu_ps(scale + i);
        auto v1_scale = _mm512_loadu_ps(scale + i + vec_len_f32_avx512);
        auto v2_scale = _mm512_loadu_ps(scale + i + vec_len_f32_avx512 * 2);
        auto v3_scale = _mm512_loadu_ps(scale + i + vec_len_f32_avx512 * 3);
        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);
        auto va2 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 2);
        auto va3 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512 * 3);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));
        auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 2));
        auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512 * 3));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);
        auto vb2 = _mm512_cvtepi32_ps(vb2_256);
        auto vb3 = _mm512_cvtepi32_ps(vb3_256);

        vb0 = _mm512_sub_ps(vb0, v0_zp);
        vb1 = _mm512_sub_ps(vb1, v1_zp);
        vb2 = _mm512_sub_ps(vb2, v2_zp);
        vb3 = _mm512_sub_ps(vb3, v3_zp);

        vb0 = _mm512_mul_ps(vb0, v0_scale);
        vb1 = _mm512_mul_ps(vb1, v1_scale);
        vb2 = _mm512_mul_ps(vb2, v2_scale);
        vb3 = _mm512_mul_ps(vb3, v3_scale);

        v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
        v512_sum1 = _mm512_fmadd_ps(va1, vb1, v512_sum1);
        v512_sum2 = _mm512_fmadd_ps(va2, vb2, v512_sum2);
        v512_sum3 = _mm512_fmadd_ps(va3, vb3, v512_sum3);
    }
    if (i + 2 * vec_len_f32_avx512 <= n) {
        auto v0_zp = _mm512_loadu_ps(zp + i);
        auto v1_zp = _mm512_loadu_ps(zp + i + vec_len_f32_avx512);
        auto v0_scale = _mm512_loadu_ps(scale + i);
        auto v1_scale = _mm512_loadu_ps(scale + i + vec_len_f32_avx512);

        auto va0 = mm512_uni_loadu_ps(a + i);
        auto va1 = mm512_uni_loadu_ps(a + i + vec_len_f32_avx512);

        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx512));

        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        auto vb1 = _mm512_cvtepi32_ps(vb1_256);

        vb0 = _mm512_sub_ps(vb0, v0_zp);
        vb1 = _mm512_sub_ps(vb1, v1_zp);

        vb0 = _mm512_mul_ps(vb0, v0_scale);
        vb1 = _mm512_mul_ps(vb1, v1_scale);

        v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
        v512_sum1 = _mm512_fmadd_ps(va1, vb1, v512_sum1);
        i += 2 * vec_len_f32_avx512;
    }
    if (i + vec_len_f32_avx512 <= n) {
        auto v0_zp = _mm512_loadu_ps(zp + i);
        auto v0_scale = _mm512_loadu_ps(scale + i);

        auto va0 = mm512_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm512_cvtepi32_ps(vb0_256);
        vb0 = _mm512_sub_ps(vb0, v0_zp);
        vb0 = _mm512_mul_ps(vb0, v0_scale);
        v512_sum0 = _mm512_fmadd_ps(va0, vb0, v512_sum0);
        i += vec_len_f32_avx512;
    }
    v512_sum0 = _mm512_add_ps(v512_sum0, v512_sum1);
    v512_sum2 = _mm512_add_ps(v512_sum2, v512_sum3);
    v512_sum0 = _mm512_add_ps(v512_sum0, v512_sum2);
    sum += _mm512_reduce_add_ps(v512_sum0);
#endif
#if defined(HAVE_AVX2)
    auto vsum0 = _mm256_set1_ps(0.0f);
    auto vsum1 = _mm256_set1_ps(0.0f);
    auto vsum2 = _mm256_set1_ps(0.0f);
    auto vsum3 = _mm256_set1_ps(0.0f);
    for (; i + 4 * vec_len_f32_avx2 <= n; i += vec_len_f32_avx2 * 4) {
        auto v0_zp = _mm256_loadu_ps(zp + i);
        auto v1_zp = _mm256_loadu_ps(zp + i + vec_len_f32_avx2);
        auto v2_zp = _mm256_loadu_ps(zp + i + vec_len_f32_avx2 * 2);
        auto v3_zp = _mm256_loadu_ps(zp + i + vec_len_f32_avx2 * 3);
        auto v0_scale = _mm256_loadu_ps(scale + i);
        auto v1_scale = _mm256_loadu_ps(scale + i + vec_len_f32_avx2);
        auto v2_scale = _mm256_loadu_ps(scale + i + vec_len_f32_avx2 * 2);
        auto v3_scale = _mm256_loadu_ps(scale + i + vec_len_f32_avx2 * 3);

        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);
        auto va2 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 2);
        auto va3 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2 * 3);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));
        auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 2));
        auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2 * 3));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
        auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
        auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);
        auto vb2 = _mm256_cvtepi32_ps(vb2_256);
        auto vb3 = _mm256_cvtepi32_ps(vb3_256);

        vb0 = _mm256_sub_ps(vb0, v0_zp);
        vb1 = _mm256_sub_ps(vb1, v1_zp);
        vb2 = _mm256_sub_ps(vb2, v2_zp);
        vb3 = _mm256_sub_ps(vb3, v3_zp);

        vb0 = _mm256_mul_ps(vb0, v0_scale);
        vb1 = _mm256_mul_ps(vb1, v1_scale);
        vb2 = _mm256_mul_ps(vb2, v2_scale);
        vb3 = _mm256_mul_ps(vb3, v3_scale);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
        vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
    }
    if (i + 2 * vec_len_f32_avx2 <= n) {
        auto v0_zp = _mm256_loadu_ps(zp + i);
        auto v1_zp = _mm256_loadu_ps(zp + i + vec_len_f32_avx2);
        auto v0_scale = _mm256_loadu_ps(scale + i);
        auto v1_scale = _mm256_loadu_ps(scale + i + vec_len_f32_avx2);

        auto va0 = mm256_uni_loadu_ps(a + i);
        auto va1 = mm256_uni_loadu_ps(a + i + vec_len_f32_avx2);

        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + vec_len_f32_avx2));

        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

        auto vb0 = _mm256_cvtepi32_ps(vb0_256);
        auto vb1 = _mm256_cvtepi32_ps(vb1_256);

        vb0 = _mm256_sub_ps(vb0, v0_zp);
        vb1 = _mm256_sub_ps(vb1, v1_zp);

        vb0 = _mm256_mul_ps(vb0, v0_scale);
        vb1 = _mm256_mul_ps(vb1, v1_scale);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
        i += 2 * vec_len_f32_avx2;
    }
    if (i + vec_len_f32_avx2 <= n) {
        auto v0_zp = _mm256_loadu_ps(zp + i);
        auto v0_scale = _mm256_loadu_ps(scale + i);

        auto va0 = mm256_uni_loadu_ps(a + i);
        auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i));
        auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
        auto vb0 = _mm256_cvtepi32_ps(vb0_256);

        vb0 = _mm256_sub_ps(vb0, v0_zp);
        vb0 = _mm256_mul_ps(vb0, v0_scale);

        vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
        i += vec_len_f32_avx2;
    }
    vsum0 = _mm256_add_ps(vsum0, vsum1);
    vsum2 = _mm256_add_ps(vsum2, vsum3);
    vsum0 = _mm256_add_ps(vsum0, vsum2);
    hsum(vsum0);
    sum += _mm256_cvtss_f32(vsum0);
#endif
    for (; i < n; i++) {
        sum += a[i] * (b[i] - zp[i]) * scale[i];
    }
    return sum;
}

template <typename TA>
static float dot_product(TA* a, uint8_t* b, size_t n, float* scale, float* zp, float* head_sum, size_t group_size) {
    float sum = 0.0f;
    size_t group_id = 0;
#if defined(HAVE_AVX512F)
    while (group_id < n / group_size) {
        auto vsum0 = _mm512_set1_ps(0.0f);
        auto vsum1 = _mm512_set1_ps(0.0f);
        auto vsum2 = _mm512_set1_ps(0.0f);
        auto vsum3 = _mm512_set1_ps(0.0f);
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        auto v_zp = _mm512_set1_ps(group_zp);
        size_t offset = group_id * group_size;
        size_t i = 0;
        for (; i + 4 * vec_len_f32_avx512 <= group_size; i += vec_len_f32_avx512 * 4) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);
            auto va2 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 2);
            auto va3 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512 * 3);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512));
            auto vb2_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512 * 2));
            auto vb3_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512 * 3));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm512_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm512_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);
            auto vb2 = _mm512_cvtepi32_ps(vb2_256);
            auto vb3 = _mm512_cvtepi32_ps(vb3_256);

            vb0 = _mm512_sub_ps(vb0, v_zp);
            vb1 = _mm512_sub_ps(vb1, v_zp);
            vb2 = _mm512_sub_ps(vb2, v_zp);
            vb3 = _mm512_sub_ps(vb3, v_zp);

            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
            vsum2 = _mm512_fmadd_ps(va2, vb2, vsum2);
            vsum3 = _mm512_fmadd_ps(va3, vb3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto va1 = mm512_uni_loadu_ps(a + offset + i + vec_len_f32_avx512);

            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx512));

            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm512_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            auto vb1 = _mm512_cvtepi32_ps(vb1_256);

            vb0 = _mm512_sub_ps(vb0, v_zp);
            vb1 = _mm512_sub_ps(vb1, v_zp);

            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va1, vb1, vsum1);
            i += 2 * vec_len_f32_avx512;
        }
        if (i + vec_len_f32_avx512 <= group_size) {
            auto va0 = mm512_uni_loadu_ps(a + offset + i);
            auto vb0_128 = _mm_loadu_si128(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb0_256 = _mm512_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm512_cvtepi32_ps(vb0_256);
            vb0 = _mm512_sub_ps(vb0, v_zp);
            vsum0 = _mm512_fmadd_ps(va0, vb0, vsum0);
            i += vec_len_f32_avx512;
        }
        vsum0 = _mm512_add_ps(vsum0, vsum1);
        vsum2 = _mm512_add_ps(vsum2, vsum3);
        vsum0 = _mm512_add_ps(vsum0, vsum2);
        float group_sum = _mm512_reduce_add_ps(vsum0);
        for (; i < group_size; i++) {
            group_sum += a[offset + i] * (b[offset + i] - group_zp);
        }
        sum += group_scale * group_sum;
        group_id += 1;
    }
    return sum;

#elif defined(HAVE_AVX2)
    while (group_id < n / group_size) {
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
        size_t i = 0;
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        for (; i + 4 * vec_len_f32_avx2 <= group_size; i += vec_len_f32_avx2 * 4) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);
            auto va2 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 2);
            auto va3 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2 * 3);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2));
            auto vb2_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2 * 2));
            auto vb3_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2 * 3));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);
            auto vb2_256 = _mm256_cvtepu8_epi32(vb2_128);
            auto vb3_256 = _mm256_cvtepu8_epi32(vb3_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);
            auto vb2 = _mm256_cvtepi32_ps(vb2_256);
            auto vb3 = _mm256_cvtepi32_ps(vb3_256);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            vsum2 = _mm256_fmadd_ps(va2, vb2, vsum2);
            vsum3 = _mm256_fmadd_ps(va3, vb3, vsum3);
        }
        if (i + 2 * vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto va1 = mm256_uni_loadu_ps(a + offset + i + vec_len_f32_avx2);

            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb1_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i + vec_len_f32_avx2));

            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb1_256 = _mm256_cvtepu8_epi32(vb1_128);

            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            auto vb1 = _mm256_cvtepi32_ps(vb1_256);

            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va1, vb1, vsum1);
            i += 2 * vec_len_f32_avx2;
        }
        if (i + vec_len_f32_avx2 <= group_size) {
            auto va0 = mm256_uni_loadu_ps(a + offset + i);
            auto vb0_128 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(b + offset + i));
            auto vb0_256 = _mm256_cvtepu8_epi32(vb0_128);
            auto vb0 = _mm256_cvtepi32_ps(vb0_256);
            vsum0 = _mm256_fmadd_ps(va0, vb0, vsum0);
            i += vec_len_f32_avx2;
        }
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        vsum2 = _mm256_add_ps(vsum2, vsum3);
        vsum0 = _mm256_add_ps(vsum0, vsum2);
        hsum(vsum0);
        float group_sum = _mm256_cvtss_f32(vsum0);
        for (; i < group_size; i++) {
            group_sum += a[offset + i] * b[offset + i];
        }
        // B = scale * (b - zero)
        // Σ (A * B) = Σ (a * scale * (b - zero)) = scale * (Σ a * b - zero Σ a) = scale * (sum - zp * head_sum)
        group_sum = group_scale * (group_sum - group_zp * head_sum[group_id]);
        sum += group_sum;
        group_id += 1;
    }
    return sum;
#else
    while (group_id < n / group_size) {
        size_t i = 0;
        float group_scale = *(scale + group_id * 2);
        float group_zp = *(zp + group_id * 2);
        size_t offset = group_id * group_size;
        float group_sum = 0.0f;
        for (; i < group_size; i++) {
            group_sum += a[i + offset] * (b[i + offset] - group_zp);
        }
        sum += group_scale * group_sum;
        group_id += 1;
    }
    return sum;
#endif
}

template <typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm512_setzero_ps();
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = _mm512_loadu_ps(src);
            result_vec_fp32 = _mm512_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        // save to bf16
        mm512_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(HAVE_AVX2)
    for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
        auto* src = temp + i;
        auto result_vec_fp32 = _mm256_set1_ps(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = mm256_uni_loadu_ps(src);
            result_vec_fp32 = _mm256_add_ps(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        mm256_uni_storeu_ps(dst + i, result_vec_fp32);
    }
#elif defined(OPENVINO_ARCH_ARM64)
#    if defined(HAVE_SVE)
    auto _dst = reinterpret_cast<float32_t*>(dst);
    size_t inc = vec_len_f32_sve();
    svbool_t pg = svptrue_b32();

    while (i < S) {
        if (S - i < vec_len_f32_sve()) {
            inc = S - i;
            pg = svwhilelt_b32(0, static_cast<int>(inc));
        }
        auto* src = temp + i;
        auto result_vec_fp32 = svdup_n_f32(0.0f);

        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = svld1_f32(pg, src);
            result_vec_fp32 = svadd_f32_m(pg, result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        svst1_f32(pg, _dst + i, result_vec_fp32);
        i += inc;
    }
#    else
    for (; i + vec_len_f32_neon <= S; i += vec_len_f32_neon) {
        auto* src = temp + i;
        auto result_vec_fp32 = vdupq_n_f32(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp32 = __vld1q_f32(src);
            result_vec_fp32 = vaddq_f32(result_vec_fp32, o_vec_fp32);
            src += temp_stride;
        }
        __vst1q_f32(dst + i, result_vec_fp32);
    }
#    endif
#endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
static void attn_reduce(ov::float16* dst, ov::float16* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#    if defined(HAVE_SVE)
    svbool_t pg = svptrue_b16();

    for (; i + vec_len_f16_sve() <= S; i += vec_len_f16_sve()) {
        auto* src = temp + i;
        auto result_vec_fp16 = svdup_n_f16(0.0f);

        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp16 = svld1_f16(pg, reinterpret_cast<float16_t*>(src));
            result_vec_fp16 = svadd_f16_m(pg, result_vec_fp16, o_vec_fp16);
            src += temp_stride;
        }
        svst1_f16(pg, reinterpret_cast<float16_t*>(dst + i), result_vec_fp16);
    }
#    else
    for (; i + vec_len_f16_neon <= S; i += vec_len_f16_neon) {
        auto* src = temp + i;
        auto result_vec_fp16 = vdupq_n_f16(0.0f);
        for (size_t m = 0; m < M; m++) {
            auto o_vec_fp16 = vld1q_f16(reinterpret_cast<float16_t*>(src));
            result_vec_fp16 = vaddq_f16(result_vec_fp16, o_vec_fp16);
            src += temp_stride;
        }
        vst1q_f16(reinterpret_cast<float16_t*>(dst + i), result_vec_fp16);
    }
#    endif
    for (; i < S; i++) {
        auto* src = temp + i;
        float sum = 0.0f;
        // sum result from all threads partition
        for (size_t m = 0; m < M; m++) {
            sum += src[0];
            src += temp_stride;
        }
        dst[i] = sum;
    }
}
#endif

template <typename T, typename T2, typename T3>
static void mha_single_token_kernel(const ov::intel_cpu::PlainTensor& query,
                                    const ov::intel_cpu::PlainTensor& present_key,
                                    const ov::intel_cpu::PlainTensor& present_value,
                                    const ov::intel_cpu::PlainTensor& alibi_mask,
                                    const ov::intel_cpu::PlainTensor& attention_mask,
                                    const ov::intel_cpu::PlainTensor& beams,
                                    ov::intel_cpu::PlainTensor& output_emb,
                                    ov::intel_cpu::PlainTensor& buf_attn_w,
                                    ov::intel_cpu::PlainTensor& buf_attn_score,
                                    bool has_out_transpose,
                                    bool auto_causal,
                                    float d_scale,
                                    const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                                    const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                                    ov::intel_cpu::PlainTensor& head_sum,
                                    size_t key_group_size,
                                    size_t value_group_size,
                                    bool quant_key_by_channel) {
    ov::intel_cpu::PlainTensor causal_mask;
    bool select_nfltmax_at_0 = false;
    auto B = query.size(0);
    auto H = query.size(1);
    auto q_len = query.size(2);
    auto S = query.size(3);
    auto SV = present_value.size(3);
    auto h_group_num = present_value.size(1);
    auto precision = ov::element::f32;
    if (std::is_same_v<T3, ov::float16>) {
        precision = ov::element::f16;
    }
    size_t h_each_group_len = 1;
    if (h_group_num != H) {
        h_each_group_len = H / h_group_num;
    }
    if (d_scale == 0.0f) {
        d_scale = 1.0f / sqrt(S);
    }
    auto nthr = parallel_get_max_threads();
    auto kv_len = present_key.size(2);
    bool pastkv_is_int8 = past_k_scale_zp;
#if defined(HAVE_AVX2) && !defined(HAVE_AVX512F)
    // avx2 will pre-compute the zero point and try to save the sub instruction in the dot_product,
    //  but it seems not necessary for avx512. Possible reason may be that for avx2 the cost of dot_product
    //  is larger than the memory access time, but for avx512 is not and the cost of pre-compute is a pure increase.
    if (pastkv_is_int8 && !quant_key_by_channel) {
        // be sure no false sharing
        size_t group_num = S / key_group_size;
        head_sum.resize<float>({B, H, q_len, group_num + 16});
        parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
            sum_q_head(query.ptr<T>(b, h, pq), S, key_group_size, head_sum.ptr<float>(b, h, pq));
        });
    }
#endif

    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

        size_t b, h_group, pk;
        if (start < end) {
            parallel_it_init(start, pk, kv_len, b, B, h_group, h_group_num);
            if (q_len == 1 && h_each_group_len == 1) {
                if (B == 1) {
                    // the memory will be continuous when b==1
                    for (size_t iwork = start; iwork < end; ++iwork) {
                        auto p = past_k_scale_zp.ptr<float>(pk, 0, h_group);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        if (std::is_same<T3, ov::float16>::value && std::is_same<T, ov::float16>::value &&
                            std::is_same<T2, ov::float16>::value) {
                            auto p_k = present_key.ptr<ov::float16>(0, h_group, pk);
                            prefetch_bytes(S, _MM_HINT_T0, 4096, p_k);
                            auto _qk = dot_product_fp16(query.ptr<ov::float16>(0, h_group),
                                                        p_k,
                                                        S,
                                                        p,
                                                        p + 1,
                                                        head_sum.ptr<float>(0, h_group));
                            buf_attn_w.ptr<T3>(0, h_group, 0)[pk] = _qk;
                            parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                            continue;
                        }
#endif
                        if (quant_key_by_channel && pastkv_is_int8) {
                            auto p_scale = past_k_scale_zp.ptr<float>(pk / key_group_size * 2, 0, h_group);
                            auto p_zp = past_k_scale_zp.ptr<float>(pk / key_group_size * 2 + 1, 0, h_group);
                            auto p_k = present_key.ptr<uint8_t>(0, h_group, pk);
                            prefetch_bytes(S, _MM_HINT_T0, 4096, p_k);
                            buf_attn_w.ptr<T3>(0, h_group, 0)[pk] =
                                dot_product_by_channel(query.ptr<T>(0, h_group), p_k, S, p_scale, p_zp, key_group_size);
                        } else {
                            auto p_k = present_key.ptr<T2>(0, h_group, pk);
                            prefetch_bytes(S, _MM_HINT_T0, 4096, p_k);
                            buf_attn_w.ptr<T3>(0, h_group, 0)[pk] = dot_product(query.ptr<T>(0, h_group),
                                                                                p_k,
                                                                                S,
                                                                                p,
                                                                                p + 1,
                                                                                head_sum.ptr<float>(0, h_group),
                                                                                key_group_size);
                        }
                        parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                    }
                } else {
                    for (size_t iwork = start; iwork < end; ++iwork) {
                        auto b_kv = beams ? beams.ptr<int32_t>(b)[pk] : b;
                        auto p = past_k_scale_zp.ptr<float>(pk, b_kv, h_group);
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                        if (std::is_same<T3, ov::float16>::value && std::is_same<T, ov::float16>::value &&
                            std::is_same<T2, ov::float16>::value) {
                            auto p_k = present_key.ptr<ov::float16>(b_kv, h_group, pk);
                            auto _qk = dot_product_fp16(query.ptr<ov::float16>(b, h_group),
                                                        p_k,
                                                        S,
                                                        p,
                                                        p + 1,
                                                        head_sum.ptr<float>(b, h_group));
                            buf_attn_w.ptr<T3>(b, h_group, 0)[pk] = _qk;
                            parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                            continue;
                        }
#endif
                        if (quant_key_by_channel && pastkv_is_int8) {
                            auto p_scale = past_k_scale_zp.ptr<float>(pk / key_group_size * 2, b_kv, h_group);
                            auto p_zp = past_k_scale_zp.ptr<float>(pk / key_group_size * 2 + 1, b_kv, h_group);
                            auto p_k = present_key.ptr<uint8_t>(b_kv, h_group, pk);
                            buf_attn_w.ptr<T3>(b, h_group, 0)[pk] =
                                dot_product_by_channel(query.ptr<T>(b, h_group), p_k, S, p_scale, p_zp, key_group_size);
                        } else {
                            auto p_k = present_key.ptr<T2>(b_kv, h_group, pk);
                            buf_attn_w.ptr<T3>(b, h_group, 0)[pk] = dot_product(query.ptr<T>(b, h_group),
                                                                                p_k,
                                                                                S,
                                                                                p,
                                                                                p + 1,
                                                                                head_sum.ptr<float>(b, h_group),
                                                                                key_group_size);
                        }
                        parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                    }
                }
            } else {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.ptr<int32_t>(b)[pk] : b;
                    for (size_t pq = 0; pq < q_len; pq++) {
                        auto p = past_k_scale_zp.ptr<float>(pk, b_kv, h_group);
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
                            if (std::is_same<T3, ov::float16>::value && std::is_same<T, ov::float16>::value &&
                                std::is_same<T2, ov::float16>::value) {
                                auto p_k = present_key.ptr<ov::float16>(b_kv, h_group, pk);
                                auto _qk = dot_product_fp16(query.ptr<ov::float16>(b, h, pq),
                                                            p_k,
                                                            S,
                                                            p,
                                                            p + 1,
                                                            head_sum.ptr<float>(b, h, pq));
                                buf_attn_w.ptr<T3>(b, h, pq)[pk] = _qk;
                                continue;
                            }
#endif
                            if (quant_key_by_channel && pastkv_is_int8) {
                                auto p_scale = past_k_scale_zp.ptr<float>(pk / key_group_size * 2, b_kv, h_group);
                                auto p_zp = past_k_scale_zp.ptr<float>(pk / key_group_size * 2 + 1, b_kv, h_group);
                                auto p_k = present_key.ptr<uint8_t>(b_kv, h_group, pk);
                                buf_attn_w.ptr<T3>(b, h, pq)[pk] = dot_product_by_channel(query.ptr<T>(b, h, pq),
                                                                                          p_k,
                                                                                          S,
                                                                                          p_scale,
                                                                                          p_zp,
                                                                                          key_group_size);
                            } else {
                                buf_attn_w.ptr<T3>(b, h, pq)[pk] = dot_product(query.ptr<T>(b, h, pq),
                                                                               present_key.ptr<T2>(b_kv, h_group, pk),
                                                                               S,
                                                                               p,
                                                                               p + 1,
                                                                               head_sum.ptr<float>(b, h, pq),
                                                                               key_group_size);
                            }
                        }
                    }
                    parallel_it_step(pk, kv_len, b, B, h_group, h_group_num);
                }
            }
        }
    });

    parallel_for3d(B, H, q_len, [&](size_t b, size_t h, size_t pq) {
        auto cur_kv_len = kv_len;
        auto ncausal = auto_causal ? (cur_kv_len - q_len + pq + 1) : cur_kv_len;
        // apply attention mask & sofmax
        T3* alibi_ptr = alibi_mask ? &alibi_mask.at<T3>({b, h, pq, 0}, true) : nullptr;
        uint8_t* attn_mask_ptr = nullptr;
        auto attn_mask_prec = attention_mask.get_precision();
        if (attention_mask) {
            attn_mask_ptr = reinterpret_cast<uint8_t*>(&attention_mask.at<T>({b, h, pq, 0}, true));
        }
        uint8_t* cmask_ptr = causal_mask ? &causal_mask.at<uint8_t>({b, h, pq, 0}, true) : nullptr;
        attn_softmax_kernel<T3>(buf_attn_w.ptr<T3>(b, h, pq),
                                buf_attn_w.ptr<T3>(b, h, pq),
                                d_scale,
                                alibi_ptr,
                                attn_mask_ptr,
                                cmask_ptr,
                                select_nfltmax_at_0,
                                ncausal,
                                cur_kv_len,
                                attn_mask_prec,
                                precision);
    });

    // attn_w * V
    // Fast Path if there are enough works for each thread
    if (B >= static_cast<size_t>(nthr)) {
        buf_attn_score.resize<T3>({static_cast<size_t>(nthr), q_len, h_each_group_len, SV});
        parallel_for2d(B, h_group_num, [&](size_t b, size_t h_group) {
            auto ithr = parallel_get_thread_num();
            memset(buf_attn_score.ptr<T3>(ithr), 0, q_len * h_each_group_len * SV * sizeof(T3));
            for (size_t pv = 0; pv < kv_len; pv++) {
                auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len;
                         h++, group_idx++) {
                        attn_acc_value(buf_attn_score.ptr<T3>(ithr, pq, group_idx),
                                       buf_attn_w.ptr<T3>(b, h, pq)[pv],
                                       v,
                                       SV,
                                       p + 0,
                                       p + 1,
                                       value_group_size);
                    }
                }
            }
            // convert to dst
            for (size_t pq = 0; pq < q_len; pq++) {
                for (size_t h = h_group * h_each_group_len, group_idx = 0; h < (h_group + 1) * h_each_group_len;
                     h++, group_idx++) {
                    auto* dst = has_out_transpose ? output_emb.ptr<T>(b, pq, h * SV) : output_emb.ptr<T>(b, h, pq);
                    cvt_copy(dst, buf_attn_score.ptr<T3>(ithr, pq, group_idx), SV);
                }
            }
        });
        return;
    }

    buf_attn_score.resize<T3>({static_cast<size_t>(nthr), B, q_len, H, SV});
    // buf_attn_w {B, H, q_len, kv_len}
    parallel_nt_static(nthr, [&](const size_t ithr, const size_t nthr) {
        size_t start{0}, end{0};
        splitter(B * h_group_num * kv_len, nthr, ithr, start, end);

        memset(buf_attn_score.ptr<T3>(ithr, 0, 0, 0, 0), 0, buf_attn_score.stride(0) * sizeof(T3));

        size_t b, h_group, pv;
        if (start < end) {
            parallel_it_init(start, pv, kv_len, b, B, h_group, h_group_num);
            if (q_len == 1 && h_each_group_len == 1) {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                    auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                    auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                    attn_acc_value(buf_attn_score.ptr<T3>(ithr, b, 0, h_group),
                                   buf_attn_w.ptr<T3>(b, h_group, 0, pv)[0],
                                   v,
                                   SV,
                                   p + 0,
                                   p + 1,
                                   value_group_size);
                    parallel_it_step(pv, kv_len, b, B, h_group, h_group_num);
                }
            } else {
                for (size_t iwork = start; iwork < end; ++iwork) {
                    auto b_kv = beams ? beams.ptr<int32_t>(b)[pv] : b;
                    auto* v = present_value.ptr<T2>(b_kv, h_group, pv);
                    auto p = past_v_scale_zp.ptr<float>(pv, b_kv, h_group);
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = h_group * h_each_group_len; h < (h_group + 1) * h_each_group_len; h++) {
                            attn_acc_value(buf_attn_score.ptr<T3>(ithr, b, pq, h),
                                           buf_attn_w.ptr<T3>(b, h, pq)[pv],
                                           v,
                                           SV,
                                           p + 0,
                                           p + 1,
                                           value_group_size);
                        }
                    }
                    parallel_it_step(pv, kv_len, b, B, h_group, h_group_num);
                }
            }
        }
    });

    auto bhl_loop = [&](size_t b, size_t h, size_t pq) {
        auto* temp = buf_attn_score.ptr<T3>(0, b, pq, h);
        size_t temp_stride = buf_attn_score.stride(0);
        auto* dst = has_out_transpose ? output_emb.ptr<T>(b, pq, h * SV) : output_emb.ptr<T>(b, h, pq);
        attn_reduce(dst, temp, nthr, SV, temp_stride);
    };

    parallel_nt_static(nthr, [&](const int ithr, const int nthr) {
        for_3d(ithr, nthr, B, H, q_len, bhl_loop);
    });
}

void mha_single_token(const ov::intel_cpu::PlainTensor& query,
                      const ov::intel_cpu::PlainTensor& present_key,
                      const ov::intel_cpu::PlainTensor& present_value,
                      const ov::intel_cpu::PlainTensor& alibi_mask,
                      const ov::intel_cpu::PlainTensor& attention_mask,
                      const ov::intel_cpu::PlainTensor& beams,
                      ov::intel_cpu::PlainTensor& output_emb,
                      ov::intel_cpu::PlainTensor& buf_attn_w,
                      ov::intel_cpu::PlainTensor& buf_attn_score,
                      bool has_out_transpose,
                      bool auto_causal,
                      float d_scale,
                      const ov::intel_cpu::PlainTensor& past_k_scale_zp,
                      const ov::intel_cpu::PlainTensor& past_v_scale_zp,
                      ov::intel_cpu::PlainTensor& head_sum,
                      size_t key_group_size,
                      size_t value_group_size,
                      bool quant_key_by_channel) {
    if (query.get_precision() == ov::element::bf16) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<ov::bfloat16, uint8_t, float>(query,
                                                                  present_key,
                                                                  present_value,
                                                                  alibi_mask,
                                                                  attention_mask,
                                                                  beams,
                                                                  output_emb,
                                                                  buf_attn_w,
                                                                  buf_attn_score,
                                                                  has_out_transpose,
                                                                  auto_causal,
                                                                  d_scale,
                                                                  past_k_scale_zp,
                                                                  past_v_scale_zp,
                                                                  head_sum,
                                                                  key_group_size,
                                                                  value_group_size,
                                                                  quant_key_by_channel);
        } else {
            mha_single_token_kernel<ov::bfloat16, ov::bfloat16, float>(query,
                                                                       present_key,
                                                                       present_value,
                                                                       alibi_mask,
                                                                       attention_mask,
                                                                       beams,
                                                                       output_emb,
                                                                       buf_attn_w,
                                                                       buf_attn_score,
                                                                       has_out_transpose,
                                                                       auto_causal,
                                                                       d_scale,
                                                                       past_k_scale_zp,
                                                                       past_v_scale_zp,
                                                                       head_sum,
                                                                       key_group_size,
                                                                       value_group_size,
                                                                       quant_key_by_channel);
        }
    } else if (query.get_precision() == ov::element::f16) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
        if (present_key.get_precision() == ov::element::f16) {
            mha_single_token_kernel<ov::float16, ov::float16, ov::float16>(query,
                                                                           present_key,
                                                                           present_value,
                                                                           alibi_mask,
                                                                           attention_mask,
                                                                           beams,
                                                                           output_emb,
                                                                           buf_attn_w,
                                                                           buf_attn_score,
                                                                           has_out_transpose,
                                                                           auto_causal,
                                                                           d_scale,
                                                                           past_k_scale_zp,
                                                                           past_v_scale_zp,
                                                                           head_sum,
                                                                           key_group_size,
                                                                           value_group_size,
                                                                           quant_key_by_channel);
        } else {
            OPENVINO_THROW("Unsupported precision: ", present_key.get_precision());
        }
#else
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<ov::float16, uint8_t, float>(query,
                                                                 present_key,
                                                                 present_value,
                                                                 alibi_mask,
                                                                 attention_mask,
                                                                 beams,
                                                                 output_emb,
                                                                 buf_attn_w,
                                                                 buf_attn_score,
                                                                 has_out_transpose,
                                                                 auto_causal,
                                                                 d_scale,
                                                                 past_k_scale_zp,
                                                                 past_v_scale_zp,
                                                                 head_sum,
                                                                 key_group_size,
                                                                 value_group_size,
                                                                 quant_key_by_channel);
        } else {
            mha_single_token_kernel<ov::float16, ov::float16, float>(query,
                                                                     present_key,
                                                                     present_value,
                                                                     alibi_mask,
                                                                     attention_mask,
                                                                     beams,
                                                                     output_emb,
                                                                     buf_attn_w,
                                                                     buf_attn_score,
                                                                     has_out_transpose,
                                                                     auto_causal,
                                                                     d_scale,
                                                                     past_k_scale_zp,
                                                                     past_v_scale_zp,
                                                                     head_sum,
                                                                     key_group_size,
                                                                     value_group_size,
                                                                     quant_key_by_channel);
        }
#endif
    } else if (query.get_precision() == ov::element::f32) {
        if (present_key.get_precision() == ov::element::u8) {
            mha_single_token_kernel<float, uint8_t, float>(query,
                                                           present_key,
                                                           present_value,
                                                           alibi_mask,
                                                           attention_mask,
                                                           beams,
                                                           output_emb,
                                                           buf_attn_w,
                                                           buf_attn_score,
                                                           has_out_transpose,
                                                           auto_causal,
                                                           d_scale,
                                                           past_k_scale_zp,
                                                           past_v_scale_zp,
                                                           head_sum,
                                                           key_group_size,
                                                           value_group_size,
                                                           quant_key_by_channel);
        } else if (present_key.get_precision() == ov::element::f16) {
            mha_single_token_kernel<float, ov::float16, float>(query,
                                                               present_key,
                                                               present_value,
                                                               alibi_mask,
                                                               attention_mask,
                                                               beams,
                                                               output_emb,
                                                               buf_attn_w,
                                                               buf_attn_score,
                                                               has_out_transpose,
                                                               auto_causal,
                                                               d_scale,
                                                               past_k_scale_zp,
                                                               past_v_scale_zp,
                                                               head_sum,
                                                               key_group_size,
                                                               value_group_size,
                                                               quant_key_by_channel);
        } else {
            mha_single_token_kernel<float, float, float>(query,
                                                         present_key,
                                                         present_value,
                                                         alibi_mask,
                                                         attention_mask,
                                                         beams,
                                                         output_emb,
                                                         buf_attn_w,
                                                         buf_attn_score,
                                                         has_out_transpose,
                                                         auto_causal,
                                                         d_scale,
                                                         past_k_scale_zp,
                                                         past_v_scale_zp,
                                                         head_sum,
                                                         key_group_size,
                                                         value_group_size,
                                                         quant_key_by_channel);
        }
    } else {
        OPENVINO_THROW("Unsupported precision: ", query.get_precision());
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
