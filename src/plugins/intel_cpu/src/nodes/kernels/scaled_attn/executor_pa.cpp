// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/core/type/float16.hpp"
#include "openvino/core/parallel.hpp"
#include "executor_pa.hpp"
#include "executor_pa_common.hpp"
#include "common.hpp"
#include "attn_quant_kernel.hpp"
#include "softmax_kernel.hpp"
#include "transpose_kernel.hpp"
#include "utils/plain_tensor.hpp"
#include "attn_memcpy.hpp"
#include "attn_quant.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;
using namespace ov::intel_cpu;

// currently depends on brgemm which only support x64
#ifdef OPENVINO_ARCH_X86_64

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)

#define prefetch_bytes(bytes, sel, advance, src) {  \
    auto *p = reinterpret_cast<char *>(src);        \
    for (size_t i = 0; i < bytes; i += 64)          \
        _mm_prefetch(p + i + advance, sel);         \
}

#else

#define prefetch_bytes(bytes, sel, advance, src)

#endif

template<typename TA, typename TB>
void cvt_copy(TA* dst, TB* src, size_t n) {
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

template<typename T>
static void attn_acc_value_block(float* out, float* weight, T* v, size_t S, size_t block_size) {
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm512_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm512_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec2, mm512_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec3, mm512_uni_loadu_ps(v + i + S * 3), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, mm512_uni_loadu_ps(v + i + S), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm512_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            v_out = _mm512_fmadd_ps(attn_w_vec0, mm512_uni_loadu_ps(v + i), v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        auto attn_w_vec2 = _mm256_set1_ps(weight[2]);
        auto attn_w_vec3 = _mm256_set1_ps(weight[3]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec2, mm256_uni_loadu_ps(v + i + S * 2), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec3, mm256_uni_loadu_ps(v + i + S * 3), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
            out[i] += weight[2] * v[i + S * 2];
            out[i] += weight[3] * v[i + S * 3];
        }
        v += 4 * S;
        weight += 4;
    }
    if (j + 2 <= block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        auto attn_w_vec1 = _mm256_set1_ps(weight[1]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);
            v_out = _mm256_fmadd_ps(attn_w_vec1, mm256_uni_loadu_ps(v + i + S), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
            out[i] += weight[1] * v[i + S];
        }
        v += 2 * S;
        weight += 2;
        j += 2;
    }
    if (j < block_size) {
        auto attn_w_vec0 = _mm256_set1_ps(weight[0]);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            v_out = _mm256_fmadd_ps(attn_w_vec0, mm256_uni_loadu_ps(v + i), v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * v[i];
        }
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * v[i];
        }
        v += S;
    }
}

static void attn_acc_value_block(float* out, float* weight, uint8_t* v, size_t S, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto v_f1 = reinterpret_cast<float*>(v + S + 8);
        auto v_f2 = reinterpret_cast<float*>(v + 2 * (S + 8));
        auto v_f3 = reinterpret_cast<float*>(v + 3 * (S + 8));
        auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
        auto attn_w_vec1 = _mm512_set1_ps(weight[1] * v_f1[0]);
        auto attn_w_vec2 = _mm512_set1_ps(weight[2] * v_f2[0]);
        auto attn_w_vec3 = _mm512_set1_ps(weight[3] * v_f3[0]);
        auto zp0 = _mm512_set1_ps(v_f0[1]);
        auto zp1 = _mm512_set1_ps(v_f1[1]);
        auto zp2 = _mm512_set1_ps(v_f2[1]);
        auto zp3 = _mm512_set1_ps(v_f3[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i)))), zp0);
            auto v1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + S + 8)))), zp1);
            auto v2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + 2 * (S + 8))))), zp2);
            auto v3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i + 3 * (S + 8))))), zp3);
            v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec1, v1, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec2, v2, v_out);
            v_out = _mm512_fmadd_ps(attn_w_vec3, v3, v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
            out[i] += weight[1] * (v[i + S + 8] - v_f1[1]) * v_f1[0];
            out[i] += weight[2] * (v[i + 2 * (S + 8)] - v_f2[1]) * v_f2[0];
            out[i] += weight[3] * (v[i + 3 * (S + 8)] - v_f3[1]) * v_f3[0];
        }
        v += 4 * (S + 8) - 8;
        weight += 4;
    }
    for (; j < block_size; j++) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto attn_w_vec0 = _mm512_set1_ps(weight[0] * v_f0[0]);
        auto zp0 = _mm512_set1_ps(v_f0[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx512 <= S; i += vec_len_f32_avx512) {
            auto v_out = mm512_uni_loadu_ps(out + i);
            auto v0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(v + i)))), zp0);
            v_out = _mm512_fmadd_ps(attn_w_vec0, v0, v_out);

            _mm512_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
        }
        v += S;
        weight++;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j < block_size; j++) {
        auto v_f0 = reinterpret_cast<float*>(v);
        auto attn_w_vec0 = _mm256_set1_ps(weight[0] * v_f0[0]);
        auto zp0 = _mm256_set1_ps(v_f0[1]);
        size_t i = 0;
        v += 8;
        for (; i + vec_len_f32_avx2 <= S; i += vec_len_f32_avx2) {
            auto v_out = mm256_uni_loadu_ps(out + i);
            auto v0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(v + i)))), zp0);
            v_out = _mm256_fmadd_ps(attn_w_vec0, v0, v_out);

            mm256_uni_storeu_ps(out + i, v_out);
        }
        for (; i < S; i++) {
            out[i] += weight[0] * (v[i] - v_f0[1]) * v_f0[0];
        }
        v += S;
        weight++;
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        auto v0 = reinterpret_cast<float*>(v);
        v += 8;
        for (size_t i = 0; i < S; i++) {
            out[i] += weight[j] * (v[i] - v0[1]) * v0[0];
        }
        v += S;
    }
}

template<typename TA, typename TB>
static void dot_product_block(TA* a, TB* b, float* c, size_t n, size_t block_size) {
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum0 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i + 3 * n), vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b +=  4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        size_t i = 0;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            vsum = _mm512_fmadd_ps(va, mm512_uni_loadu_ps(b + i), vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_set1_ps(0.0f);
        auto vsum1 = _mm256_set1_ps(0.0f);
        auto vsum2 = _mm256_set1_ps(0.0f);
        auto vsum3 = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum0 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum0);
            vsum1 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + n), vsum1);
            vsum2 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 2 * n), vsum2);
            vsum3 = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i + 3 * n), vsum3);
        }
        hsum(vsum0);
        hsum(vsum1);
        hsum(vsum2);
        hsum(vsum3);
        float sum0 = _mm256_cvtss_f32(vsum0);
        float sum1 = _mm256_cvtss_f32(vsum1);
        float sum2 = _mm256_cvtss_f32(vsum2);
        float sum3 = _mm256_cvtss_f32(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * b[i];
            sum1 += a[i] * b[i + n];
            sum2 += a[i] * b[i + 2 * n];
            sum3 += a[i] * b[i + 3 * n];
        }
        c[0] = sum0;
        c[1] = sum1;
        c[2] = sum2;
        c[3] = sum3;
        c += 4;
        b +=  4 * n;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_set1_ps(0.0f);
        size_t i = 0;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            vsum = _mm256_fmadd_ps(va, mm256_uni_loadu_ps(b + i), vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * b[i];
        }
        b += n;
        *c++ = sum;
    }
}

template<typename TA>
static void dot_product_block(TA* a, uint8_t* b, float* c, size_t n, size_t block_size) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
#if defined(HAVE_AVX512F)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm512_setzero_ps();
        auto vsum1 = _mm512_setzero_ps();
        auto vsum2 = _mm512_setzero_ps();
        auto vsum3 = _mm512_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto b1 = reinterpret_cast<float*>(b + n + 8);
        auto b2 = reinterpret_cast<float*>(b + (n + 8) * 2);
        auto b3 = reinterpret_cast<float*>(b + (n + 8) * 3);
        auto v_zp0 = _mm512_set1_ps(b0[1]);
        auto v_zp1 = _mm512_set1_ps(b1[1]);
        auto v_zp2 = _mm512_set1_ps(b2[1]);
        auto v_zp3 = _mm512_set1_ps(b3[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb0 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp0);
            auto vb1 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + n + 8)))), v_zp1);
            auto vb2 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 2 * (n + 8))))), v_zp2);
            auto vb3 = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i + 3 * (n + 8))))), v_zp3);

            vsum0 = _mm512_fmadd_ps(va, vb0, vsum0);
            vsum1 = _mm512_fmadd_ps(va, vb1, vsum1);
            vsum2 = _mm512_fmadd_ps(va, vb2, vsum2);
            vsum3 = _mm512_fmadd_ps(va, vb3, vsum3);
        }
        float sum0 = _mm512_reduce_add_ps(vsum0);
        float sum1 = _mm512_reduce_add_ps(vsum1);
        float sum2 = _mm512_reduce_add_ps(vsum2);
        float sum3 = _mm512_reduce_add_ps(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * (b[i] - b0[1]);
            sum1 += a[i] * (b[i + n + 8] - b1[1]);
            sum2 += a[i] * (b[i + 2 * (n + 8)] - b2[1]);
            sum3 += a[i] * (b[i + 3 * (n + 8)] - b3[1]);
        }
        c[0] = sum0 * b0[0];
        c[1] = sum1 * b1[0];
        c[2] = sum2 * b2[0];
        c[3] = sum3 * b3[0];
        c += 4;
        b +=  4 * (n + 8) - 8;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm512_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = _mm512_set1_ps(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx512 <= n; i += vec_len_f32_avx512) {
            auto va = mm512_uni_loadu_ps(a + i);
            auto vb = _mm512_sub_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<__m128i*>(b + i)))), v_zp);
            vsum = _mm512_fmadd_ps(va, vb, vsum);
        }
        float sum = _mm512_reduce_add_ps(vsum);
        for (; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
    return;
#elif defined(HAVE_AVX2)
    size_t j = 0;
    for (; j + 4 <= block_size; j += 4) {
        auto vsum0 = _mm256_setzero_ps();
        auto vsum1 = _mm256_setzero_ps();
        auto vsum2 = _mm256_setzero_ps();
        auto vsum3 = _mm256_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto b1 = reinterpret_cast<float*>(b + n + 8);
        auto b2 = reinterpret_cast<float*>(b + (n + 8) * 2);
        auto b3 = reinterpret_cast<float*>(b + (n + 8) * 3);
        auto v_zp0 = _mm256_set1_ps(b0[1]);
        auto v_zp1 = _mm256_set1_ps(b1[1]);
        auto v_zp2 = _mm256_set1_ps(b2[1]);
        auto v_zp3 = _mm256_set1_ps(b3[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            auto vb0 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i)))), v_zp0);
            auto vb1 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + n + 8)))), v_zp1);
            auto vb2 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + 2 * (n + 8))))), v_zp2);
            auto vb3 = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i + 3 * (n + 8))))), v_zp3);

            vsum0 = _mm256_fmadd_ps(va, vb0, vsum0);
            vsum1 = _mm256_fmadd_ps(va, vb1, vsum1);
            vsum2 = _mm256_fmadd_ps(va, vb2, vsum2);
            vsum3 = _mm256_fmadd_ps(va, vb3, vsum3);
        }
        hsum(vsum0);
        hsum(vsum1);
        hsum(vsum2);
        hsum(vsum3);
        float sum0 = _mm256_cvtss_f32(vsum0);
        float sum1 = _mm256_cvtss_f32(vsum1);
        float sum2 = _mm256_cvtss_f32(vsum2);
        float sum3 = _mm256_cvtss_f32(vsum3);
        for (; i < n; i++) {
            sum0 += a[i] * (b[i] - b0[1]);
            sum1 += a[i] * (b[i + n + 8] - b1[1]);
            sum2 += a[i] * (b[i + 2 * (n + 8)] - b2[1]);
            sum3 += a[i] * (b[i + 3 * (n + 8)] - b3[1]);
        }
        c[0] = sum0 * b0[0];
        c[1] = sum1 * b1[0];
        c[2] = sum2 * b2[0];
        c[3] = sum3 * b3[0];
        c += 4;
        b +=  4 * (n + 8) - 8;
    }
    for (; j < block_size; j++) {
        auto vsum = _mm256_setzero_ps();
        auto b0 = reinterpret_cast<float*>(b);
        auto v_zp = _mm256_set1_ps(b0[1]);
        size_t i = 0;
        b += 8;
        for (; i + vec_len_f32_avx2 <= n; i += vec_len_f32_avx2) {
            auto va = mm256_uni_loadu_ps(a + i);
            auto vb = _mm256_sub_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<__m128i*>(b + i)))), v_zp);
            vsum = _mm256_fmadd_ps(va, vb, vsum);
        }
        hsum(vsum);
        float sum = _mm256_cvtss_f32(vsum);
        for (; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
    return;
#endif
    for (size_t j = 0; j < block_size; j++) {
        float sum = 0;
        auto b0 = reinterpret_cast<float*>(b);
        b += 8;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * (b[i] - b0[1]);
        }
        b += n;
        *c++ = sum * b0[0];
    }
}

template<typename T>
static void attn_reduce(T* dst, float* temp, size_t M, size_t S, size_t temp_stride) {
    size_t i = 0;
#if defined(HAVE_AVX512F)
    for (; i + vec_len_f32_avx512 <= S; i+= vec_len_f32_avx512) {
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

// N must be multiple of 16
template<typename TDST, typename TSRC>
void transpose_16NxK(TDST* dst, TSRC* src, TDST* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    size_t k = 0;
    for (; k + 16 <= K; k += 16) {
        for (size_t n = 0; n < N; n += 16) {
            transpose_16x16_kernel(dst + n, src + n * src_stride, dst_stride, src_stride);
        }

        dst += 16 * dst_stride;
        src += 16;
    }
    if (k < K) {
        for (size_t n = 0; n < N; n += 16) {
            transpose_16xK_kernel(dst + n, src + n * src_stride, K - k, dst_stride, src_stride);
        }
    }
}

#if defined(HAVE_AVX512F)
template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void transpose_16NxK(T* dst, T* src, T* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    // will treat as uint32_t transpose
    auto s = reinterpret_cast<uint32_t*>(src);
    auto d = reinterpret_cast<uint32_t*>(dst);
    transpose_16NxK(d, s, reinterpret_cast<uint32_t*>(0), N, K >> 1, dst_stride, src_stride >> 1);
}
#endif

template<typename TDST>
void transpose_16NxK(TDST* dst, uint8_t* src, TDST* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = src;
    auto t = tmp;
    for (size_t n = 0; n < N; n ++) {
        auto f = reinterpret_cast<float*>(s);
        attn_dequant_u8_kernel(s + 2 * sizeof(float), t, K, f[0], f[1]);
        s += src_stride + 2 * sizeof(float);
        t += src_stride;
    }
    transpose_16NxK(dst, tmp, reinterpret_cast<TDST*>(0), N, K, dst_stride, src_stride);
}

// dequant f16/u8 to float
template<typename T>
static inline void dequant(T* dst, T* src, size_t N, size_t K) {
    // never called
    OPENVINO_THROW("dequant: should not be called.");
}

static inline void dequant(float* dst, ov::float16* src, size_t N, size_t K) {
    cvt_copy(dst, src, K * N);
}

template<typename TDST>
void dequant(TDST* dst, uint8_t* src, size_t N, size_t K) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = src;
    for (size_t n = 0; n < N; n ++) {
        auto f = reinterpret_cast<float*>(s);
        attn_dequant_u8_kernel(s + 2 * sizeof(float), dst, K, f[0], f[1]);
        s += K + 2 * sizeof(float);
        dst += K;
    }
}

#if defined(HAVE_AVX512F)
template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32x32_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto a = _mm512_loadu_si512(src);               // [a1  a2  a3 a4 | a5  a6  a7 a8]   total 512-bits in 8 64bits unit
        auto b = _mm512_loadu_si512(src + src_stride);  // [b1  b2  b3 b4 | b5  b6  b7 b8]   total 512-bits
        a = _mm512_permutexvar_epi64(midx, a);          // [a1 a5 | a2 a6 | a3 a7 | a4 a8]
        b = _mm512_permutexvar_epi64(midx, b);          // [b1 b5 | b2 b6 | b3 b7 | b4 b8]
        auto B0 = _mm512_unpacklo_epi16(a, b);          // [ a1&b1  a2&b2   a3&b3   a4&b4] for each 128-bits lane, interleave word in low 64 bits
        auto B1 = _mm512_unpackhi_epi16(a, b);          // [ a5&b5  a6&b6   a7&b7   a8&b8] for each 128-bits lane, interleave word in high 64 bits
        _mm512_storeu_si512(dst, B0);
        _mm512_storeu_si512(dst + 32, B1);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32x16_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    for (size_t i = 0; i < 16; i++) {
        auto x = _mm256_loadu_si256(reinterpret_cast<__m256i*>(src));               // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_loadu_si256(reinterpret_cast<__m256i*>(src + src_stride));  // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);                                      // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);                                      // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_storeu_si512(dst, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32xK_kernel(T* dst, T* src, size_t dst_stride, size_t src_stride, size_t K) {
    static const uint64_t idx[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    auto midx = _mm512_loadu_si512(idx);
    __mmask16 mask = (1 << K) - 1;
    for (size_t i = 0; i < K; i++) {
        auto x = _mm256_maskz_loadu_epi16(mask, src);                              // [a1  a2  a3 a4]   total 256-bits in 4 64bits unit
        auto y = _mm256_maskz_loadu_epi16(mask, src + src_stride);                 // [b1  b2  b3 b4]   total 256-bits
        auto a = _mm512_castsi256_si512(x);
        auto b = _mm512_castsi256_si512(y);
        a = _mm512_permutexvar_epi64(midx, a);                                      // [a1 x | a2 x | a3 x | a4 x]
        b = _mm512_permutexvar_epi64(midx, b);                                      // [b1 x | b2 x | b3 x | b4 x]
        auto B0 = _mm512_unpacklo_epi16(a, b);
        _mm512_mask_storeu_epi32(dst, mask, B0);
        src += 2 * src_stride;
        dst += 2 * dst_stride;
    }
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32NxK(T* dst, T* src, T* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    for (size_t n = 0; n < N; n += 32) {
        size_t k = 0;
        for (; k + 32 <= K; k += 32) {
            pack_32x32_kernel(dst + k * 2, src + k, dst_stride, src_stride);
        }
        if (k + 16 <= K) {
            pack_32x16_kernel(dst + k * 2, src + k, dst_stride, src_stride);
            k += 16;
        }
        if (k < K) {
            pack_32xK_kernel(dst + k * 2, src + k, dst_stride, src_stride, K - k);
        }

        dst += 32 * dst_stride;
        src += 32 * src_stride;
    }
}

template<typename T, typename = typename std::enable_if<(std::is_same<T, ov::bfloat16>::value || std::is_same<T, ov::float16>::value), bool>::type>
static void pack_32NxK(T* dst, uint8_t* src, T* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    // The layout for per token per head:
    // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
    // The quantized feature will start from 8bytes=sizeof(float)+sizeof(float)
    auto s = src;
    auto t = tmp;
    for (size_t n = 0; n < N; n ++) {
        auto f = reinterpret_cast<float*>(s);
        attn_dequant_u8_kernel(s + 2 * sizeof(float), t, K, f[0], f[1]);
        s += src_stride + 2 * sizeof(float);
        t += src_stride;
    }
    pack_32NxK(dst, tmp, reinterpret_cast<T*>(0), N, K, dst_stride, src_stride);
}
#endif

template<typename T>
static void pack_32NxK(float* dst, T* src, float* tmp, size_t N, size_t K, size_t dst_stride, size_t src_stride) {
    // never called
    OPENVINO_THROW("pack_32NxK: should not be called.");
}

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct MHAHelper {
    // initialize once
    size_t _H;
    size_t _S;
    size_t _SV;
    size_t _Hk;
    size_t _h_each_group_len;
    size_t _block_size;
    size_t _nthr;
    size_t _sliding_window;
    float _d_scale;

    PlainTensor _weight;            // [nthr, H, 32, rnd_up(kv_len, block_size)], shared by first and second loop along bh
    PlainTensor _output;            // [nthr, 32, H, S], shared by first and second loop along bh
    PlainTensor _qk_scratch_a;      // [nthr, scratch_a_size]
    PlainTensor _qk_scratch_b;      // [B, rnd_up(kv_len, block_size), Hk, scratch_b_size]
    PlainTensor _wv_scratch_a;
    PlainTensor _wv_scratch_b;
    PlainTensor _alibi_lookup;
    PlainTensor _score_output;
    std::vector<size_t> _wsp;
    size_t _wsp_size_per_thread = 0;

    std::vector<std::shared_ptr<BrgemmKernel>> _qk_gemm;
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm;
    // will accumulate C buffer
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm_acc;
    // second token
    std::shared_ptr<JitMatMulVecAMX> _gemv;
    ov::element::Type _fastpath_valid_prec = ov::element::undefined;
    // second token for bhl loop
    PlainTensor _weight_bhl;
    PlainTensor _output_bhl;
    PlainTensor _score_offsets_aligned;
    PlainTensor _score_offsets;

    MHAHelper() {
        _weight.resize<float>({size_t{1}, size_t{1}, size_t{1}, size_t{1}});
    }

    void init(size_t H, size_t S, size_t SV, size_t Hk, size_t h_each_group_len, size_t block_size, size_t sliding_window,
              float d_scale, size_t kv_len, bool init_alibi_lookup) {
        // query shape: [B, H, L, S]
        // present_key shape: [block, H, 32, S]
        // Q*K': [M1, S] * [M2, S]'
        //   kernel: Q:[1~block_size, S] * K':[block_size, S]'
        //   aka: M:1~block_size, N:block_size, K:S
        // (Q*K')*V: [M1, M2] * [M2, S]
        //   kernel: (Q*K'):[1~block_size, block_size] * V:[block_size, S]
        //   aka: M:1~block_size, N:S, K:block_size
        // Because K and V are from cache, can use M2'=rnd_up(M2, block_size) to simplify logic
        auto in_type = precision_of<DATA_TYPE>::value;
        _H = H;
        _S = S;
        _SV = SV;
        _Hk = Hk;
        _h_each_group_len = h_each_group_len;
        _block_size = block_size;
        _nthr = static_cast<size_t>(parallel_get_max_threads());
        _sliding_window = sliding_window;
        _d_scale = d_scale;

        auto prev_score_stride = _weight.stride(2);
        auto want_score_stride = rnd_up(kv_len, _block_size);
        auto new_score_stride = std::max(prev_score_stride, want_score_stride);
        // resize temporary buffers, weight.size(3) will be aligned to block_size
        _weight.resize<float>({static_cast<size_t>(_nthr), H, _block_size, new_score_stride});
        _output.resize<float>({static_cast<size_t>(_nthr), _block_size, H, SV});

        // TODO: kernel supports stride
        if (_qk_gemm.empty() || prev_score_stride < new_score_stride) {
            _qk_gemm.resize(_block_size);
            _wv_gemm.resize(_block_size);
            _wv_gemm_acc.resize(_block_size);
            for (size_t i = 0; i < _block_size; i++) {
                _qk_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                             _block_size,
                                                             _S,
                                                             _H * _S,
                                                             _block_size,
                                                             _weight.stride(2),
                                                             false,
                                                             in_type);
                _wv_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                             _SV,
                                                             _block_size,
                                                             // if it's bf16, the stride needs double due to reuse float buffer
                                                             (in_type == ov::element::Type_t::f32 ? 1 : 2) * _weight.stride(2),
                                                             _SV,
                                                             _output.stride(1),
                                                             false,
                                                             in_type);
                _wv_gemm_acc[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                                 _SV,
                                                                 _block_size,
                                                                 // if it's bf16, the stride needs double due to reuse float buffer
                                                                 (in_type == ov::element::Type_t::f32 ? 1 : 2) * _weight.stride(2),
                                                                 _SV,
                                                                 _output.stride(1),
                                                                 false,
                                                                 in_type,
                                                                 true);
            }

            // wsp is used to compute beta when K is blocked
            _wsp_size_per_thread = _wv_gemm[0]->get_wsp_size();
            _wsp.resize(_nthr * _wsp_size_per_thread);

            // allocate scratch a/b, notice get_scratch_a_size/get_scratch_b_size returns in bytes
            _qk_scratch_a.resize<DATA_TYPE>({_nthr, _qk_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});
            _wv_scratch_a.resize<DATA_TYPE>({_nthr, _wv_gemm[_block_size - 1]->get_scratch_a_size() / sizeof(DATA_TYPE)});

            if ((S % 32 == 0) && (block_size % 16 == 0) && (S <= 32 * 6)) {
                if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16) &&
                    precision_of<DATA_TYPE>::value == ov::element::bf16 &&
                    precision_of<KVCACHE_TYPE>::value == ov::element::bf16) {
                    _fastpath_valid_prec = ov::element::bf16;
                } else if (dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_fp16) &&
                           precision_of<DATA_TYPE>::value == ov::element::f16 &&
                           precision_of<KVCACHE_TYPE>::value == ov::element::f16) {
                    _fastpath_valid_prec = ov::element::f16;
                }
            }
            if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16) && !_gemv) {
                _gemv = std::make_shared<JitMatMulVecAMX>(static_cast<int>(S), static_cast<int>(block_size), _fastpath_valid_prec);
            }
        }

        if (init_alibi_lookup && (!_alibi_lookup || _alibi_lookup.m_dims[0] < kv_len)) {
            _alibi_lookup.resize<float>({kv_len * 2});
            for (size_t i = 0; i < _alibi_lookup.m_dims[0]; i++)
                _alibi_lookup.ptr<float>()[i] = - static_cast<int>((_alibi_lookup.m_dims[0] - 1 - i));
        }
    }

    void init_reorder_buffers(size_t batch, size_t kv_len_in_blocks) {
        _qk_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, _Hk, _block_size * _S});
        _wv_scratch_b.resize<DATA_TYPE>({batch, kv_len_in_blocks, _Hk, _block_size * rnd_up(_SV, _block_size)});
    }

    void init_score_buffers(const PlainTensor& past_lens, const PlainTensor& subsequence_begins) {
        static constexpr int cache_line_size = dnnl::impl::cpu::platform::get_cache_line_size();
        auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
        _score_offsets_aligned.resize<int32_t>({past_lens.m_dims[0]});
        _score_offsets.resize<int32_t>({past_lens.m_dims[0]});
        int32_t total_kv_len_aligned = 0;
        int32_t total_kv_len = 0;
        for (int32_t i = 0; i < seq_cout; i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            _score_offsets_aligned.ptr<int32_t>()[i] = total_kv_len_aligned;
            _score_offsets.ptr<int32_t>()[i] = total_kv_len;
            // aligned to cache line to avoid false sharing
            total_kv_len_aligned += rnd_up(kv_len, cache_line_size / sizeof(float));
            total_kv_len += kv_len;
        }

        _score_output.resize<float>({total_kv_len_aligned * _H});
    }

    // compute one block(such as 32 tokens) of query in M dimension: softmax(q_block*k')*v
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_value: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  qk_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    //  wv_scratch_b: [rnd_up(kv_len, block_size), Hk, scratch_b_size]
    void exec_kernel_multiple(const PlainTensor& query, const PlainTensor& present_value, const PlainTensor& output_emb,
        const PlainTensor& qk_scratch_b, const PlainTensor& wv_scratch_b, const int32_t* block_table, size_t ithr, size_t q_blk,
        size_t hk, size_t q_len, size_t cur_kv_len, const PlainTensor& alibi_slopes, float* score_output) {
        auto q_start = q_blk * _block_size;
        auto q_end = std::min(q_start + _block_size, q_len);
        auto q_cnt = q_end - q_start;
        constexpr bool q_is_xf16 = one_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == precision_of<KVCACHE_TYPE>::value;
        auto cur_kv_len_blocks = div_up(cur_kv_len, _block_size);
        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
            auto* q_ptr = query.ptr<DATA_TYPE>(h, q_start, 0);
            float* c_ptr = _weight.ptr<float>(ithr, h, 0, 0);
            // for each query block, loop through all key block
            // for blocks:
            // 1 0 0 0 ...
            // 1 1 0 0 ...
            // 1 1 1 0 ...
            // just computing the positions of 1 should be enough
            for (size_t k_blk = 0; k_blk < cur_kv_len_blocks; k_blk++) {
                auto* k_ptr = qk_scratch_b.ptr<DATA_TYPE>(k_blk, hk);
                _qk_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                 q_ptr,
                                                 k_ptr,
                                                 c_ptr + k_blk * _block_size,
                                                 _wsp.data() + ithr * _wsp_size_per_thread,
                                                 _qk_scratch_a ? _qk_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
            }

            for (size_t m = q_start; m < q_end; m++) {
                // apply attention mask & sofmax
                auto ncausal = (cur_kv_len - q_cnt + (m - q_start) + 1);
                auto score = _weight.ptr<float>(ithr, h, m - q_start);
                if (_sliding_window) {
                    size_t start_idx = 0;
                    auto new_causal = ncausal;
                    float* alibi_lookup = nullptr;
                    if (ncausal > _sliding_window) {
                        start_idx = ncausal - static_cast<size_t>(_sliding_window);
                        new_causal = _sliding_window;
                    }
                    attn_softmax_kernel<float>(score + start_idx,
                                               reinterpret_cast<DATA_TYPE*>(score) + start_idx,
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               new_causal,
                                               rnd_up(cur_kv_len, _block_size) - start_idx,
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value);

                    memset(score, 0, sizeof(DATA_TYPE) * start_idx);
                } else {
                    // alibi may available when _sliding_window is false
                    float* alibi_lookup = nullptr;
                    float alibi_slope = 0.f;
                    if (alibi_slopes) {
                        alibi_slope = alibi_slopes.ptr<float>()[h];
                        alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - ncausal;
                    }
                    attn_softmax_kernel<float>(score,
                                               reinterpret_cast<DATA_TYPE*>(score),
                                               _d_scale,
                                               alibi_lookup,
                                               nullptr,
                                               nullptr,
                                               false,
                                               ncausal,
                                               rnd_up(cur_kv_len, _block_size),
                                               precision_of<DATA_TYPE>::value,
                                               precision_of<DATA_TYPE>::value,
                                               alibi_slope);
                }
                if (score_output) {
                    cvt_copy(score_output + h * rnd_up(cur_kv_len, 16), reinterpret_cast<DATA_TYPE*>(score), cur_kv_len);
                }
            }

            // reuse float buffer, need to use float to compute offset
            auto* w_ptr = reinterpret_cast<DATA_TYPE*>(_weight.ptr<float>(ithr, h, 0, 0));
            float* fp32_out_ptr = q_is_xf16 ? _output.ptr<float>(ithr, 0, h, 0) : output_emb.ptr<float>(q_start, h * _SV);

            // for each weight block, loop through all value block
            for (size_t v_blk = 0; v_blk < cur_kv_len_blocks; v_blk++) {
                DATA_TYPE* v_ptr;
                if (q_is_xf16 || !q_cache_is_same) {
                    v_ptr = wv_scratch_b.ptr<DATA_TYPE>(v_blk, hk);
                } else {
                    v_ptr = present_value.ptr<DATA_TYPE>(block_table[v_blk], hk);
                }
                if (v_blk == 0) {
                    _wv_gemm[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                     w_ptr + v_blk * _block_size,
                                                     v_ptr,
                                                     fp32_out_ptr,
                                                     _wsp.data() + ithr * _wsp_size_per_thread,
                                                     _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                } else {
                    _wv_gemm_acc[q_cnt - 1]->executeGemm(q_cnt < _block_size,
                                                         w_ptr + v_blk * _block_size,
                                                         v_ptr,
                                                         fp32_out_ptr,
                                                         _wsp.data() + ithr * _wsp_size_per_thread,
                                                         _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(ithr, 0) : nullptr);
                }
            }
            if (q_is_xf16) {
                attn_memcpy2d_kernel(_output.ptr<float>(ithr, 0, h, 0),
                                     output_emb.ptr<DATA_TYPE>(q_start, h * _SV),
                                     ov::element::f32,
                                     precision_of<DATA_TYPE>::value,
                                     _output.stride(1),
                                     output_emb.stride(0),
                                     _SV,
                                     q_cnt);
            }
        }
    }

    // compute one token, loop along batch and head dimensions
    // all tensors such as query... have no batch dimension because batch dimension is varying
    //  query: [H, L, S]
    //  present_*: [block_number, H, 32, S]
    //  output_emb: [L, H * S]
    //  weight: [nthr, H, 32, rnd_up(kv_len, block_size)]
    //  output: [nthr, 32, H, S]
    void exec_kernel_one_bh(const PlainTensor& query, const PlainTensor& present_key, const PlainTensor& present_value, const PlainTensor& output_emb,
        const int32_t* block_table, size_t ithr, size_t hk, size_t q_len, size_t cur_kv_len, const PlainTensor& alibi_slopes, float* score_output) {
        if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
            _gemv->tile_config();
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                        (*_gemv)(query.ptr<DATA_TYPE>(h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                            _weight.ptr<float>(ithr, h, pq) + pk);
                    }
                }
            }
            _gemv->tile_release();
        } else {
            for (size_t pk = 0, i = 0; pk < cur_kv_len; pk += _block_size, i++) {
                auto block_number = block_table[i];
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                        dot_product_block(query.ptr<DATA_TYPE>(h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                            _weight.ptr<float>(ithr, h, pq) + pk, _S, std::min(_block_size, cur_kv_len - pk));
                    }
                }
            }
        }

        for (size_t pq = 0; pq < q_len; pq++) {
            for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                // apply attention mask & sofmax
                float* alibi_lookup = nullptr;
                float alibi_slope = 0.f;
                if (alibi_slopes) {
                    alibi_slope = alibi_slopes.ptr<float>()[h];
                    alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
                }
                attn_softmax_kernel<float>(_weight.ptr<float>(ithr, h, pq),
                                           _weight.ptr<float>(ithr, h, pq),
                                           _d_scale,
                                           alibi_lookup,
                                           nullptr,
                                           nullptr,
                                           false,
                                           cur_kv_len,
                                           cur_kv_len,
                                           ov::element::f32,
                                           ov::element::f32,
                                           alibi_slope);
                if (score_output) {
                    memcpy(score_output + h * rnd_up(cur_kv_len, 16), _weight.ptr<float>(ithr, h, pq), cur_kv_len * sizeof(float));
                }
            }
        }

        memset(_output.ptr<float>(ithr), 0, q_len * _H * _SV * sizeof(float));
        for (size_t pv = 0, i = 0; pv < cur_kv_len; pv += _block_size, i++) {
            auto block_number = block_table[i];
            auto* v = present_value.ptr<KVCACHE_TYPE>(block_number, hk);
            for (size_t pq = 0; pq < q_len; pq++) {
                for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                    attn_acc_value_block(_output.ptr<float>(ithr, pq, h),
                                         _weight.ptr<float>(ithr, h, pq) + pv,
                                         v,
                                         _SV,
                                         std::min(_block_size, cur_kv_len - pv));
                }
            }
        }
        // convert to dst
        for (size_t pq = 0; pq < q_len; pq++)
            for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++)
                cvt_copy(output_emb.ptr<DATA_TYPE>(pq, h * _SV), _output.ptr<float>(ithr, pq, h), _SV);
    }

    // compute one token, loop along batch, head dimensions and kv_len, it's special for very long kv_len with small batch tokens.
    // It will assume NO mixture execution of first and second token.
    // all tensors such as query... have batch dimension which is DIFFERENT from above
    //  query: [B, H, L, S]
    //  present_*: [block_number, H, 32, S]
    //  output_emb: [B, L, H * S]
    // 3 loops along batch, head, kv cache length dimensions
    void exec_loop_bhl(const PlainTensor& query,
                       const PlainTensor& present_key,
                       const PlainTensor& present_value,
                       const PlainTensor& output_emb,
                       const PlainTensor& output_score,
                       size_t max_context_len,
                       const PlainTensor& past_lens,
                       const PlainTensor& subsequence_begins,
                       const PlainTensor& block_indices,
                       const PlainTensor& block_indices_begins,
                       const PlainTensor& alibi_slopes) {
        auto B = past_lens.size(0);
        auto q_len = query.size(2);
        auto kv_len_in_blocks = div_up(max_context_len, _block_size);

        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        _weight_bhl.resize<float>({B, _H, q_len, rnd_up(max_context_len, std::max(_block_size, size_t{16}))});

        parallel_for3d_dynamic(B, kv_len_in_blocks, _Hk, [&](size_t b, size_t pk_in_blocks, size_t hk) {
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            // kv_len must be valid
            auto pk = pk_in_blocks * _block_size;
            if (pk < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pk_in_blocks];
                if (one_of(_fastpath_valid_prec, ov::element::bf16, ov::element::f16)) {
                    _gemv->tile_config();
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            (*_gemv)(query.ptr<DATA_TYPE>(b, h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                                _weight_bhl.ptr<float>(b, h, pq) + pk);
                        }
                    }
                    _gemv->tile_release();
                } else {
                    for (size_t pq = 0; pq < q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            dot_product_block(query.ptr<DATA_TYPE>(b, h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                                _weight_bhl.ptr<float>(b, h, pq) + pk, _S, std::min(_block_size, context_len - pk));
                        }
                    }
                }
            }
        });

        parallel_for3d_dynamic(B, _H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto ncausal = cur_kv_len;
            // apply attention mask & sofmax
            float* alibi_lookup = nullptr;
            float alibi_slope = 0.f;
            if (alibi_slopes) {
                alibi_slope = alibi_slopes.ptr<float>()[h];
                alibi_lookup = _alibi_lookup.ptr<float>() + _alibi_lookup.m_dims[0] - cur_kv_len;
            }
            attn_softmax_kernel<float>(_weight_bhl.ptr<float>(b, h, pq),
                                       _weight_bhl.ptr<float>(b, h, pq),
                                       _d_scale,
                                       alibi_lookup,
                                       nullptr,
                                       nullptr,
                                       false,
                                       ncausal,
                                       cur_kv_len,
                                       ov::element::f32,
                                       ov::element::f32,
                                       alibi_slope);
        });

        if (output_score) {
            parallel_for2d_dynamic(B, q_len, [&](size_t b, size_t pq) {
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
                auto* src = _weight_bhl.ptr<float>(b, 0, pq);
                size_t src_stride = _weight_bhl.stride(2);
                auto* dst = output_score.ptr<float>() + _score_offsets.ptr<int32_t>()[b];
                attn_reduce(dst, src, _H, cur_kv_len, src_stride);
            });
        }

        // attn_w * V
        _output_bhl.resize<float>({static_cast<size_t>(_nthr), B, q_len, _H, _SV});
        // m_attn_w {B, H, q_len, kv_len}
        parallel_nt_static(_nthr, [&](const size_t ithr, const size_t nthr) {
            memset(_output_bhl.ptr<float>(ithr, 0, 0, 0, 0), 0, _output_bhl.stride(0) * sizeof(float));
        });

        parallel_for3d_dynamic(B, kv_len_in_blocks, _Hk, [&](size_t b, size_t pv_in_blocks, size_t hk) {
            auto ithr = parallel_get_thread_num();
            auto context_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + 1;
            auto pv = pv_in_blocks * _block_size;
            // kv_len must be valid
            if (pv < context_len) {
                auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[b] + pv_in_blocks];
                auto* v = present_value.ptr<KVCACHE_TYPE>(block_number, hk);
                for (size_t pq = 0; pq < q_len; pq++) {
                    for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                        attn_acc_value_block(_output_bhl.ptr<float>(ithr, b, pq, h),
                                             _weight_bhl.ptr<float>(b, h, pq) + pv,
                                             v,
                                             _SV,
                                             std::min(_block_size, context_len - pv));
                    }
                }
            }
        });

        parallel_for3d(B, _H, q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = _output_bhl.ptr<float>(0, b, pq, h);
            size_t temp_stride = _output_bhl.stride(0);
            auto* dst = output_emb.ptr<DATA_TYPE>(b, pq, h * _SV);
            attn_reduce(dst, temp, _nthr, _SV, temp_stride);
        });
    }
};

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct MHA {
    MHAHelper<DATA_TYPE, KVCACHE_TYPE>& _helper;
    struct AttnWorkItem {
        int32_t batch_in_reorder;                   // which batch in reorder buffer will be used
        int32_t batch_in_seq;                       // batch idx in sequence
        int32_t q_len;                              // current sequence length, 1 for second token, 2+ for first token
        int32_t q_block_id;                         // block id in this seq, valid at first token
    };
    struct ReorderWorkItem {
        int32_t batch_in_seq;                       // batch idx in sequence
        int32_t batch_in_reorder;                   // which batch in reorder buffer will be used
        int32_t kv_block_id;                        // block id in this kv cache seq
    };
    struct WorkItems {
    private:
        std::vector<AttnWorkItem> attn_items;
        std::vector<ReorderWorkItem> reorder_items;
        int32_t max_kv_len_in_reorder;              // max kv len between first tokens
        int32_t max_batch_in_reorder;
        int32_t total_kv_len;

    public:
        void reset(const PlainTensor& query, const PlainTensor& past_lens, const PlainTensor& subsequence_begins, size_t block_size) {
            attn_items.clear();
            reorder_items.clear();
            max_kv_len_in_reorder = 0;
            max_batch_in_reorder = 0;
            total_kv_len = 0;

            auto seq_cout = static_cast<int32_t>(past_lens.m_dims[0]);
            for (int32_t i = 0; i < seq_cout; i++) {
                auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
                auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
                auto kv_len_in_block = static_cast<int32_t>(div_up(kv_len, block_size));
                if (q_len == 1) {
                    attn_items.emplace_back(AttnWorkItem{
                        0,                          // batch_in_reorder
                        i,                          // batch_in_seq
                        1ull,                       // q_len
                        // kv_len in blocks, used in the sort function
                        kv_len_in_block - 1
                    });
                } else {
                    auto reorder_sub_work_count = kv_len_in_block;
                    max_kv_len_in_reorder = std::max(max_kv_len_in_reorder, kv_len);
                    for (int32_t block_id = 0; block_id < reorder_sub_work_count; block_id++) {
                        reorder_items.emplace_back(ReorderWorkItem{
                            i,                       // batch_in_seq
                            max_batch_in_reorder,    // batch_in_reorder
                            block_id                 // kv_block_id
                        });
                    }

                    // workitems for attention
                    auto attn_sub_work_count = static_cast<int32_t>(div_up(q_len, block_size));
                    for (int32_t block_id = 0; block_id < attn_sub_work_count; block_id++) {
                        attn_items.emplace_back(AttnWorkItem{
                            max_batch_in_reorder,    // batch_in_reorder
                            i,                       // batch_in_seq
                            q_len,                   // q_len
                            block_id                 // q_block_id
                        });
                    }
                    max_batch_in_reorder++;
                }
                total_kv_len += kv_len;
            }
            // std::sort(attn_items.begin(), attn_items.end(), [] (const AttnWorkItem& left, const AttnWorkItem& right) {
            //     // kv block number which will be acessed later
            //     auto left_kv_blocks = left.q_block_id;
            //     auto right_kv_blocks = right.q_block_id;
            //     return left_kv_blocks > right_kv_blocks;
            // });
        }
        const AttnWorkItem& get_attn_work_item(size_t idx) const {
            return attn_items[idx];
        }
        size_t attn_work_size() const {
            return attn_items.size();
        }
        const ReorderWorkItem& get_reorder_work_item(size_t idx) const {
            return reorder_items[idx];
        }
        size_t reorder_work_size() const {
            return reorder_items.size();
        }
        size_t get_reorder_max_batch_size() const {
            return static_cast<size_t>(max_batch_in_reorder);
        }
        size_t get_reorder_max_kv_len() const {
            return static_cast<size_t>(max_kv_len_in_reorder);
        }
        size_t get_total_kv_len() const {
            return static_cast<size_t>(total_kv_len);
        }
    };

    WorkItems _workitems;

    MHA(MHAHelper<DATA_TYPE, KVCACHE_TYPE>& helper) : _helper(helper) {}

    // one loop to handle first and second tokens
    void exec_loop_mixed(const PlainTensor& q,
                         const PlainTensor& k_cache,
                         const PlainTensor& v_cache,
                         const PlainTensor& output_emb,
                         const PlainTensor& output_score,
                         size_t max_context_len,
                         const PlainTensor& past_lens,
                         const PlainTensor& subsequence_begins,
                         const PlainTensor& block_indices,
                         const PlainTensor& block_indices_begins,
                         const PlainTensor& alibi_slopes) {
        auto Hk = v_cache.m_dims[1];

        constexpr bool q_is_xf16 = one_of(precision_of<DATA_TYPE>::value, ov::element::bf16, ov::element::f16);
        constexpr bool q_cache_is_same = precision_of<DATA_TYPE>::value == precision_of<KVCACHE_TYPE>::value;
        auto attn_work_count = _workitems.attn_work_size();
        auto reorder_work_count = _workitems.reorder_work_size();

        // buffer for transpose and repack
        _helper.init_reorder_buffers(_workitems.get_reorder_max_batch_size(), div_up(_workitems.get_reorder_max_kv_len(), _helper._block_size));

        // packed k, v
        parallel_for2d_dynamic(reorder_work_count, Hk, [&](size_t w, size_t hk) {
            const auto& item = _workitems.get_reorder_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_reorder = item.batch_in_reorder;
            const auto kv_block = item.kv_block_id;
            auto block_number = block_indices.ptr<int32_t>()[block_indices_begins.ptr<int32_t>()[batch_in_seq] + kv_block];
            if (block_number < 0)
                return;

            auto ithr = parallel_get_thread_num();
            auto* k_ptr = k_cache.ptr<KVCACHE_TYPE>(block_number, hk);
            auto* v_ptr = v_cache.ptr<KVCACHE_TYPE>(block_number, hk);
            transpose_16NxK(_helper._qk_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                k_ptr,
                _helper._output.template ptr<DATA_TYPE>(ithr),
                _helper._block_size,
                _helper._S, _helper._block_size, _helper._S);
            if (q_is_xf16) {
                pack_32NxK(_helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk),
                           v_ptr,
                           _helper._output.template ptr<DATA_TYPE>(ithr),
                           _helper._block_size,
                           _helper._SV,
                           rnd_up(_helper._SV, _helper._block_size),
                           _helper._SV);
            } else {
                // need to decompress
                if (!q_cache_is_same) {
                    dequant(_helper._wv_scratch_b.template ptr<DATA_TYPE>(batch_in_reorder, kv_block, hk), v_ptr, _helper._block_size, _helper._SV);
                }
            }
        });

        parallel_for2d_dynamic(attn_work_count, Hk, [&](size_t w, size_t hk) {
            const auto& item = _workitems.get_attn_work_item(w);
            const auto batch_in_seq = item.batch_in_seq;
            const auto batch_in_token = subsequence_begins.ptr<int32_t>()[batch_in_seq];
            const auto q_len = static_cast<size_t>(item.q_len);
            size_t ithr = parallel_get_thread_num();

            if (q_len == 1) {
                const auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + 1;
                float* score_output = nullptr;
                if (output_score) {
                    auto score_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[batch_in_seq];
                    score_output = _helper._score_output.template ptr<float>() + score_offset * _helper._H;
                }

                _helper.exec_kernel_one_bh(q.slice(0, batch_in_token, batch_in_token), k_cache, v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr, hk, 1ul, cur_kv_len, alibi_slopes,
                    score_output);
            } else {
                const auto batch_in_reorder = item.batch_in_reorder;
                const auto q_blk = item.q_block_id;
                const auto q_cnt = std::min(_helper._block_size, q_len - q_blk * _helper._block_size);
                const auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[batch_in_seq]) + q_blk * _helper._block_size + q_cnt;
                float* score_output = nullptr;
                if (output_score) {
                    // last block
                    if (q_len - q_blk * _helper._block_size <= _helper._block_size) {
                        auto score_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[batch_in_seq];
                        score_output = _helper._score_output.template ptr<float>() + score_offset * _helper._H;
                    }
                }

                PlainTensor sub_query;
                sub_query.resize({q_len, _helper._H, _helper._S}, q.ptr<DATA_TYPE>(batch_in_token));
                sub_query = sub_query.permute({1, 0, 2});
                _helper.exec_kernel_multiple(sub_query,
                    v_cache,
                    output_emb.slice(0, batch_in_token, batch_in_token + q_len).reshape({q_len, _helper._H * _helper._SV}),
                    _helper._qk_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    _helper._wv_scratch_b.slice(0, batch_in_reorder, batch_in_reorder),
                    block_indices.ptr<int32_t>() + block_indices_begins.ptr<int32_t>()[batch_in_seq],
                    ithr,
                    q_blk,
                    hk,
                    q_len,
                    cur_kv_len,
                    alibi_slopes,
                    score_output);
            }
        });
        if (output_score) {
            parallel_for2d_dynamic(past_lens.m_dims[0], 1, [&](size_t b, size_t pq) {
                auto seq_len = static_cast<size_t>(subsequence_begins.ptr<int32_t>()[b + 1] - subsequence_begins.ptr<int32_t>()[b]);
                auto cur_kv_len = static_cast<size_t>(past_lens.ptr<int32_t>()[b]) + seq_len;
                auto src_offset = _helper._score_offsets_aligned.template ptr<int32_t>()[b];
                auto* src = _helper._score_output.template ptr<float>() + src_offset * _helper._H;
                size_t src_stride = rnd_up(cur_kv_len, 16);
                auto dst_offset = _helper._score_offsets.template ptr<int32_t>()[b];
                auto* dst = output_score.ptr<float>() + dst_offset;
                attn_reduce(dst, src, _helper._H, cur_kv_len, src_stride);
            });
        }
    }

    // Q, K, V is ready, do attention
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    PlainTensor& output_score,
                    size_t max_context_len,
                    const PlainTensor& past_lens,
                    const PlainTensor& subsequence_begins,
                    const PlainTensor& block_indices,
                    const PlainTensor& block_indices_begins,
                    const PlainTensor& alibi_slopes) {
        _workitems.reset(query, past_lens, subsequence_begins, _helper._block_size);
        if (output_score)
            _helper.init_score_buffers(past_lens, subsequence_begins);

        auto nthr = static_cast<size_t>(parallel_get_max_threads());

        if (past_lens.m_dims[0] >= nthr || _workitems.get_reorder_max_batch_size() > 0) {
            exec_loop_mixed(query, present_key, present_value, output_emb, output_score, max_context_len, past_lens, subsequence_begins,
                block_indices, block_indices_begins, alibi_slopes);
        } else {
            _helper.exec_loop_bhl(query, present_key, present_value, output_emb, output_score, max_context_len, past_lens, subsequence_begins,
                block_indices, block_indices_begins, alibi_slopes);
        }
    }
};

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct AttentionExecutor : public PagedAttentionExecutor {
    MHAHelper<DATA_TYPE, KVCACHE_TYPE> _helper;
    MHA<DATA_TYPE, KVCACHE_TYPE> _kernel;
    PlainTensor _slot_mapping;

    AttentionExecutor() : _kernel(_helper) {}

    void init(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr>& outputs, PlainTensor& q, PlainTensor& k, PlainTensor& v, PlainTensor& k_cache,
        PlainTensor& v_cache, PlainTensor& past_lens, PlainTensor& subsequence_begins, PlainTensor& block_indices, PlainTensor& block_indices_begins,
        float& scale, size_t& sliding_window, PlainTensor& alibi_slopes, size_t& max_context_len, PlainTensor& output_emb, PlainTensor& output_score) {
        q.reset(inputs[ID_Q]);                                      // [B_token, H * S]
        k.reset(inputs[ID_K]);
        v.reset(inputs[ID_V]);
        k_cache.reset(inputs[ID_KCACHE]);                           // [NUM_BLOCKS, H, 32, S]
        v_cache.reset(inputs[ID_VCACHE]);                           // [NUM_BLOCKS, H, 32, S]
        past_lens.reset(inputs[ID_PAST_LENS]);                      // [B_seq]
        subsequence_begins.reset(inputs[ID_SUBSEQUENCE_BEGINS]);    // [B_seq+1]
        block_indices.reset(inputs[ID_BLOCK_INDICES]);              // [num_blocks]
        block_indices_begins.reset(inputs[ID_BLOCK_INDICES_BEGINS]);// [B_seq+1]
        scale = *inputs[ID_SCALE]->getDataAs<float>();
        sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        if (!inputs[ID_ALIBI_SLOPES]->getShape().hasZeroDims())
            alibi_slopes.reset(inputs[ID_ALIBI_SLOPES]);
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());
        output_emb.reset(outputs[0]);
        if (outputs.size() == 2)
            output_score.reset(outputs[1]);

        auto B_token = q.size(0);
        auto Hk = k_cache.size(1);
        // The layout for per token per head for u8 kv cache:
        // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
        // The actual size needs to deduct scale and zeropoint.
        auto S = k_cache.size(3) - (k_cache.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto SV = v_cache.size(3) - (k_cache.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto block_size = k_cache.size(2);
        auto H = q.size(1) / S;
        auto h_each_group_len = 1;
        if (Hk != H) {
            h_each_group_len = H / Hk;
        }
        auto B_seq = past_lens.size(0);

        q.assert_dims({B_token, H * S});
        k.assert_dims({B_token, Hk * S});
        v.assert_dims({B_token, Hk * SV});
        q = q.reshape({B_token, H, 1, S});
        k = k.reshape({B_token, Hk, 1, S});
        v = v.reshape({B_token, Hk, 1, SV});
        if (k_cache.m_dt == ov::element::Type_t::u8) {
            k_cache.assert_dims({0, Hk, block_size, S + sizeof(float) * 2}, true);
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV + sizeof(float) * 2});
        } else {
            k_cache.assert_dims({0, Hk, block_size, S}, true);
            v_cache.assert_dims({k_cache.m_dims[0], Hk, block_size, SV});
        }
        past_lens.assert_dims({B_seq});
        subsequence_begins.assert_dims({B_seq + 1});
        block_indices.assert_dims({0}, true);
        block_indices_begins.assert_dims({B_seq + 1});
        if (scale == 0.0f)
            scale = 1.0f / sqrt(S);
        if (alibi_slopes) {
            alibi_slopes.assert_dims({H});
        }
        output_emb.assert_dims({B_token, H * SV});
        output_emb = output_emb.reshape({B_token, 1, H * SV});

        // TODO: enable block_size to be multiple of 32
        OPENVINO_ASSERT(block_size == 32, "CPU: block size must be 32, current: ", block_size);

        _helper.init(H, S, SV, Hk, h_each_group_len, block_size, sliding_window, scale, max_context_len, alibi_slopes);
    }

    void concat_pastkv(const PlainTensor& k, const PlainTensor& v, const PlainTensor& k_cache, const PlainTensor& v_cache,
        const PlainTensor& past_lens, const PlainTensor& subsequence_begins, const PlainTensor& block_indices, const PlainTensor& block_indices_begins) {
        auto B_token = k.size(0);
        _slot_mapping.resize<int32_t>({B_token});

        size_t idx = 0;
        for (size_t i = 0; i < past_lens.size(0); i++) {
            auto q_len = subsequence_begins.ptr<int32_t>()[i + 1] - subsequence_begins.ptr<int32_t>()[i];
            auto kv_len = past_lens.ptr<int32_t>()[i] + q_len;
            auto block_number_start = block_indices_begins.ptr<int32_t>()[i];
            auto block_offset_start = kv_len - q_len;
            for (int32_t j = 0; j < q_len; j++) {
                auto block_offset = block_offset_start + j;
                auto block_number = block_indices.ptr<int32_t>()[block_number_start + block_offset / _helper._block_size];
                _slot_mapping.ptr<int32_t>()[idx++] = block_number * _helper._block_size + block_offset % _helper._block_size;
            }
        }

        if (k_cache.m_dt == ov::element::Type_t::u8) {
            paged_attn_quantkv(k, v, k_cache, v_cache, _slot_mapping);
        } else {
            paged_attn_memcpy(k, v, k_cache, v_cache, _slot_mapping);
        }
    }

    void execute(const std::vector<MemoryPtr>& inputs, const std::vector<MemoryPtr> outputs) override {
        PlainTensor q, k, v, k_cache, v_cache;
        PlainTensor past_lens, subsequence_begins, block_indices, block_indices_begins;
        float scale;
        size_t sliding_window;
        PlainTensor alibi_slopes;
        size_t max_context_len;
        PlainTensor output_emb;
        PlainTensor output_score;

        init(inputs, outputs, q, k, v, k_cache, v_cache, past_lens, subsequence_begins, block_indices, block_indices_begins,
            scale, sliding_window, alibi_slopes, max_context_len, output_emb, output_score);
        concat_pastkv(k, v, k_cache, v_cache, past_lens, subsequence_begins, block_indices, block_indices_begins);

        _kernel(q, k_cache, v_cache, output_emb, output_score, max_context_len, past_lens, subsequence_begins, block_indices,
            block_indices_begins, alibi_slopes);
    }
};
#endif

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type, ov::element::Type kvcache_type) {
    std::shared_ptr<PagedAttentionExecutor> executor;

#ifdef OPENVINO_ARCH_X86_64
    if (data_type == ov::element::bf16) {
#if defined(HAVE_AVX512F)
        if (kvcache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, uint8_t>>();
        } else {
            OPENVINO_ASSERT(kvcache_type == ov::element::bf16, "expect kvcache type bf16, current: ", kvcache_type);
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::bfloat16>>();
        }
#else
        OPENVINO_THROW("make_pa_executor: bf16 needs avx512+ hardware.");
#endif
    } else if (data_type == ov::element::f16) {
#if defined(HAVE_AVX512F)
        if (kvcache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<ov::float16, uint8_t>>();
        } else {
            OPENVINO_ASSERT(kvcache_type == ov::element::f16, "expect kvcache type f16, current: ", kvcache_type);
            executor = std::make_shared<AttentionExecutor<ov::float16, ov::float16>>();
        }
#else
     OPENVINO_THROW("make_pa_executor: f16 needs avx512+ hardware.");
#endif
    } else if (data_type == ov::element::f32) {
        if (kvcache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<float, uint8_t>>();
        } else if (kvcache_type == ov::element::f16) {
            executor = std::make_shared<AttentionExecutor<float, ov::float16>>();
        } else {
            OPENVINO_ASSERT(kvcache_type == ov::element::f32, "expect kvcache type f32, current: ", kvcache_type);
            executor = std::make_shared<AttentionExecutor<float, float>>();
        }
    } else {
        OPENVINO_THROW("make_pa_executor: unsupported precision: ", data_type);
    }
#else
    OPENVINO_THROW("make_pa_executor: only support x64 platform");
#endif
    return executor;
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov