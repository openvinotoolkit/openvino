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
#include "openvino/core/parallel.hpp"
#include "executor_pa.hpp"
#include "executor_pa_common.hpp"
#include "common.hpp"
#include "softmax_kernel.hpp"
#include "utils/plain_tensor.hpp"
#include "utils/profiler.hpp"
#include "attn_memcpy.hpp"
#include "nodes/kernels/x64/brgemm_kernel.hpp"

namespace ov {
namespace Extensions {
namespace Cpu {
namespace XARCH {

using namespace ov;
using namespace ov::intel_cpu;

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

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct MHAKernel {
    // q: [B, H, q_len, S]
    // k: [B, H, kv_len, S]
    // v: [B, H, kv_len, S]
    PlainTensor _score;
    PlainTensor _weight;
    PlainTensor _fp32_out;
    PlainTensor _qk_scratch_a;
    PlainTensor _qk_scratch_b;
    PlainTensor _wv_scratch_a;
    PlainTensor _wv_scratch_b;
    std::vector<size_t> _wsp;
    size_t _wsp_size_per_thread = 0;

    std::vector<std::shared_ptr<BrgemmKernel>> _qk_gemm;
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm;
    // will accumulate C buffer
    std::vector<std::shared_ptr<BrgemmKernel>> _wv_gemm_acc;

    size_t _B;
    size_t _H;
    size_t _q_len;
    size_t _S;
    size_t _Hk;
    size_t _h_each_group_len;
    size_t _block_size;
    size_t _nthr;
    size_t _kv_len_in_blocks;

    MHAKernel() {
        _score.resize<float>({1ul, 1ul, 1ul, 1ul});
        _weight.resize<float>({1ul, 1ul, 1ul, 1ul});
    }

    void prepare_brgemm_prim(PlainTensor& query, PlainTensor& present_key, const PlainTensor& block_tables) {
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

        auto prev_score_stride = _score.stride(2);
        auto want_score_stride = rnd_up(_q_len, _block_size);
        auto new_score_stride = std::max(prev_score_stride, want_score_stride);
        // resize temporary buffers, score.size(3) will be aligned to block_size
        _score.resize<float>({static_cast<size_t>(_nthr), _H, _q_len, new_score_stride});
        _weight.resize<float>({static_cast<size_t>(_nthr), _H, _q_len, new_score_stride});
        _fp32_out.resize<float>({static_cast<size_t>(_nthr), _q_len, _H, _S});

        // TODO: kernel supports stride
        if (_qk_gemm.empty() || prev_score_stride < new_score_stride) {
            _qk_gemm.resize(_block_size);
            _wv_gemm.resize(_block_size);
            _wv_gemm_acc.resize(_block_size);
            for (size_t i = 0; i < _block_size; i++) {
                _qk_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                             _block_size,
                                                             _S,
                                                             query.stride(2),
                                                             present_key.stride(2),
                                                             _score.stride(2),
                                                             true,
                                                             in_type);
                _wv_gemm[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                             _S,
                                                             _block_size,
                                                             _weight.stride(2),
                                                             present_key.stride(2),
                                                             _fp32_out.stride(1),
                                                             false,
                                                             in_type);
                _wv_gemm_acc[i] = std::make_shared<BrgemmKernel>(i + 1,
                                                                 _S,
                                                                 _block_size,
                                                                 _weight.stride(2),
                                                                 present_key.stride(2),
                                                                 _fp32_out.stride(1),
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
        }
        _qk_scratch_b.resize<DATA_TYPE>({_B, _Hk, block_tables.size(1), _qk_gemm[_block_size - 1]->get_scratch_b_size() / sizeof(DATA_TYPE)});
        _wv_scratch_b.resize<DATA_TYPE>({_B, _Hk, block_tables.size(1), _wv_gemm[_block_size - 1]->get_scratch_b_size() / sizeof(DATA_TYPE)});
    }

    void execute_brgemm(PlainTensor& query,
                        PlainTensor& present_key,
                        PlainTensor& present_value,
                        PlainTensor& output_emb,
                        const PlainTensor& block_tables,
                        size_t max_context_len,
                        const PlainTensor& context_lens,
                        float d_scale = 0.0f,
                        size_t sliding_window = 0) {
        bool is_bf16 = precision_of<DATA_TYPE>::value == ov::element::bf16;
        PROFILE(_attn, "brgemm_pack");
        // packed k, v
        parallel_for3d_dynamic(_B, _kv_len_in_blocks, _Hk, [&](size_t b, size_t kv_block, size_t hk) {
            auto block_number = block_tables.ptr<int32_t>(b)[kv_block];
            if (block_number < 0)
                return;
            auto* k_ptr = present_key.ptr<DATA_TYPE>(block_number, hk);
            auto* v_ptr = present_value.ptr<DATA_TYPE>(block_number, hk);
            _qk_gemm[_block_size - 1]->copy_buffer_b(k_ptr, _qk_scratch_b.ptr<DATA_TYPE>(b, hk, kv_block));
            if (is_bf16)
                _wv_gemm[_block_size - 1]->copy_buffer_b(v_ptr, _wv_scratch_b.ptr<DATA_TYPE>(b, hk, kv_block));
        });

        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("brgemm_attn");
        // query breaks to [B, H, m_blocks, block_size, S], k cache is split to [B, H, m_blocks', S, block_size]
        // v cache may be [B, H, m_blocks', block_size, S] or [block_number, H, block_size, S]
        // outer loop will use B, H, m_blocks to walkthrough query
        parallel_for3d_dynamic(_B, _kv_len_in_blocks, _Hk, [&](size_t b, size_t m_blk, size_t hk) {
            if (block_tables.ptr<int32_t>(b)[m_blk] < 0)
                return;
            auto cur_kv_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto q_len = cur_kv_len;
            auto m_start = m_blk * _block_size;
            auto m_end = std::min(m_start + _block_size, q_len);
            auto m_cnt = m_end - m_start;
            size_t tid = parallel_get_thread_num();
            for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                auto* q_ptr = query.ptr<DATA_TYPE>(b, h, m_start, 0);
                float* c_ptr = _score.ptr<float>(tid, h, m_start, 0);
                // for each query block, loop through all key block
                for (size_t k_blk = 0; k_blk <= m_blk; k_blk++) {
                    auto* k_ptr = _qk_scratch_b.ptr<DATA_TYPE>(b, hk, k_blk);
                    _qk_gemm[m_cnt - 1]->executeGemm(m_cnt < _block_size,
                                                     q_ptr,
                                                     k_ptr,
                                                     c_ptr + k_blk * _block_size,
                                                     _wsp.data() + tid * _wsp_size_per_thread,
                                                     _qk_scratch_a ? _qk_scratch_a.ptr<DATA_TYPE>(tid, 0) : nullptr);
                }

                for (size_t m = m_start; m < m_end; m++) {
                    // apply attention mask & sofmax
                    auto ncausal = (cur_kv_len - q_len + m + 1);
                    if (sliding_window) {
                        size_t start_idx = 0;
                        auto new_causal = ncausal;
                        if (ncausal > sliding_window) {
                            start_idx = ncausal - static_cast<size_t>(sliding_window);
                            new_causal = sliding_window;
                        }
                        attn_softmax_kernel(_score.ptr<float>(tid, h, m, start_idx),
                                            _weight.ptr<DATA_TYPE>(tid, h, m, start_idx),
                                            d_scale,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            false,
                                            new_causal,
                                            rnd_up(cur_kv_len, _block_size) - start_idx,
                                            precision_of<DATA_TYPE>::value,
                                            precision_of<DATA_TYPE>::value);

                        memset(_weight.ptr<DATA_TYPE>(b, h, m, 0), 0, sizeof(DATA_TYPE) * start_idx);
                    } else {
                        attn_softmax_kernel(_score.ptr<float>(tid, h, m, 0),
                                            _weight.ptr<DATA_TYPE>(tid, h, m, 0),
                                            d_scale,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            false,
                                            ncausal,
                                            rnd_up(cur_kv_len, _block_size),
                                            precision_of<DATA_TYPE>::value,
                                            precision_of<DATA_TYPE>::value);
                    }
                }

                auto* w_ptr = _weight.ptr<DATA_TYPE>(tid, h, m_start, 0);
                float* fp32_out_ptr = is_bf16 ? _fp32_out.ptr<float>(tid, m_start, h, 0) : output_emb.ptr<float>(b, m_start, h * _S);

                // for each weight block, loop through all value block
                for (size_t v_blk = 0; v_blk <= m_blk; v_blk++) {
                    DATA_TYPE* v_ptr;
                    if (is_bf16) {
                        v_ptr = _wv_scratch_b.ptr<DATA_TYPE>(b, hk, v_blk);
                    } else {
                        v_ptr = present_value.ptr<DATA_TYPE>(block_tables.ptr<int32_t>(b)[v_blk], hk);
                    }
                    if (v_blk == 0) {
                        _wv_gemm[m_cnt - 1]->executeGemm(m_cnt < _block_size,
                                                         w_ptr + v_blk * _block_size,
                                                         v_ptr,
                                                         fp32_out_ptr,
                                                         _wsp.data() + tid * _wsp_size_per_thread,
                                                         _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(tid, 0) : nullptr);
                    } else {
                        _wv_gemm_acc[m_cnt - 1]->executeGemm(m_cnt < _block_size,
                                                             w_ptr + v_blk * _block_size,
                                                             v_ptr,
                                                             fp32_out_ptr,
                                                             _wsp.data() + tid * _wsp_size_per_thread,
                                                             _wv_scratch_a ? _wv_scratch_a.ptr<DATA_TYPE>(tid, 0) : nullptr);
                    }
                }
                if (is_bf16) {
                    attn_memcpy2d_kernel(_fp32_out.ptr<float>(tid, m_start, h, 0),
                                         output_emb.ptr<DATA_TYPE>(b, m_start, h * _S),
                                         ov::element::f32,
                                         ov::element::bf16,
                                         _fp32_out.stride(1),
                                         output_emb.stride(1),
                                         _S,
                                         m_cnt);
                }
            }
        });
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // attention_mask [B, 1, q_len, kv_len]
    // alibi          [B, H, q_len, kv_len]
    // output_emb    [B, L1, H*S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    const PlainTensor& block_tables,
                    size_t max_context_len,
                    const PlainTensor& context_lens,
                    float d_scale = 0.0f,
                    size_t sliding_window = 0) {
        PROFILE(_attn, "prepare_prim");
        _B = query.size(0);
        _H = query.size(1);
        _q_len = query.size(2);
        _S = query.size(3);
        _Hk = present_value.size(1);
        _h_each_group_len = 1;
        _block_size = present_value.size(2);
        _kv_len_in_blocks = block_tables.m_dims[1];
        if (_Hk != _H) {
            _h_each_group_len = _H / _Hk;
        }
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(_S);

        _nthr = static_cast<size_t>(parallel_get_max_threads());

        prepare_brgemm_prim(query, present_key, block_tables);
        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("exec_qk");
        execute_brgemm(query,
                       present_key,
                       present_value,
                       output_emb,
                       block_tables,
                       max_context_len,
                       context_lens,
                       d_scale,
                       sliding_window);
    }
};

// 2nd token case : only 1 token in query
template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct MHASingleToken {
    PlainTensor _attn_weight;
    PlainTensor _attn_output;
    std::shared_ptr<JitMatMulVecAMX> _gemv;
    bool _init = false;
    bool _fastpath_valid = false;

    size_t _B;
    size_t _H;
    size_t _q_len;
    size_t _S;
    size_t _Hk;
    size_t _h_each_group_len;
    size_t _block_size;
    size_t _nthr;
    size_t _kv_len_in_blocks;

    void init(size_t block_size, size_t S) {
        _init = true;
        _fastpath_valid = dnnl::impl::cpu::x64::mayiuse(dnnl::impl::cpu::x64::amx_bf16) &&
            (S % 32 == 0) && (block_size % 16 == 0) && (S <= 32 * 6) && precision_of<KVCACHE_TYPE>::value == ov::element::bf16;
        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        if (_fastpath_valid)
            _gemv = std::make_shared<JitMatMulVecAMX>(static_cast<int>(S), static_cast<int>(block_size));
    }

    // one loop along batch and head dimensions
    void loop_bh(PlainTensor& query,
                 PlainTensor& present_key,
                 PlainTensor& present_value,
                 PlainTensor& output_emb,
                 const PlainTensor& block_tables,
                 size_t max_context_len,
                 const PlainTensor& context_lens,
                 float d_scale) {
        _attn_weight.resize<float>({static_cast<size_t>(_nthr), _q_len, _H, rnd_up(max_context_len, _block_size)});
        _attn_output.resize<float>({static_cast<size_t>(_nthr), _q_len, _H, _S});

        parallel_for2d_dynamic(_B, _Hk, [&](size_t b, size_t hk) {
            auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto ithr = parallel_get_thread_num();

            if (ithr % 8 != 0)
                ov::intel_cpu::profilerManagerInstance.enabled = false;
            char name_buf[256];
            snprintf(name_buf, sizeof(name_buf), "t1_qk_%d", ithr);
            PROFILE(_attn, name_buf);
            auto block_table = block_tables.ptr<int32_t>(b);
            if (_fastpath_valid) {
                _gemv->tile_config();
                for (size_t pk = 0; pk < context_len; pk += _block_size) {
                    auto block_number = *block_table++;
                    for (size_t pq = 0; pq < _q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            (*_gemv)(query.ptr<ov::bfloat16>(b, h, pq), present_key.ptr<ov::bfloat16>(block_number, hk),
                                _attn_weight.ptr<float>(ithr, pq, h) + pk);
                        }
                    }
                }
                _gemv->tile_release();
            } else {
                for (size_t pk = 0; pk < context_len; pk += _block_size) {
                    auto block_number = *block_table++;
                    for (size_t pq = 0; pq < _q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            dot_product_block(query.ptr<DATA_TYPE>(b, h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                                _attn_weight.ptr<float>(ithr, pq, h) + pk, _S, std::min(_block_size, context_len - pk));
                        }
                    }
                }
            }

            snprintf(name_buf, sizeof(name_buf), "t1_softmax_%d", ithr);
            _attn = ov::intel_cpu::profilerManagerInstance.startProfile(name_buf);
            for (size_t pq = 0; pq < _q_len; pq++) {
                for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                    // apply attention mask & sofmax
                    attn_softmax_kernel(_attn_weight.ptr<float>(ithr, pq, h),
                                        _attn_weight.ptr<float>(ithr, pq, h),
                                        d_scale,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        false,
                                        context_len,
                                        context_len,
                                        ov::element::f32,
                                        ov::element::f32);
                }
            }

            snprintf(name_buf, sizeof(name_buf), "t1_kv_%d", ithr);
            _attn = ov::intel_cpu::profilerManagerInstance.startProfile(name_buf);
            memset(_attn_output.ptr<float>(ithr), 0, _q_len * _H * _S * sizeof(float));
            block_table = block_tables.ptr<int32_t>(b);
            for (size_t pv = 0; pv < context_len; pv += _block_size) {
                auto block_number = *block_table++;
                auto* v = present_value.ptr<KVCACHE_TYPE>(block_number, hk);
                for (size_t pq = 0; pq < _q_len; pq++) {
                    for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                        attn_acc_value_block(_attn_output.ptr<float>(ithr, pq, h),
                                             _attn_weight.ptr<float>(ithr, pq, h) + pv,
                                             v,
                                             _S,
                                             std::min(_block_size, context_len - pv));
                    }
                }
            }
            // convert to dst
            for (size_t pq = 0; pq < _q_len; pq++)
                for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++)
                    cvt_copy(output_emb.ptr<DATA_TYPE>(b, pq, h * _S), _attn_output.ptr<float>(ithr, pq, h), _S);
        });
    }

    // 3 loops along batch, head, kv cache length dimensions
    void loop_bh_kv(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    const PlainTensor& block_tables,
                    size_t max_context_len,
                    const PlainTensor& context_lens,
                    float d_scale) {
        // aligned to cache line (64bytes=16*sizeof(float)) to avoid false sharing
        _attn_weight.resize<float>({_B, _H, _q_len, rnd_up(max_context_len, std::max(_block_size, 16ul))});

        parallel_for3d_dynamic(_B, _kv_len_in_blocks, _Hk, [&](size_t b, size_t pk_in_blocks, size_t hk) {
            auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            // kv_len must be valid
            auto pk = pk_in_blocks * _block_size;
            if (pk < context_len) {
                auto block_number = block_tables.ptr<int32_t>(b)[pk_in_blocks];
                if (_fastpath_valid) {
                    _gemv->tile_config();
                    for (size_t pq = 0; pq < _q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            (*_gemv)(query.ptr<ov::bfloat16>(b, h, pq), present_key.ptr<ov::bfloat16>(block_number, hk),
                                _attn_weight.ptr<float>(b, h, pq) + pk);
                        }
                    }
                    _gemv->tile_release();
                } else {
                    for (size_t pq = 0; pq < _q_len; pq++) {
                        for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                            dot_product_block(query.ptr<DATA_TYPE>(b, h, pq), present_key.ptr<KVCACHE_TYPE>(block_number, hk),
                                _attn_weight.ptr<float>(b, h, pq) + pk, _S, std::min(_block_size, context_len - pk));
                        }
                    }
                }
            }
        });

        PROFILE(_attn, "t1_softmax");
        parallel_for3d_dynamic(_B, _H, _q_len, [&](size_t b, size_t h, size_t pq) {
            auto cur_kv_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto ncausal = cur_kv_len;
            // apply attention mask & sofmax
            attn_softmax_kernel(_attn_weight.ptr<float>(b, h, pq),
                                _attn_weight.ptr<float>(b, h, pq),
                                d_scale,
                                nullptr,
                                nullptr,
                                nullptr,
                                false,
                                ncausal,
                                cur_kv_len,
                                ov::element::f32,
                                ov::element::f32);
        });

        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("t1_kv");
        // attn_w * V
        _attn_output.resize<float>({static_cast<size_t>(_nthr), _B, _q_len, _H, _S});
        // m_attn_w {B, H, q_len, kv_len}
        parallel_nt_static(_nthr, [&](const size_t ithr, const size_t nthr) {
            memset(_attn_output.ptr<float>(ithr, 0, 0, 0, 0), 0, _attn_output.stride(0) * sizeof(float));
        });

        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("t1_kv_old");
        parallel_for3d_dynamic(_B, _kv_len_in_blocks, _Hk, [&](size_t b, size_t pv_in_blocks, size_t hk) {
            auto ithr = parallel_get_thread_num();
            auto context_len = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            auto pv = pv_in_blocks * _block_size;
            // kv_len must be valid
            if (pv < context_len) {
                auto block_number = block_tables.ptr<int32_t>(b)[pv_in_blocks];
                auto* v = present_value.ptr<KVCACHE_TYPE>(block_number, hk);
                for (size_t pq = 0; pq < _q_len; pq++) {
                    for (size_t h = hk * _h_each_group_len; h < (hk + 1) * _h_each_group_len; h++) {
                        attn_acc_value_block(_attn_output.ptr<float>(ithr, b, pq, h),
                                             _attn_weight.ptr<float>(b, h, pq) + pv,
                                             v,
                                             _S,
                                             std::min(_block_size, context_len - pv));
                    }
                }
            }
        });

        _attn = ov::intel_cpu::profilerManagerInstance.startProfile("t1_reduce");
        parallel_for3d(_B, _H, _q_len, [&](size_t b, size_t h, size_t pq) {
            auto* temp = _attn_output.ptr<float>(0, b, pq, h);
            size_t temp_stride = _attn_output.stride(0);
            auto* dst = output_emb.ptr<DATA_TYPE>(b, pq, h * _S);
            attn_reduce(dst, temp, _nthr, _S, temp_stride);
        });
    }

    // Q, K, V is ready, do attention
    // query         [B, H, q_len, S]
    // present_key   [B, H, kv_len, S]  stride of last dim maybe > 1
    // present_value [B, H, kv_len, S]
    // output_emb    [B, L1, H, S]
    void operator()(PlainTensor& query,
                    PlainTensor& present_key,
                    PlainTensor& present_value,
                    PlainTensor& output_emb,
                    const PlainTensor& block_tables,
                    size_t max_context_len,
                    const PlainTensor& context_lens,
                    float d_scale) {
        _B = query.size(0);
        _H = query.size(1);
        _q_len = query.size(2);
        _S = query.size(3);
        _Hk = present_value.size(1);
        _h_each_group_len = 1;
        _block_size = present_value.size(2);
        _kv_len_in_blocks = block_tables.m_dims[1];
        if (_Hk != _H) {
            _h_each_group_len = _H / _Hk;
        }
        if (d_scale == 0.0f)
            d_scale = 1.0f / sqrt(_S);

        if (!_init)
            init(_block_size, _S);
        _nthr = static_cast<size_t>(parallel_get_max_threads());

        char buf[256];
        size_t real_len = 0;
        size_t aligned_len = 0;
        for (size_t b = 0; b < _B; b++) {
            auto s = static_cast<size_t>(context_lens.ptr<int32_t>()[b]);
            real_len += s;
            aligned_len += (s + 15) / 16 * 16;
        }
        size_t access_size;
        if (present_value.m_dt != ov::element::u8)
            access_size = _Hk * real_len * _S * (32 * present_value.m_element_size);
        else
            access_size = _Hk * real_len * (_S + 8) * 32;
        access_size += _B * _H * _S * 2 * 32;    // query
        access_size += aligned_len * _H * 4 * 32; // m_attn_w
        // only consider k or v theoretical cost
        snprintf(buf, sizeof(buf), "t1_BL%ld,%ld,MC%.2f,%.2f", _B, real_len, access_size / 260000000.0f,
            _H * real_len * _S * 32 / 16 * 2 / 60 / 1800000.0f);
        PROFILE(_attn, buf);

        if (_B >= _nthr) {
            loop_bh(query, present_key, present_value, output_emb, block_tables, max_context_len, context_lens, d_scale);
        } else {
            loop_bh_kv(query, present_key, present_value, output_emb, block_tables, max_context_len, context_lens, d_scale);
        }
    }
};

template <typename DATA_TYPE, typename KVCACHE_TYPE>
struct AttentionExecutor : public PagedAttentionExecutor {
    MHAKernel<DATA_TYPE, KVCACHE_TYPE> kernel;
    MHASingleToken<DATA_TYPE, KVCACHE_TYPE> kernel_single_token;

    void execute(const std::vector<MemoryPtr>& inputs, const MemoryPtr output) override {
        bool is_prompt = false;
        PlainTensor present_key, present_value;
        PlainTensor q_input;           // f32[B, H, L1, S]
        PlainTensor k_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor v_input;           // f32[B, H|1, L1, S] / [B, H|1, L0+L1, S]
        PlainTensor block_tables;      // i32[B, max_kvLen]
        PlainTensor context_lens;
        PlainTensor output_emb(output);
        float scale_input = 0.0f;
        size_t B, L1, S;
        size_t sliding_window = 0;
        size_t max_context_len = 0;

        PROFILE(_attn, "attn_execute");
        q_input.reset(inputs[0]);
        k_input.reset(inputs[1]);
        v_input.reset(inputs[2]);
        present_key.reset(inputs[ID_KCACHE]);
        present_value.reset(inputs[ID_VCACHE]);
        auto block_size = present_key.size(2);

        is_prompt = *inputs[ID_IS_PROMPT]->getDataAs<uint8_t>() == 1;
        max_context_len = static_cast<size_t>(*inputs[ID_MAX_CONTEXT_LEN]->getDataAs<int32_t>());
        context_lens.reset(inputs[ID_CONTEXT_LENS]);
        block_tables.reset(inputs[ID_BLOCK_TABLES]);
        scale_input = *inputs[ID_SCALE]->getDataAs<float>();

        if (q_input.get_precision() == ov::element::bf16 && (block_size % 32 != 0))
            OPENVINO_THROW("CPU: block size must be multiple of 32 when precision is bf16, current: " + std::to_string(block_size));
        else if (block_size % 16 != 0)
            OPENVINO_THROW("CPU: block size must be multiple of 16 when precision is f32, current: " + std::to_string(block_size));

        // q: [B, L1, H*S], kv: [B, L1, Hk*S]
        // k_cache: [NUM_BLOCKS, Hk, 32, S]
        // v_cache: [NUM_BLOCKS, Hk, 32, S]
        // context_lens: [B]
        // block_tables: [B, max_block_per_request]
        B = k_input.size(0);
        L1 = k_input.size(1);
        auto Hk = present_key.size(1);
        // The layout for per token per head for u8 kv cache:
        // |scale(f32)|zeropoint(f32)|quantized feature(u8,idx_1)|quantized feature(u8,idx_2)|...|quantized feature(u8,idx_S)|
        // The actual size needs to deduct scale and zeropoint.
        S = present_value.size(3) - (present_value.m_dt == ov::element::Type_t::u8 ? sizeof(float) * 2 : 0);
        auto H = q_input.size(2) / S;

        q_input.assert_dims({B, L1, H * S});
        if (!is_prompt) {
            context_lens.assert_dims({B});
            block_tables.assert_dims({B, 0}, true);
        } else {
            sliding_window = static_cast<size_t>(*inputs[ID_SLIDING_WINDOW]->getDataAs<int32_t>());
        }
        output_emb.assert_dims({B, L1, H * S});
        q_input = q_input.reshape({B, L1, H, S}).permute({0, 2, 1, 3});
        k_input = k_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});
        v_input = v_input.reshape({B, L1, Hk, S}).permute({0, 2, 1, 3});

        // second token, or first token with pastkv fusing
        if (is_prompt) {
            char buf[256];
            snprintf(buf, sizeof(buf), "first_BL%ld,%ld", B, L1);
            _attn = ov::intel_cpu::profilerManagerInstance.startProfile(buf);

            if (!block_tables) {
                // construct block_tables, max_context_len, context_lens from slot_mapping
                PlainTensor slot_mapping;
                slot_mapping.reset(inputs[ID_SLOT_MAPPING]);    // [B, max_context_len]
                max_context_len = slot_mapping.m_dims[1];
                block_tables.resize<int32_t>({B, div_up(max_context_len, block_size)});
                context_lens.resize<int32_t>({B});
                for (size_t i = 0; i < B; i++) {
                    context_lens.ptr<int32_t>()[i] = 0;
                    for (size_t j = 0; j < block_tables.m_dims[1]; j++) {
                        auto slot = slot_mapping.ptr<int32_t>(i)[j * block_size];
                        block_tables.ptr<int32_t>(i)[j] = slot >= 0 ? slot / block_size : -1;
                        for (size_t k = j * block_size; k < (j + 1) * block_size && k < max_context_len; k++) {
                            if (slot_mapping.ptr<int32_t>(i)[k] < 0)
                                break;
                            context_lens.ptr<int32_t>()[i]++;
                        }
                    }
                }
            }
            // multi-token version
            kernel(q_input, present_key, present_value,
                output_emb, block_tables, max_context_len, context_lens, scale_input, sliding_window);
        } else {
            kernel_single_token(q_input, present_key, present_value,
                output_emb, block_tables, max_context_len, context_lens, scale_input);
        }
    }
};

std::shared_ptr<PagedAttentionExecutor> make_pa_executor(ov::element::Type data_type, ov::element::Type kvcache_type) {
    std::shared_ptr<PagedAttentionExecutor> executor;

    if (data_type == ov::element::bf16) {
        if (kvcache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, uint8_t>>();
        } else {
            executor = std::make_shared<AttentionExecutor<ov::bfloat16, ov::bfloat16>>();
        }
    } else if (data_type == ov::element::f32) {
        if (kvcache_type == ov::element::u8) {
            executor = std::make_shared<AttentionExecutor<float, uint8_t>>();
        } else if (kvcache_type == ov::element::f16) {
            executor = std::make_shared<AttentionExecutor<float, ov::float16>>();
        } else {
            executor = std::make_shared<AttentionExecutor<float, float>>();
        }
    } else {
        OPENVINO_THROW("Unsupported precision: ", data_type);
    }

    return executor;
}

}  // namespace XARCH
}  // namespace Cpu
}  // namespace Extensions
}  // namespace ov