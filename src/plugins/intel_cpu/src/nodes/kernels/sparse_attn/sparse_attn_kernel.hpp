// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

// SIMD (no AMX) products of the query heads of one GQA group with a run of key / value rows. The heads share the K/V
// rows, so each row is converted once for all of them. AVX512 tiles four heads (AVX2: two); leftovers use
// dot_product_block / attn_acc_value_block.
//
//   qk_scores_gqa:     scores[head, r] = q[head] . keys[r]
//   pv_accumulate_gqa: acc[head, :]   += sum_r weights[head, r] * values[r, :]

#include <cstddef>

#include "nodes/kernels/scaled_attn/common.hpp"
#include "nodes/kernels/scaled_attn/paged_attn_kernel.hpp"
#include "utils/plain_tensor.hpp"

namespace ov::Extensions::Cpu::XARCH {

// scores[head * score_stride + r] = q[head, :] . keys[r, :] for head < num_q_heads, r < rows.
// q: f32 [num_q_heads, head_size]; keys: [rows, head_size] contiguous.
template <typename T>
static inline void qk_scores_gqa(const float* q,
                                 size_t num_q_heads,
                                 size_t head_size,
                                 T* keys,
                                 size_t rows,
                                 float* scores,
                                 size_t score_stride) {
    constexpr auto prec = ov::intel_cpu::precision_of<T>::value;
    size_t h = 0;
#if defined(HAVE_AVX512F)
    // 4 heads x 4 rows: 16 independent accumulators, a key vector is loaded and converted once for the 4 heads
    for (; h + 4 <= num_q_heads; h += 4) {
        const float* q0 = q + h * head_size;
        const float* q1 = q0 + head_size;
        const float* q2 = q1 + head_size;
        const float* q3 = q2 + head_size;
        float* s0 = scores + h * score_stride;
        size_t r = 0;
        for (; r + 4 <= rows; r += 4) {
            const T* k0 = keys + r * head_size;
            const T* k1 = k0 + head_size;
            const T* k2 = k1 + head_size;
            const T* k3 = k2 + head_size;
            __m512 acc[4][4];
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    acc[i][j] = _mm512_setzero_ps();
                }
            }
            size_t d = 0;
            for (; d + vec_len_f32_avx512 <= head_size; d += vec_len_f32_avx512) {
                const __m512 key[4] = {mm512_uni_loadu_ps(k0 + d),
                                       mm512_uni_loadu_ps(k1 + d),
                                       mm512_uni_loadu_ps(k2 + d),
                                       mm512_uni_loadu_ps(k3 + d)};
                const __m512 query[4] = {_mm512_loadu_ps(q0 + d),
                                         _mm512_loadu_ps(q1 + d),
                                         _mm512_loadu_ps(q2 + d),
                                         _mm512_loadu_ps(q3 + d)};
                for (size_t i = 0; i < 4; i++) {
                    for (size_t j = 0; j < 4; j++) {
                        acc[i][j] = _mm512_fmadd_ps(key[j], query[i], acc[i][j]);
                    }
                }
            }
            for (size_t i = 0; i < 4; i++) {
                const float* qi = q0 + i * head_size;
                for (size_t j = 0; j < 4; j++) {
                    const T* kj = keys + (r + j) * head_size;
                    float sum = _mm512_reduce_add_ps(acc[i][j]);
                    for (size_t t = d; t < head_size; t++) {
                        sum += qi[t] * static_cast<float>(kj[t]);
                    }
                    s0[i * score_stride + r + j] = sum;
                }
            }
        }
        // the rows left over
        for (size_t i = 0; r < rows && i < 4; i++) {
            dot_product_block<float, prec>(const_cast<float*>(q0) + i * head_size,
                                           keys + r * head_size,
                                           s0 + i * score_stride + r,
                                           head_size,
                                           rows - r,
                                           0);
        }
    }
#elif defined(HAVE_AVX2)
    // 2 heads x 4 rows: 8 independent accumulators (16 registers), a key vector is loaded and converted once for the 2
    // heads
    for (; h + 2 <= num_q_heads; h += 2) {
        const float* q0 = q + h * head_size;
        const float* q1 = q0 + head_size;
        float* s0 = scores + h * score_stride;
        size_t r = 0;
        for (; r + 4 <= rows; r += 4) {
            const T* k0 = keys + r * head_size;
            const T* k1 = k0 + head_size;
            const T* k2 = k1 + head_size;
            const T* k3 = k2 + head_size;
            __m256 acc[2][4];
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    acc[i][j] = _mm256_setzero_ps();
                }
            }
            size_t d = 0;
            for (; d + vec_len_f32_avx2 <= head_size; d += vec_len_f32_avx2) {
                const __m256 key[4] = {mm256_uni_loadu_ps(k0 + d),
                                       mm256_uni_loadu_ps(k1 + d),
                                       mm256_uni_loadu_ps(k2 + d),
                                       mm256_uni_loadu_ps(k3 + d)};
                const __m256 query[2] = {_mm256_loadu_ps(q0 + d), _mm256_loadu_ps(q1 + d)};
                for (size_t i = 0; i < 2; i++) {
                    for (size_t j = 0; j < 4; j++) {
                        acc[i][j] = _mm256_fmadd_ps(key[j], query[i], acc[i][j]);
                    }
                }
            }
            for (size_t i = 0; i < 2; i++) {
                const float* qi = q0 + i * head_size;
                for (size_t j = 0; j < 4; j++) {
                    const T* kj = keys + (r + j) * head_size;
                    hsum(acc[i][j]);
                    float sum = _mm256_cvtss_f32(acc[i][j]);
                    for (size_t t = d; t < head_size; t++) {
                        sum += qi[t] * static_cast<float>(kj[t]);
                    }
                    s0[i * score_stride + r + j] = sum;
                }
            }
        }
        // the rows left over
        for (size_t i = 0; r < rows && i < 2; i++) {
            dot_product_block<float, prec>(const_cast<float*>(q0) + i * head_size,
                                           keys + r * head_size,
                                           s0 + i * score_stride + r,
                                           head_size,
                                           rows - r,
                                           0);
        }
    }
#endif
    // the heads left over, and every head without SIMD (the scalar block kernel)
    for (; h < num_q_heads; h++) {
        dot_product_block<float, prec>(const_cast<float*>(q) + h * head_size,
                                       keys,
                                       scores + h * score_stride,
                                       head_size,
                                       rows,
                                       0);
    }
}

// acc[head, :] += sum_r weights[head * weight_stride + r] * values[r, :] for head < num_q_heads, r < rows.
// acc: f32 [num_q_heads, value_size]; values: [rows, value_size] contiguous.
template <typename T>
static inline void pv_accumulate_gqa(float* acc,
                                     size_t num_q_heads,
                                     size_t value_size,
                                     const float* weights,
                                     size_t weight_stride,
                                     T* values,
                                     size_t rows) {
    constexpr auto prec = ov::intel_cpu::precision_of<T>::value;
    size_t h = 0;
#if defined(HAVE_AVX512F)
    // 4 heads x 4 column vectors: 16 independent accumulators, a value vector is loaded and converted once for the 4
    // heads and a weight is broadcast once for the 4 column vectors
    for (; h + 4 <= num_q_heads; h += 4) {
        float* a0 = acc + h * value_size;
        const float* w0 = weights + h * weight_stride;
        size_t d = 0;
        for (; d + 4 * vec_len_f32_avx512 <= value_size; d += 4 * vec_len_f32_avx512) {
            __m512 a[4][4];
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    a[i][j] = _mm512_loadu_ps(a0 + i * value_size + d + j * vec_len_f32_avx512);
                }
            }
            for (size_t r = 0; r < rows; r++) {
                const T* v = values + r * value_size + d;
                const __m512 value[4] = {mm512_uni_loadu_ps(v),
                                         mm512_uni_loadu_ps(v + vec_len_f32_avx512),
                                         mm512_uni_loadu_ps(v + 2 * vec_len_f32_avx512),
                                         mm512_uni_loadu_ps(v + 3 * vec_len_f32_avx512)};
                for (size_t i = 0; i < 4; i++) {
                    const __m512 w = _mm512_set1_ps(w0[i * weight_stride + r]);
                    for (size_t j = 0; j < 4; j++) {
                        a[i][j] = _mm512_fmadd_ps(w, value[j], a[i][j]);
                    }
                }
            }
            for (size_t i = 0; i < 4; i++) {
                for (size_t j = 0; j < 4; j++) {
                    _mm512_storeu_ps(a0 + i * value_size + d + j * vec_len_f32_avx512, a[i][j]);
                }
            }
        }
        // the columns left over: one vector at a time for the 4 heads, then the last few as scalars
        for (; d + vec_len_f32_avx512 <= value_size; d += vec_len_f32_avx512) {
            __m512 a[4];
            for (size_t i = 0; i < 4; i++) {
                a[i] = _mm512_loadu_ps(a0 + i * value_size + d);
            }
            for (size_t r = 0; r < rows; r++) {
                const __m512 value = mm512_uni_loadu_ps(values + r * value_size + d);
                for (size_t i = 0; i < 4; i++) {
                    a[i] = _mm512_fmadd_ps(_mm512_set1_ps(w0[i * weight_stride + r]), value, a[i]);
                }
            }
            for (size_t i = 0; i < 4; i++) {
                _mm512_storeu_ps(a0 + i * value_size + d, a[i]);
            }
        }
        for (; d < value_size; d++) {
            for (size_t i = 0; i < 4; i++) {
                for (size_t r = 0; r < rows; r++) {
                    a0[i * value_size + d] +=
                        w0[i * weight_stride + r] * static_cast<float>(values[r * value_size + d]);
                }
            }
        }
    }
#elif defined(HAVE_AVX2)
    // 2 heads x 4 column vectors: 8 independent accumulators (16 registers), a value vector is loaded and converted
    // once for the 2 heads and a weight is broadcast once for the 4 column vectors
    for (; h + 2 <= num_q_heads; h += 2) {
        float* a0 = acc + h * value_size;
        const float* w0 = weights + h * weight_stride;
        size_t d = 0;
        for (; d + 4 * vec_len_f32_avx2 <= value_size; d += 4 * vec_len_f32_avx2) {
            __m256 a[2][4];
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    a[i][j] = _mm256_loadu_ps(a0 + i * value_size + d + j * vec_len_f32_avx2);
                }
            }
            for (size_t r = 0; r < rows; r++) {
                const T* v = values + r * value_size + d;
                const __m256 value[4] = {mm256_uni_loadu_ps(v),
                                         mm256_uni_loadu_ps(v + vec_len_f32_avx2),
                                         mm256_uni_loadu_ps(v + 2 * vec_len_f32_avx2),
                                         mm256_uni_loadu_ps(v + 3 * vec_len_f32_avx2)};
                for (size_t i = 0; i < 2; i++) {
                    const __m256 w = _mm256_set1_ps(w0[i * weight_stride + r]);
                    for (size_t j = 0; j < 4; j++) {
                        a[i][j] = _mm256_fmadd_ps(w, value[j], a[i][j]);
                    }
                }
            }
            for (size_t i = 0; i < 2; i++) {
                for (size_t j = 0; j < 4; j++) {
                    _mm256_storeu_ps(a0 + i * value_size + d + j * vec_len_f32_avx2, a[i][j]);
                }
            }
        }
        // the columns left over: one vector at a time for the 2 heads, then the last few as scalars
        for (; d + vec_len_f32_avx2 <= value_size; d += vec_len_f32_avx2) {
            __m256 a[2];
            for (size_t i = 0; i < 2; i++) {
                a[i] = _mm256_loadu_ps(a0 + i * value_size + d);
            }
            for (size_t r = 0; r < rows; r++) {
                const __m256 value = mm256_uni_loadu_ps(values + r * value_size + d);
                for (size_t i = 0; i < 2; i++) {
                    a[i] = _mm256_fmadd_ps(_mm256_set1_ps(w0[i * weight_stride + r]), value, a[i]);
                }
            }
            for (size_t i = 0; i < 2; i++) {
                _mm256_storeu_ps(a0 + i * value_size + d, a[i]);
            }
        }
        for (; d < value_size; d++) {
            for (size_t i = 0; i < 2; i++) {
                for (size_t r = 0; r < rows; r++) {
                    a0[i * value_size + d] +=
                        w0[i * weight_stride + r] * static_cast<float>(values[r * value_size + d]);
                }
            }
        }
    }
#endif
    // the heads left over, and every head without SIMD (the scalar block kernel)
    for (; h < num_q_heads; h++) {
        attn_acc_value_block<T, prec>(acc + h * value_size, weights + h * weight_stride, values, value_size, rows, 0);
    }
}

}  // namespace ov::Extensions::Cpu::XARCH
