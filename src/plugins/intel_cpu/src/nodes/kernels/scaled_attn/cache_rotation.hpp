// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "common.hpp"
#include "openvino/openvino.hpp"

#if defined(HAVE_AVX2) || defined(HAVE_AVX512F)
#    include <immintrin.h>
#endif

#if defined(HAVE_AVX512F)
template <class CT>
inline static void rotate_kv_cache_chunk_avx512(CT* current_x_values_ptr,
                                                CT* current_y_values_ptr,
                                                float* current_rotation_coeffts_cos_ptr,
                                                float* current_rotation_coeffts_sin_ptr,
                                                size_t num_vectorized_elements_per_iteration,
                                                bool is_tail) {
    using namespace ov::Extensions::Cpu::XARCH;

    auto result_x = _mm512_setzero_ps();
    auto result_y = _mm512_setzero_ps();

    auto coeffts_cos = _mm512_undefined_ps();
    auto coeffts_sin = _mm512_undefined_ps();

    auto cache_values_x = _mm512_undefined_ps();
    auto cache_values_y = _mm512_undefined_ps();

    if (!is_tail) {
        coeffts_cos = mm512_uni_loadu_ps(current_rotation_coeffts_cos_ptr);
        coeffts_sin = mm512_uni_loadu_ps(current_rotation_coeffts_sin_ptr);

        cache_values_x = mm512_uni_loadu_ps(current_x_values_ptr);
        cache_values_y = mm512_uni_loadu_ps(current_y_values_ptr);
    } else {
        coeffts_cos = mm512_uni_loadu_tail_ps(current_rotation_coeffts_cos_ptr, num_vectorized_elements_per_iteration);
        coeffts_sin = mm512_uni_loadu_tail_ps(current_rotation_coeffts_sin_ptr, num_vectorized_elements_per_iteration);

        cache_values_x = mm512_uni_loadu_tail_ps(current_x_values_ptr, num_vectorized_elements_per_iteration);
        cache_values_y = mm512_uni_loadu_tail_ps(current_y_values_ptr, num_vectorized_elements_per_iteration);
    }

    result_x = _mm512_fmadd_ps(cache_values_x, coeffts_cos, result_x);
    result_x = _mm512_fnmadd_ps(cache_values_y, coeffts_sin, result_x);  // negative multiply-add

    result_y = _mm512_fmadd_ps(cache_values_x, coeffts_sin, result_y);
    result_y = _mm512_fmadd_ps(cache_values_y, coeffts_cos, result_y);

    if (!is_tail) {
        mm512_uni_storeu_ps(current_x_values_ptr, result_x);
        mm512_uni_storeu_ps(current_y_values_ptr, result_y);
    } else {
        mm512_uni_storeu_tail_ps(current_x_values_ptr, result_x, num_vectorized_elements_per_iteration);
        mm512_uni_storeu_tail_ps(current_y_values_ptr, result_y, num_vectorized_elements_per_iteration);
    }
}
#endif

#if defined(HAVE_AVX2)
template <class CT>
inline static void rotate_kv_cache_chunk_avx2(CT* current_x_values_ptr,
                                              CT* current_y_values_ptr,
                                              float* current_rotation_coeffts_cos_ptr,
                                              float* current_rotation_coeffts_sin_ptr,
                                              size_t num_vectorized_elements_per_iteration,
                                              size_t is_tail) {
    using namespace ov::Extensions::Cpu::XARCH;

    auto result_x = _mm256_setzero_ps();
    auto result_y = _mm256_setzero_ps();

    auto coeffts_cos = _mm256_undefined_ps();
    auto coeffts_sin = _mm256_undefined_ps();

    auto cache_values_x = _mm256_undefined_ps();
    auto cache_values_y = _mm256_undefined_ps();

    if (!is_tail) {
        coeffts_cos = mm256_uni_loadu_ps(current_rotation_coeffts_cos_ptr);
        coeffts_sin = mm256_uni_loadu_ps(current_rotation_coeffts_sin_ptr);

        cache_values_x = mm256_uni_loadu_ps(current_x_values_ptr);
        cache_values_y = mm256_uni_loadu_ps(current_y_values_ptr);
    } else {
        coeffts_cos = mm256_uni_loadu_tail_ps(current_rotation_coeffts_cos_ptr, num_vectorized_elements_per_iteration);
        coeffts_sin = mm256_uni_loadu_tail_ps(current_rotation_coeffts_sin_ptr, num_vectorized_elements_per_iteration);

        cache_values_x = mm256_uni_loadu_tail_ps(current_x_values_ptr, num_vectorized_elements_per_iteration);
        cache_values_y = mm256_uni_loadu_tail_ps(current_y_values_ptr, num_vectorized_elements_per_iteration);
    }

    result_x = _mm256_fmadd_ps(cache_values_x, coeffts_cos, result_x);
    result_x = _mm256_fnmadd_ps(cache_values_y, coeffts_sin, result_x);  // negative multiply-add

    result_y = _mm256_fmadd_ps(cache_values_x, coeffts_sin, result_y);
    result_y = _mm256_fmadd_ps(cache_values_y, coeffts_cos, result_y);

    if (!is_tail) {
        mm256_uni_storeu_ps(current_x_values_ptr, result_x);
        mm256_uni_storeu_ps(current_y_values_ptr, result_y);
    } else {
        mm256_uni_storeu_tail_ps(current_x_values_ptr, result_x, num_vectorized_elements_per_iteration);
        mm256_uni_storeu_tail_ps(current_y_values_ptr, result_y, num_vectorized_elements_per_iteration);
    }
}
#endif

template <class CT>
inline static void rotate_kv_cache_block_opt(CT* cache_block_ptr,
                                             float* block_rotation_coefficients_ptr,
                                             size_t num_heads,
                                             size_t block_size,
                                             size_t embedding_size) {
#if !defined(HAVE_AVX2) && !defined(HAVE_AVX512F)
    OPENVINO_THROW("host CPU must support either AVX2 or AVX512 instructions");
#else
    bool is_tail = false;

#    if defined(HAVE_AVX512F)
    constexpr size_t vec_len_in_f32_elts = ov::Extensions::Cpu::XARCH::vec_len_f32_avx512;
#    else   // HAVE_AVX2
    constexpr size_t vec_len_in_f32_elts = ov::Extensions::Cpu::XARCH::vec_len_f32_avx2;
#    endif  // defined(HAVE_AVX512F)

    size_t num_processed_elements_per_iteration =
        2 * vec_len_in_f32_elts;  // implementations act on pairs of cache values at once using separate registers, each
                                  // elt is expanded to f32 on load
    size_t num_iterations = embedding_size / num_processed_elements_per_iteration;

    if (embedding_size >= num_processed_elements_per_iteration) {
        OPENVINO_ASSERT(!(num_processed_elements_per_iteration % vec_len_in_f32_elts));
    } else {
        is_tail = true;
        OPENVINO_ASSERT(!(embedding_size % 2));
        num_processed_elements_per_iteration = embedding_size;
        num_iterations = 1;
    }

    CT* current_cache_element_ptr = cache_block_ptr;

    for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        // the rotation coefficients are taken to be the same for all heads
        float* current_rotation_coeffts_ptr = block_rotation_coefficients_ptr;
        for (size_t tok_idx = 0; tok_idx < block_size;
             tok_idx++, current_cache_element_ptr += embedding_size, current_rotation_coeffts_ptr += embedding_size) {
            CT* current_x_values_ptr = current_cache_element_ptr;
            CT* current_y_values_ptr = current_cache_element_ptr + embedding_size / 2;

            float* current_rotation_coeffts_cos_ptr = current_rotation_coeffts_ptr;
            float* current_rotation_coeffts_sin_ptr = current_rotation_coeffts_ptr + embedding_size / 2;

            for (size_t iter_idx = 0; iter_idx < num_iterations; iter_idx++,
                        current_x_values_ptr += vec_len_in_f32_elts,
                        current_y_values_ptr += vec_len_in_f32_elts,
                        current_rotation_coeffts_cos_ptr += vec_len_in_f32_elts,
                        current_rotation_coeffts_sin_ptr += vec_len_in_f32_elts) {
#    if defined(HAVE_AVX512F)
                rotate_kv_cache_chunk_avx512(current_x_values_ptr,
                                             current_y_values_ptr,
                                             current_rotation_coeffts_cos_ptr,
                                             current_rotation_coeffts_sin_ptr,
                                             num_processed_elements_per_iteration / 2,
                                             is_tail);
#    else   // HAVE_AVX2
                rotate_kv_cache_chunk_avx2(current_x_values_ptr,
                                           current_y_values_ptr,
                                           current_rotation_coeffts_cos_ptr,
                                           current_rotation_coeffts_sin_ptr,
                                           num_processed_elements_per_iteration / 2,
                                           is_tail);
#    endif  // defined(HAVE_AVX512F)
            }
        }
    }
#endif      // !defined(HAVE_AVX512F) && !defined(HAVE_AVX2F)
}

template <class CT>
inline static void rotate_kv_cache_block_ref(CT* cache_block_ptr,
                                             float* block_rotation_coefficients_ptr,
                                             size_t num_heads,
                                             size_t block_size,
                                             size_t embedding_size) {
    for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        for (size_t tok_idx = 0; tok_idx < block_size; tok_idx++) {
            size_t token_offset = embedding_size * tok_idx;
            CT* token_embedding_data_start_in_cache =
                cache_block_ptr + head_idx * embedding_size * block_size + embedding_size * tok_idx;
            float* token_data_start_in_rotation_coefficients = block_rotation_coefficients_ptr + token_offset;
            for (size_t embedding_pair_idx = 0; embedding_pair_idx < embedding_size / 2; embedding_pair_idx++) {
                // NB: below is the llama-style rotation (x-like values are in the first half of the embedding vector,
                // y-like values are in the second half), which is different from the original RoFormer style (x- and y-
                // values are interleaved), but still preserves the relative positional encoding property
                CT* cache_value_0_ptr = token_embedding_data_start_in_cache + embedding_pair_idx;
                CT* cache_value_1_ptr = cache_value_0_ptr + (embedding_size / 2);

                float rotation_value_cos = token_data_start_in_rotation_coefficients[embedding_pair_idx];
                float rotation_value_sin =
                    token_data_start_in_rotation_coefficients[embedding_pair_idx + (embedding_size / 2)];

                CT cache_value_0 = *cache_value_0_ptr;
                CT cache_value_1 = *cache_value_1_ptr;

                *cache_value_0_ptr = cache_value_0 * rotation_value_cos - cache_value_1 * rotation_value_sin;
                *cache_value_1_ptr = cache_value_0 * rotation_value_sin + cache_value_1 * rotation_value_cos;
            }
        }
    }
}

template <class CT>
inline static void rotate_kv_cache_block(CT* cache_block_ptr,
                                         float* block_rotation_coefficients_ptr,
                                         size_t num_heads,
                                         size_t block_size,
                                         size_t embedding_size) {
#if defined(HAVE_AVX512F) || defined(HAVE_AVX2)
    rotate_kv_cache_block_opt(cache_block_ptr, block_rotation_coefficients_ptr, num_heads, block_size, embedding_size);
#else
    rotate_kv_cache_block_ref(cache_block_ptr, block_rotation_coefficients_ptr, num_heads, block_size, embedding_size);
#endif  // defined(HAVE_AVX512F) || defined(HAVE_AVX2)
}

template <>
inline void rotate_kv_cache_block(uint8_t* cache_block_ptr,
                                  float* block_rotation_coefficients_ptr,
                                  size_t num_heads,
                                  size_t block_size,
                                  size_t embedding_size) {
    OPENVINO_THROW("cache rotation is not implemented for INT8");
}
