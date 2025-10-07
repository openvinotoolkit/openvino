// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>

#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/normalize_l2.hpp"
#include "openvino/reference/reduce_mean.hpp"
#include "openvino/reference/slice.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::reference {


/** @brief Reference implementation of the XAttention sparse attention prefill mechanism
 * (https://arxiv.org/abs/2503.16428) */
template <typename T>
class AdaptiveRKVDiversityCalculator {
public:
    /** @param threshold Defines a threshold for introduced block sparsity - XAttention attempts to preserve the
     * smallest subset of attention score matrix blocks so that the ratio of the attention score sum to the total sum of
     * attention score matrix elements is no less than `threshold`. In other words, `threshold` defines a fraction of
     * the attention score mass which is to be preserved by most "important" blocks. Valid range is 0.0-1.0, with 0.0
     * corresponding to 0% of the blocks retained, and 1.0 corresponding to 100% of the blocks retained.
     * @param block_size The size of blocks into which the attention score matrix [num_heads, query_token_dimension,
     * key_token_dimension] will be subdivided for purposes of determining the subset of the most important blocks
     * according to `threshold`. This subdivision occurs on query and key dimensions of the attention score matrix with
     * the same granularity, i.e. the resulting blocks have equal size on both dimensions. Essentially `block_size`
     * defines the granularity of the eventual sparse attention computations. Must be a multiple of `stride`.
     * @param stride The stride at which the full attention matrix is subsampled in a block-antidiagonal fashion to
     * estimate the block importance. Note that the full attention matrix is not computed, instead the original query
     * and key matrices are reshaped appropriately so that only the necessary elements are computed. Ideally, the
     * computational complexity of the entire block estimation operation is `stride` times lower than the full attention
     * matrix computation.
     * */
    AdaptiveRKVDiversityCalculator(size_t start_size, size_t eviction_size, size_t block_size)
        : m_start_size(start_size),
          m_eviction_size(eviction_size),
          m_block_size(block_size) {
        OPENVINO_ASSERT(start_size % block_size == 0);
        OPENVINO_ASSERT(eviction_size % block_size == 0);
    }

    /** Divides the input rank-3 tensor into blocks along last two dimensions, performs the addition of the values
     * inside each block and outputs each block sum into corresponding positions in the output tensor downsampled along
     * the same dimensions. The output tensor dimensions are such that the query and key token dimensions are
     * downsampled by `block_size` when compared to the *original* query and key tensors.
     * @param attention_scores_data Pointer to the attention score input.
     * @param attention_score_shape Shape of the attention score input tensor. Expected shape is [num_heads,
     * num_query_tokens / stride, num_key_tokens / stride], where `num_query_tokens` and `num_key_tokens` must be
     * multiples of `block_size`.
     * @param out Pointer to the output tensor data (block sums)
     * @param out_shape Shape of the output tensor data. Expected shape is [num_heads, num_query_tokens / block_size,
     * num_key_tokens / block_size].
     */
    void fill_diagonal_(T* in_out,
                        const Shape& in_out_shape,
                        T val) {
        OPENVINO_ASSERT(in_out_shape.size() == 3);  // [num_heads, token_dim, token_dim]
        OPENVINO_ASSERT(in_out_shape[1] == in_out_shape[2]);  // [num_heads, token_dim, token_dim]


        for (size_t head_idx = 0; head_idx < in_out_shape[0]; head_idx++) {
            size_t in_head_offset = head_idx * in_out_shape[1] * in_out_shape[2];
            for (size_t token_dim_idx = 0; token_dim_idx < in_out_shape[1]; token_dim_idx++) {
                size_t diagonal_element_offset = token_dim_idx + token_dim_idx * in_out_shape[1];
                auto diagonal_element_ptr = in_out + in_head_offset + diagonal_element_offset;
                *diagonal_element_ptr = val;
            }
        }
    }

    void fill_low_values_with_zeros_(T* in_out,
                                     const Shape& in_out_shape,
                                     const T* means,
                                     const Shape& means_shape) {
        OPENVINO_ASSERT(in_out_shape.size() == 3);  // [num_heads, token_dim, token_dim]
        OPENVINO_ASSERT(in_out_shape[1] == in_out_shape[2]);
        OPENVINO_ASSERT(means_shape.size() == 2);   // [num_heads, token_dim]
        OPENVINO_ASSERT(means_shape[0] == in_out_shape[0]);
        OPENVINO_ASSERT(means_shape[1] == in_out_shape[1]);

        for (size_t head_idx = 0; head_idx < in_out_shape[0]; head_idx++) {
            size_t in_head_offset = head_idx * in_out_shape[1] * in_out_shape[2];
            size_t means_head_offset = head_idx * means_shape[1];
            for (size_t token_dim_idx = 0; token_dim_idx < in_out_shape[1]; token_dim_idx++) {
                T mean_val = means[means_head_offset + token_dim_idx];
                size_t token_offset = token_dim_idx * in_out_shape[2];
                for (size_t reduced_dim_idx = 0; reduced_dim_idx < in_out_shape[2]; reduced_dim_idx++) {
                    size_t target_offset = in_head_offset + token_offset + reduced_dim_idx;
                    T filled_val = in_out[target_offset];
                    in_out[target_offset] = filled_val >= mean_val ? filled_val : 0.0;
                }
            }
        }
    }

    void block_sum_diversity_values(const T* processed_similarity_token_data,
                                    const Shape& processed_similarity_token_data_shape,
                                    T* out,
                                    const Shape& out_shape) {
        OPENVINO_ASSERT(processed_similarity_token_data_shape.size() == 2);  // [token_dim, token_dim]
        OPENVINO_ASSERT(processed_similarity_token_data_shape[0] == processed_similarity_token_data_shape[1]);
        OPENVINO_ASSERT(processed_similarity_token_data_shape[0] % m_block_size == 0);

        OPENVINO_ASSERT(out_shape.size() == 2);  // [block_dim, token_dim]
        OPENVINO_ASSERT(out_shape[0] == processed_similarity_token_data_shape[0] / m_block_size);
        OPENVINO_ASSERT(out_shape[1] == processed_similarity_token_data_shape[1]);

        std::memset(out, 0, out_shape[0] * out_shape[1] * sizeof(T));

        for (size_t out_block_dim_idx = 0; out_block_dim_idx < out_shape[0]; out_block_dim_idx++) {
            size_t out_block_offset = out_block_dim_idx * out_shape[1];
            for (size_t out_token_dim_idx = 0; out_token_dim_idx < out_shape[1]; out_token_dim_idx++) {
               size_t in_block_offset = (out_block_dim_idx * m_block_size) * out_shape[1];
               for (size_t in_token_in_block_idx = 0; in_token_in_block_idx < m_block_size; in_token_in_block_idx++) {
                  size_t source_offset = in_block_offset + in_token_in_block_idx * processed_similarity_token_data_shape[1] + out_token_dim_idx;
                  out[out_block_offset + out_token_dim_idx] -= processed_similarity_token_data[source_offset];
               }
            }
        }
    }

    /** Applies XAttention to the provided query and key matrices, returning the subset of the most important blocks for
     * each attention head, according to the configured block size and threshold, which are to be preserved in the
     * subsequent sparse attention computation.
     * @param query_data Pointer to the query input tensor data
     * @param query_shape Shape of the query input tensor data. Expected shape is [num_heads, num_query_tokens,
     * head_size], where `num_query_tokens` must be a multiple of both `block_size` and `stride`, padded with zeroes if
     * necessary to do so in the real-world scenario.
     * @param key_data Pointer to the key input tensor data
     * @param key_shape Shape of the key input tensor data. Expected shape is [num_heads, num_key_tokens, head_size],
     * where `num_key_tokens` must be a multiple of both `block_size` and `stride`, padded with zeroes if necessary to
     * do so in the real-world scenario.
     * @return A vector of size `num_heads` of sets, each set containing pairs of block indices (.first is the block
     * index along the query dimension, .second - along the key). Each set is the head-specific subset of blocks that
     * must be preserved in the sparse attention computation. Indices are given in units of XAttention-specific
     * `block_size` (as configured), which may differ from the block size in the paged attention implementation.
     */
    std::vector<std::vector<T>> calculate_block_diversity(const T* key_data,
                                                  const Shape& key_shape) {
        OPENVINO_ASSERT(key_shape.size() == 3);    // [num_heads, key_token_len, head_dim]
        OPENVINO_ASSERT(key_shape[1] >= m_start_size + m_eviction_size);


        auto normalized_key_data_buf = allocate_buf(key_shape);
        // Should be safe to use this in-place
        ov::reference::normalize_l2(key_data, normalized_key_data_buf.get(), key_shape, {2}, std::numeric_limits<float>::epsilon(), ov::op::EpsMode::ADD);

        Shape cos_similar_shape = {key_shape[0], key_shape[1], key_shape[1]};
        auto cos_similar_buf = allocate_buf(cos_similar_shape);
        ov::reference::matmul(normalized_key_data_buf.get(), normalized_key_data_buf.get(), cos_similar_buf.get(), key_shape, key_shape, cos_similar_shape, /* transpose_arg0 = */ false, /* transpose_arg1 = */ true);
        normalized_key_data_buf.reset();

        Shape evictable_subset_shape = {key_shape[0], m_eviction_size, m_eviction_size};
        auto evictable_subset_buf = allocate_buf(evictable_subset_shape);
        // stops?
        ov::reference::slice(reinterpret_cast<char*>(cos_similar_buf.get()), cos_similar_shape, reinterpret_cast<char*>(evictable_subset_buf.get()), evictable_subset_shape, sizeof(T), /* starts = */ {m_start_size, m_start_size}, /* steps = */ {1, 1}, /* axes = */{1, 2});
        cos_similar_buf.reset();

        fill_diagonal_(evictable_subset_buf.get(), evictable_subset_shape, 0.0);

        Shape means_shape = {key_shape[0], m_eviction_size};
        auto means_buf = allocate_buf(means_shape);
        ov::reference::reduce_mean(evictable_subset_buf.get(), means_buf.get(), evictable_subset_shape, {2});

        fill_low_values_with_zeros_(evictable_subset_buf.get(), evictable_subset_shape, means_buf.get(), means_shape);
        means_buf.reset();

        Shape aggregated_token_similarities_shape = {m_eviction_size, m_eviction_size};
        auto aggregated_token_similarities_buf = allocate_buf(aggregated_token_similarities_shape);
        ov::reference::reduce_mean(evictable_subset_buf.get(), aggregated_token_similarities_buf.get(), evictable_subset_shape, {0});
        evictable_subset_buf.reset();

        Shape block_diversity_shape = {m_eviction_size / m_block_size, m_eviction_size};
        auto block_diversity_buf = allocate_buf(block_diversity_shape);
        block_sum_diversity_values(aggregated_token_similarities_buf.get(), aggregated_token_similarities_shape, block_diversity_buf.get(), block_diversity_shape);
        std::vector<std::vector<T>> retval(block_diversity_shape[0], std::vector<T>(block_diversity_shape[1]));
        for (size_t block_idx = 0; block_idx < block_diversity_shape[0]; block_idx++) {
            for (size_t token_idx = 0; token_idx < block_diversity_shape[1]; token_idx++) {
                retval[block_idx][token_idx] = block_diversity_buf[block_idx * block_diversity_shape[1] + token_idx];
            }
        }

        return retval;
    }

    /**
     * @param shape Shape of a tensor
     * @return A shared_ptr owning a buffer that can be used to store tensor data for the given shape.
     * */
    std::shared_ptr<T[]> allocate_buf(const Shape& shape) {
        return std::shared_ptr<T[]>(new T[ov::shape_size(shape)]);
    }


    size_t m_start_size;
    size_t m_eviction_size;
    size_t m_block_size;
};

}  // namespace ov::reference
