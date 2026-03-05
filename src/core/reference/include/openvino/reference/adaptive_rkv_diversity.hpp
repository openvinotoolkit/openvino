// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/normalize_l2.hpp"
#include "openvino/reference/reduce_mean.hpp"
#include "openvino/reference/slice.hpp"

namespace ov::reference {

/** @brief Reference implementation of the Adaptive R-KV token diversity calculation mechanism
 * (https://arxiv.org/pdf/2505.24133v3) */
template <typename T>
class AdaptiveRKVDiversityCalculator {
public:
    /** @param start_size Size, in tokens, of the key cache area that will be ignored for purposes of diversity
     * calculation, starting from the beginning of the token dimension ("start area"). Must be a multiple of
     * `block_size`.
     * @param eviction_size Size, in tokens, from the beginning of the start area, the tokens in which will be
     * considered for purposes of diversity calculation ("eviction area"). The rest of the tokens after the eviction
     * area, if any, are ignored. Must be a multiple of `block_size`.
     * @param block_size Block size of the underlying paged attention implementation. The diversity values will be
     * sum-reduced from per-token values to per-block values based on this number of tokens in a block.
     * */
    AdaptiveRKVDiversityCalculator(size_t start_size, size_t eviction_size, size_t block_size)
        : m_start_size(start_size),
          m_eviction_size(eviction_size),
          m_block_size(block_size) {
        OPENVINO_ASSERT(start_size % block_size == 0);
        OPENVINO_ASSERT(eviction_size % block_size == 0);
    }

    /** Fills the diagonal of each square matrix slice (at ranks 1 and 2, zero-based) of the input rank-3 tensor with
     * a provided value. The operation is done in-place.
     * @param in_out Pointer to the matrix data.
     * @param in_out_shape Shape of the matrix data. Expected shape is [num_heads, token_dim, token_dim].
     * @param val Value to fill in the diagonal positions.
     */
    void fill_diagonal_(T* in_out, const Shape& in_out_shape, T val) {
        OPENVINO_ASSERT(in_out_shape.size() == 3);            // [num_heads, token_dim, token_dim]
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

    /** For a rank-3 tensor, zeroes out the values that are less than the mean of the values of the corresponding slice
     * at rank 2 (zero-based). Ranks 1 and 2 of the input tensor must be equal. Mean values are computed and provided
     * externally. The operation is done in-place.
     * @param in_out Pointer to the tensor data.
     * @param in_out_shape Shape of the tensor data. Expected shape is [num_heads, token_dim, token_dim].
     * @param means Pointer to the tensor data containing the means of each slice of the `in_out` tensor along its rank
     * 2 (zero-based).
     * @param means_shape Shape of the means tensor. Expected shape is [num_heads, token_dim].
     */
    void fill_low_values_with_zeros_(T* in_out, const Shape& in_out_shape, const T* means, const Shape& means_shape) {
        OPENVINO_ASSERT(in_out_shape.size() == 3);  // [num_heads, token_dim, token_dim]
        OPENVINO_ASSERT(in_out_shape[1] == in_out_shape[2]);
        OPENVINO_ASSERT(means_shape.size() == 2);  // [num_heads, token_dim]
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
                    in_out[target_offset] = filled_val >= mean_val ? filled_val : T{0};
                }
            }
        }
    }

    /** For a square matrix, sums each `block_size`-sized group of matrix rows to produce a row in the output matrix.
     * In the overall algorithm context, each summed value represents diversity (the negative of inter-token cosine
     * similarity), where larger absolute values indicate greater diversity.
     * @param in_data Pointer to the matrix data.
     * @param in_shape Shape of the matrix data. Expected shape is [token_dim, token_dim], where token_dim must be a
     * multiple of `block_size`.
     * @param out Pointer to the output matrix data.
     * @param out_shape Shape of the output matrix. Expected shape is [token_dim / block_size, token_dim].
     */
    void block_sum_diversity_values(const T* in_data, const Shape& in_shape, T* out, const Shape& out_shape) {
        OPENVINO_ASSERT(in_shape.size() == 2);  // [token_dim, token_dim]
        OPENVINO_ASSERT(in_shape[0] == in_shape[1]);
        OPENVINO_ASSERT(in_shape[0] % m_block_size == 0);

        OPENVINO_ASSERT(out_shape.size() == 2);  // [block_dim, token_dim]
        OPENVINO_ASSERT(out_shape[0] == in_shape[0] / m_block_size);
        OPENVINO_ASSERT(out_shape[1] == in_shape[1]);

        std::memset(out, 0, out_shape[0] * out_shape[1] * sizeof(T));

        for (size_t out_block_dim_idx = 0; out_block_dim_idx < out_shape[0]; out_block_dim_idx++) {
            size_t out_block_offset = out_block_dim_idx * out_shape[1];
            for (size_t out_token_dim_idx = 0; out_token_dim_idx < out_shape[1]; out_token_dim_idx++) {
                size_t in_block_offset = (out_block_dim_idx * m_block_size) * out_shape[1];
                for (size_t in_token_in_block_idx = 0; in_token_in_block_idx < m_block_size; in_token_in_block_idx++) {
                    size_t source_offset = in_block_offset + in_token_in_block_idx * in_shape[1] + out_token_dim_idx;
                    out[out_block_offset + out_token_dim_idx] -= in_data[source_offset];
                }
            }
        }
    }

    /** Calculates token diversity in the eviction area, partially aggregating the results per-block. The resulting
     * diversity values have the shape of [num_eviction_blocks (== eviction_size / block_size), eviction_size]. Note
     * that the 1-st rank is left unaggregated when compared to the full diversity calculation algorithm. The reason
     * for this is as follows. The final per-block diversity value computation relies on knowing the subset of blocks
     * in the eviction area that will be retained regardless of calculated diversity. This subset must be filtered out
     * from the rank-1 dimension when performing reduce-mean in the original algorithm to get 1 diversity value per
     * block in the eviction area. Due to implementation specifics the paged attention kernel does not know ahead of
     * time which blocks will be "retained" - this information is only available on the openvino.genai level after the
     * PA kernel has executed. Therefore the PA kernel will provide raw per-token values on the rank 1 of the returned
     * diversity value matrix and delegate the final reduce-mean and filtering to the openvino.genai level.
     * @param key_data Pointer to the key cache tensor data
     * @param key_shape Shape of the key input tensor data. Expected shape is [num_heads, num_key_tokens, head_size],
     * where `num_key_tokens` must be no less than `start_size + eviction_size`.
     * @return A rank-2 matrix in the std::vector representation with dimensions [eviction_size / block_size,
     * eviction_size] containing the diversity values. The values are expected to be further mean-reduced along rank 1
     * (zero-based) at the point in time when the subset of blocks to be exclusively retained is known.
     */
    std::vector<std::vector<T>> calculate_block_diversity(const T* key_data, const Shape& key_shape) {
        OPENVINO_ASSERT(key_shape.size() == 3);  // [num_heads, key_token_len, head_dim]
        OPENVINO_ASSERT(key_shape[1] >= m_start_size + m_eviction_size);

        auto normalized_key_data_buf = allocate_buf(key_shape);
        // Should be safe to use this in-place
        ov::reference::normalize_l2(key_data,
                                    normalized_key_data_buf.get(),
                                    key_shape,
                                    {2},
                                    std::numeric_limits<float>::epsilon(),
                                    ov::op::EpsMode::ADD);

        Shape cos_similar_shape = {key_shape[0], key_shape[1], key_shape[1]};
        auto cos_similar_buf = allocate_buf(cos_similar_shape);
        ov::reference::matmul(normalized_key_data_buf.get(),
                              normalized_key_data_buf.get(),
                              cos_similar_buf.get(),
                              key_shape,
                              key_shape,
                              cos_similar_shape,
                              /* transpose_arg0 = */ false,
                              /* transpose_arg1 = */ true);
        normalized_key_data_buf.reset();

        Shape evictable_subset_shape = {key_shape[0], m_eviction_size, m_eviction_size};
        auto evictable_subset_buf = allocate_buf(evictable_subset_shape);
        ov::reference::slice(reinterpret_cast<char*>(cos_similar_buf.get()),
                             cos_similar_shape,
                             reinterpret_cast<char*>(evictable_subset_buf.get()),
                             evictable_subset_shape,
                             sizeof(T),
                             /* starts = */ {static_cast<int64_t>(m_start_size), static_cast<int64_t>(m_start_size)},
                             /* steps = */ {1, 1},
                             /* axes = */ {1, 2});  // stops are defined by output shape
        cos_similar_buf.reset();

        fill_diagonal_(evictable_subset_buf.get(), evictable_subset_shape, 0.0);

        Shape means_shape = {key_shape[0], m_eviction_size};
        auto means_buf = allocate_buf(means_shape);
        ov::reference::reduce_mean(evictable_subset_buf.get(), means_buf.get(), evictable_subset_shape, {2});

        fill_low_values_with_zeros_(evictable_subset_buf.get(), evictable_subset_shape, means_buf.get(), means_shape);
        means_buf.reset();

        Shape aggregated_token_similarities_shape = {m_eviction_size, m_eviction_size};
        auto aggregated_token_similarities_buf = allocate_buf(aggregated_token_similarities_shape);
        ov::reference::reduce_mean(evictable_subset_buf.get(),
                                   aggregated_token_similarities_buf.get(),
                                   evictable_subset_shape,
                                   {0});
        evictable_subset_buf.reset();

        Shape block_diversity_shape = {m_eviction_size / m_block_size, m_eviction_size};
        auto block_diversity_buf = allocate_buf(block_diversity_shape);
        block_sum_diversity_values(aggregated_token_similarities_buf.get(),
                                   aggregated_token_similarities_shape,
                                   block_diversity_buf.get(),
                                   block_diversity_shape);
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
