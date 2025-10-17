// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/reference/divide.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/softmax.hpp"
#include "openvino/reference/transpose.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::reference {

using XAttentionBlockIndex =
    std::pair<size_t, size_t>;  // .first is the *query* dimension block index, .second is *key*
using XAttentionRetainedBlockIndices = std::set<XAttentionBlockIndex>;
using XAttentionRetainedBlockIndicesForAllHeads = std::vector<XAttentionRetainedBlockIndices>;

/** @brief Reference implementation of the XAttention sparse attention prefill mechanism
 * (https://arxiv.org/abs/2503.16428) */
template <typename T>
class XAttentionBlockSelector {
public:
    /** @param threshold Defines a threshold for introduced block sparsity - XAttention attempts to preserve the
     * smallest subset of causal non-diagonal attention score matrix blocks so that the ratio of their attention score
     * sum to the total sum of causal non-diagonal attention score matrix blocks in the same K-row is no less than
     * `threshold`. In other words, `threshold` defines a fraction of the block non-diagonal causal attention score mass
     * which is to be preserved by most "important" blocks. Valid range is 0.0-1.0, with 0.0 corresponding to 0% of the
     * non-diagonal causal blocks retained, and 1.0 corresponding to 100% of the non-diagonal causal blocks retained.
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
    XAttentionBlockSelector(double threshold, size_t block_size, size_t stride)
        : m_threshold(threshold),
          m_block_size(block_size),
          m_stride(stride) {
        OPENVINO_ASSERT(m_block_size % m_stride == 0);
    }

    /** Assuming the input tensor is either a query tensor or key tensor, reshapes it in a diagonal or antidiagonal
     * fashion as appropriate so that the resulting matrices could be used to compute the block-antidiagonal subset of
     * the attention matrix in further operations. For the query tensor, the antidiagonal reshaping should be applied,
     * and diagonal - for the key tensor. Note that for the diagonal reshaping the data layout is effectively unchanged
     * and only the shape can be adjusted in the efficient implementation of the same operation in HW.
     * @param input_data Pointer to the input tensor data (query or key)
     * @param input_shape Shape of the input tensor data (query or key). Expected shape is [num_heads, num_tokens,
     * head_size], where `num_tokens` must be a multiple of `stride`.
     * @param output_data Pointer to the output tensor data (reshaped query or key storage)
     * @param out_shape Shape of the output tensor data. Expected shape is [num_heads, num_tokens / stride, head_size *
     * stride]
     * @param is_antidiagonal Whether to reshape antidiagonally (true) or diagonally (false). Use `true` for query
     * tensor and `false` for key tensor.
     */
    void diagonal_reshape(const T* input_data,
                          const Shape& input_shape,
                          T* output_data,
                          const Shape& out_shape,
                          bool is_antidiagonal) {
        OPENVINO_ASSERT(input_shape.size() == 3);  // [num_heads, num_tokens, head_size]
        OPENVINO_ASSERT(out_shape.size() == 3);
        OPENVINO_ASSERT(input_shape[0] == out_shape[0]);
        OPENVINO_ASSERT(input_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(input_shape[1] / m_stride == out_shape[1]);
        OPENVINO_ASSERT(input_shape[2] * m_stride == out_shape[2]);

        size_t num_elts_in_strided_slice = input_shape[1] / m_stride;
        for (size_t head_idx = 0; head_idx < input_shape[0]; head_idx++) {
            size_t head_offset = head_idx * input_shape[1] * input_shape[2];
            for (size_t stride_num = 0; stride_num < m_stride; stride_num++) {
                for (size_t intra_slice_step = 0; intra_slice_step < num_elts_in_strided_slice; intra_slice_step++) {
                    size_t input_offset = head_offset;
                    size_t output_offset = head_offset + intra_slice_step * out_shape[2] + stride_num * input_shape[2];
                    if (is_antidiagonal) {
                        input_offset += (m_stride - 1 - stride_num + intra_slice_step * m_stride) * input_shape[2];
                    } else {
                        input_offset += (stride_num + intra_slice_step * m_stride) * input_shape[2];
                    }
                    std::memcpy(output_data + output_offset, input_data + input_offset, input_shape[2] * sizeof(T));
                }
            }
        }
    }

    /** Performs a matrix multiplication on the input tensors Q and K and scales the result in a typical attention op
     * fashion, i.e. Q @ K^T / (sqrt(D) * S). Additionally rescales by the stride value, as compared to the regular
     * attention.
     * @param reshaped_query_data Pointer to the reshaped query input.
     * @param reshaped_key_data Pointer to the reshaped key input.
     * @param reshaped_query_shape Shape of the reshaped query input data. Expected shape is [num_heads,
     * num_query_tokens / stride, head_size * stride].
     * @param reshaped_key_shape Shape of the reshaped key input data. Expected shape is [num_heads, num_key_tokens /
     * stride, head_size * stride].
     * @param out Pointer to the output tensor data (attention logit scores)
     * @param out_shape Shape of the output tensor data. Expected shape is [num_heads, num_query_tokens / stride,
     * num_key_tokens / stride]
     */
    void transpose_matmul_scale(const T* reshaped_query_data,
                                const T* reshaped_key_data,
                                const Shape& reshaped_query_shape,
                                const Shape& reshaped_key_shape,
                                T* out,
                                const Shape& out_shape) {
        OPENVINO_ASSERT(reshaped_key_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_query_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_query_shape[0] == reshaped_key_shape[0]);
        OPENVINO_ASSERT(reshaped_query_shape[2] == reshaped_key_shape[2]);

        OPENVINO_ASSERT(out_shape.size() == 3);
        OPENVINO_ASSERT(out_shape[0] == reshaped_query_shape[0]);
        OPENVINO_ASSERT(out_shape[1] == reshaped_query_shape[1]);
        OPENVINO_ASSERT(out_shape[2] == reshaped_key_shape[1]);

        ov::reference::matmul(reshaped_query_data,
                              reshaped_key_data,
                              out,
                              reshaped_query_shape,
                              reshaped_key_shape,
                              out_shape,
                              /* transpose_arg0 = */ false,
                              /* transpose_arg1 = */ true);

        size_t out_size = out_shape[0] * out_shape[1] * out_shape[2];

        for (size_t i = 0; i < out_size; i++) {
            // The D in the formula above refers to the original head dimension, while
            // reshaped_query_shape[2] had been scaled in the process of reshaping, therefore
            // the formula is also adjusted:
            out[i] = out[i] / std::sqrt(reshaped_query_shape[2] * m_stride);
        }
    }

    /** Applies the softmax causal mask along the last two dimensions of the rank-3 input tensor in-place.
     * @param in_out_data Pointer to the softmax input values (logits).
     * @param in_out_shape Shape of the input tensor. Expected shape is [num_heads, num_query_tokens /
     * stride, num_key_tokens / stride].
     */
    void apply_causal_mask_(T* in_out_data, const Shape& in_out_shape) {
        OPENVINO_ASSERT(in_out_shape.size() == 3);
        OPENVINO_ASSERT(in_out_shape[1] <= in_out_shape[2]);
        size_t query_dim = in_out_shape[1];
        size_t key_dim = in_out_shape[2];
        for (size_t head_idx = 0; head_idx < in_out_shape[0]; head_idx++) {
            size_t head_offset = head_idx * in_out_shape[1] * in_out_shape[2];
            for (size_t query_dim_idx = 0; query_dim_idx < in_out_shape[1]; query_dim_idx++) {
                size_t query_dim_offset = query_dim_idx * in_out_shape[2];
                for (size_t key_dim_idx = key_dim - query_dim + query_dim_idx + 1; key_dim_idx < key_dim;
                     key_dim_idx++) {
                    in_out_data[head_offset + query_dim_offset + key_dim_idx] = -INFINITY;
                }
            }
        }
    }

    /** Performs a softmax operation on the last dimension of the rank-3 input tensor.
     * @param reshaped_qk_product_data Pointer to the reshaped query-key product input (attention logits pre-softmax).
     * @param reshaped_qk_product_shape Shape of the input tensor. Expected shape is [num_heads, num_query_tokens /
     * stride, num_key_tokens / stride].
     * @param out Pointer to the output tensor data (attention scores)
     * @param out_shape Shape of the output tensor data. Expected shape is strictly equal to
     * `reshaped_qk_product_shape`.
     */
    void softmax(const T* reshaped_qk_product_data,
                 const Shape& reshaped_qk_product_shape,
                 T* out,
                 const Shape& out_shape) {
        OPENVINO_ASSERT(reshaped_qk_product_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_qk_product_shape == out_shape);
        ov::reference::softmax(reshaped_qk_product_data, out, reshaped_qk_product_shape, {2});
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
    void block_sum_attention_scores(const T* attention_scores_data,
                                    const Shape& attention_scores_shape,
                                    T* out,
                                    const Shape& out_shape) {
        OPENVINO_ASSERT(attention_scores_shape.size() == 3);  // [num_heads, query_antidiagonals, key_antidiagonals]
        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        OPENVINO_ASSERT(attention_scores_shape[1] % antidiagonals_per_xattention_block == 0);
        OPENVINO_ASSERT(attention_scores_shape[2] % antidiagonals_per_xattention_block == 0);

        OPENVINO_ASSERT(out_shape[0] == attention_scores_shape[0]);
        OPENVINO_ASSERT(out_shape[1] ==
                        attention_scores_shape[1] / antidiagonals_per_xattention_block);  // query length, blocked
        OPENVINO_ASSERT(out_shape[2] ==
                        attention_scores_shape[2] / antidiagonals_per_xattention_block);  // key length, blocked

        std::memset(out, 0, out_shape[0] * out_shape[1] * out_shape[2] * sizeof(T));

        for (size_t head_idx = 0; head_idx < attention_scores_shape[0]; head_idx++) {
            size_t in_head_offset = head_idx * attention_scores_shape[1] * attention_scores_shape[2];
            size_t out_head_offset = head_idx * out_shape[1] * out_shape[2];
            for (size_t query_len_idx = 0; query_len_idx < attention_scores_shape[1]; query_len_idx++) {
                for (size_t key_len_idx = 0; key_len_idx < attention_scores_shape[2]; key_len_idx++) {
                    size_t query_block_idx = query_len_idx / antidiagonals_per_xattention_block;
                    size_t key_block_idx = key_len_idx / antidiagonals_per_xattention_block;
                    auto target_block_sum_ptr = out + out_head_offset + query_block_idx * out_shape[2] + key_block_idx;
                    *target_block_sum_ptr += *(attention_scores_data + in_head_offset +
                                               query_len_idx * attention_scores_shape[2] + key_len_idx);
                }
            }
        }
    }

    /** Selects the elements of the input tensor along the last dimension, independently along the first two dimensions,
     * so that the selected elements constitute a smallest subset amounting to a sum portion no less than `threshold`
     * of the total "causal" element sum. "Causal" is understood in the sense of the last two dimensions being
     * treated as the query-block and key-block dimensions in the context of attention matrix scores. The
     * first-in-row, the "diagonal" and "non-causal" elements are disregarded when calculating the sum. "Non-causal"
     * elements are never preserved, while "diagonal" and first-in-row elements are always preserved.
     * @param blocked_scores_data Pointer to the blocked score input.
     * @param blocked_attention_scores_shape Shape of the blocked score input tensor. Expected shape is [num_heads,
     * num_query_tokens / block_size, num_key_tokens / block_size]
     * @return A vector of size `num_heads` of sets, each set containing pairs of block indices (.first is the block
     * index along the query dimension, .second - along the key). Each set is the head-specific subset of blocks
     * corresponding to the property described above.
     */
    XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
                                                                        const Shape& blocked_attention_scores_shape) {
        OPENVINO_ASSERT(blocked_attention_scores_shape.size() ==
                        3);  // [num_heads, num_blocks_in_query, num_blocks_in_key]
                             //
        OPENVINO_ASSERT(blocked_attention_scores_shape[1] <= blocked_attention_scores_shape[2]);

        auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

        struct IndexAndScore {
            XAttentionBlockIndex idx;
            T score;
            bool operator<(const IndexAndScore& rhs) const {
                return score < rhs.score;
            }
        };

        for (size_t head_idx = 0; head_idx < blocked_attention_scores_shape[0]; head_idx++) {
            size_t head_offset = head_idx * blocked_attention_scores_shape[1] * blocked_attention_scores_shape[2];
            for (size_t q_block_idx = 0; q_block_idx < blocked_attention_scores_shape[1]; q_block_idx++) {
                std::priority_queue<IndexAndScore> indices_and_scores_queue;
                double total_sum = 0.0;
                double cumsum = 0.0;
                for (size_t k_block_idx = 0; k_block_idx < blocked_attention_scores_shape[2]; k_block_idx++) {
                    if (k_block_idx >
                        (blocked_attention_scores_shape[2] - blocked_attention_scores_shape[1] + q_block_idx)) {
                        // Disregard non-causal blocks entirely
                        continue;
                    }
                    size_t target_offset = head_offset + blocked_attention_scores_shape[2] * q_block_idx + k_block_idx;
                    T current_score = *(blocked_attention_scores_data + target_offset);
                    total_sum += current_score;

                    if ((k_block_idx ==
                         (blocked_attention_scores_shape[2] - blocked_attention_scores_shape[1] + q_block_idx)) ||
                        k_block_idx == 0) {
                        // We preserve first-in-row and diagonal blocks always, and include their score in the
                        // cumulative sum. The target for the rest of the blocks in row is to fill up the
                        // rest of the attention mass fraction so that with the diagonal and first blocks they
                        // comprise the `threshold` portion of the entire causal attention mass in this row
                        retval[head_idx].insert({q_block_idx, k_block_idx});
                        cumsum += current_score;
                    } else {
                        indices_and_scores_queue.push({{q_block_idx, k_block_idx}, current_score});
                    }
                }
                double required_sum = m_threshold * total_sum;
                while (cumsum < required_sum && !indices_and_scores_queue.empty()) {
                    auto index_and_largest_score = indices_and_scores_queue.top();
                    indices_and_scores_queue.pop();
                    cumsum += index_and_largest_score.score;
                    retval[head_idx].insert(index_and_largest_score.idx);
                }
            }
        }
        return retval;
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
    XAttentionRetainedBlockIndicesForAllHeads select_blocks(const T* query_data,
                                                            const Shape& query_shape,
                                                            const T* key_data,
                                                            const Shape& key_shape) {
        OPENVINO_ASSERT(query_shape.size() == 3);  // [num_heads, query_token_len, head_dim]
        OPENVINO_ASSERT(key_shape.size() == 3);    // [num_heads, key_token_len, head_dim]

        OPENVINO_ASSERT(key_shape[0] == query_shape[0]);
        OPENVINO_ASSERT(key_shape[2] == query_shape[2]);

        OPENVINO_ASSERT(query_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(key_shape[1] % m_stride == 0);

        OPENVINO_ASSERT(query_shape[1] % m_block_size == 0);
        OPENVINO_ASSERT(key_shape[1] % m_block_size == 0);

        Shape reshaped_query_shape = {query_shape[0], query_shape[1] / m_stride, query_shape[2] * m_stride};
        auto q_buf = allocate_buf(reshaped_query_shape);
        diagonal_reshape(query_data, query_shape, q_buf.get(), reshaped_query_shape, /* is_antidiagonal = */ true);

        Shape reshaped_key_shape = {key_shape[0], key_shape[1] / m_stride, key_shape[2] * m_stride};
        auto k_buf = allocate_buf(reshaped_key_shape);
        diagonal_reshape(key_data, key_shape, k_buf.get(), reshaped_key_shape, /* is_antidiagonal = */ false);

        Shape transpose_matmul_scaled_shape = {key_shape[0], query_shape[1] / m_stride, key_shape[1] / m_stride};
        auto qk_buf = allocate_buf(transpose_matmul_scaled_shape);
        transpose_matmul_scale(q_buf.get(),
                               k_buf.get(),
                               reshaped_query_shape,
                               reshaped_key_shape,
                               qk_buf.get(),
                               transpose_matmul_scaled_shape);
        q_buf.reset();
        k_buf.reset();

        apply_causal_mask_(qk_buf.get(), transpose_matmul_scaled_shape);

        Shape attention_scores_shape = transpose_matmul_scaled_shape;
        auto attn_score_buf = allocate_buf(attention_scores_shape);
        softmax(qk_buf.get(), transpose_matmul_scaled_shape, attn_score_buf.get(), attention_scores_shape);
        qk_buf.reset();

        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        Shape block_sum_shape = {attention_scores_shape[0],
                                 attention_scores_shape[1] / antidiagonals_per_xattention_block,
                                 attention_scores_shape[2] / antidiagonals_per_xattention_block};
        auto block_sum_buf = allocate_buf(block_sum_shape);
        block_sum_attention_scores(attn_score_buf.get(), attention_scores_shape, block_sum_buf.get(), block_sum_shape);
        attn_score_buf.reset();

        auto selected_block_indices = get_block_indices_to_keep(block_sum_buf.get(), block_sum_shape);
        block_sum_buf.reset();

        return selected_block_indices;
    }

    /**
     * @param shape Shape of a tensor
     * @return A shared_ptr owning a buffer that can be used to store tensor data for the given shape.
     * */
    std::shared_ptr<T[]> allocate_buf(const Shape& shape) {
        return std::shared_ptr<T[]>(new T[ov::shape_size(shape)]);
    }

    /**
     * @param token_length An integer value
     * @return The closest multiple of `block_size` to `token_length`, rounding up.
     * */
    size_t pad_to_block(size_t token_length) {
        return (token_length + m_block_size - 1) / m_block_size * m_block_size;
    }

    double m_threshold;
    size_t m_block_size;
    size_t m_stride;
};

}  // namespace ov::reference
