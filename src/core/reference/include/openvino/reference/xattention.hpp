// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <queue>

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
                          const Shape& output_shape,
                          bool is_antidiagonal) {
        OPENVINO_ASSERT(input_shape.size() == 3);
        OPENVINO_ASSERT(output_shape.size() == 3);
        size_t H = input_shape[0];
        size_t Q_orig = input_shape[1];
        size_t D = input_shape[2];
        size_t Q_new = output_shape[1];

        OPENVINO_ASSERT(Q_orig % m_stride == 0);
        OPENVINO_ASSERT(Q_orig / m_stride == Q_new);

        for (size_t h = 0; h < H; ++h) {
            size_t head_in_offset = h * Q_orig * D;
            size_t head_out_offset = h * Q_new * m_stride * D;

            for (size_t s = 0; s < m_stride; ++s) {
                for (size_t q = 0; q < Q_new; ++q) {
                    size_t in_idx;
                    if (is_antidiagonal) {
                        // Anti-diagonal: (stride - 1 - s + q * stride)
                        in_idx = head_in_offset + (m_stride - 1 - s + q * m_stride) * D;
                    } else {
                        // Normal diagonal: (s + q * stride)
                        in_idx = head_in_offset + (s + q * m_stride) * D;
                    }

                    size_t out_idx = head_out_offset + q * m_stride * D + s * D;
                    std::memcpy(output_data + out_idx, input_data + in_idx, D * sizeof(T));
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

    /** Selects the elements of the input tensor along the last two dimensions, independently along the first dimension,
     * so that the elements constitute a smallest subset constituting a sum portion no less than `threshold` of the
     * total element sum.
     * @param blocked_scores_data Pointer to the blocked score input.
     * @param blocked_attention_scores_shape Shape of the blocked score input tensor. Expected shape is [num_heads,
     * num_query_tokens / block_size, num_key_tokens / block_size]
     * @return A vector of size `num_heads` of sets, each set containing pairs of block indices (.first is the block
     * index along the query dimension, .second - along the key). Each set is the head-specific subset of blocks
     * corresponding to the property described above.
     */
    XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(T* blocked_attention_scores_data,
                                                                        const Shape& blocked_attention_scores_shape) {
        OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3,
                        "Expected shape [num_heads, q_block_num, k_block_num]");

        size_t num_heads = blocked_attention_scores_shape[0];
        size_t q_block_num = blocked_attention_scores_shape[1];
        size_t k_block_num = blocked_attention_scores_shape[2];

        XAttentionRetainedBlockIndicesForAllHeads retval(num_heads);

        std::vector<std::vector<std::vector<bool>>> mask(
            num_heads,
            std::vector<std::vector<bool>>(q_block_num, std::vector<bool>(k_block_num, false)));

        for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
            for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
                size_t diagonal_k = q_block_idx;
                if (diagonal_k < k_block_num) {
                    mask[head_idx][q_block_idx][diagonal_k] = true;
                }
                // Step1: First column reserved
                mask[head_idx][q_block_idx][0] = true;

                // Step2: Create other_values（masked_fill）
                std::vector<std::pair<float, size_t>> other_values;
                for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
                    if (mask[head_idx][q_block_idx][k_block_idx])
                        continue;
                    size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
                    other_values.emplace_back(static_cast<float>(blocked_attention_scores_data[offset]), k_block_idx);
                }

                // Step3: Sort other-values in descending order
                std::sort(other_values.begin(), other_values.end(), [](const auto& a, const auto& b) {
                    return a.first > b.first;
                });

                // Step4: Create cumulative_sum_without_self，cat([0, diagonal_sum, sorted_values[:-1]])
                std::vector<float> sorted_scores;
                sorted_scores.push_back(0.0);
                // diagonal + First column score
                size_t offset_diag = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + diagonal_k;
                float diag_score = static_cast<float>(blocked_attention_scores_data[offset_diag]);
                float first_col_score = 0.0;
                if (diagonal_k != 0) {
                    size_t offset_first = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + 0;
                    first_col_score = static_cast<float>(blocked_attention_scores_data[offset_first]);
                }
                sorted_scores.push_back(diag_score + first_col_score);

                for (auto& p : other_values) {
                    sorted_scores.push_back(p.first);
                }
                if (q_block_idx == 0) {
                    sorted_scores.pop_back();
                }

                // Step5: Calculate cumsum_without_self: cumsum of right-shifted sorted_scores
                std::vector<float> cumsum_without_self(sorted_scores.size(), 0.0);
                float running = 0.0;
                for (size_t i = 0; i < sorted_scores.size(); ++i) {
                    cumsum_without_self[i] = running;
                    running += sorted_scores[i];
                }

                // Step6: Generate required_sum
                size_t offset_row_start = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num;
                float row_sum = 0.0;
                for (size_t k = 0; k < k_block_num; k++) {
                    row_sum += static_cast<float>(blocked_attention_scores_data[offset_row_start + k]);
                }
                float required_sum = row_sum * m_threshold;

                // Step7: Create index_mask
                std::vector<bool> index_mask(cumsum_without_self.size(), false);
                for (size_t i = 0; i < cumsum_without_self.size(); i++) {
                    index_mask[i] = (cumsum_without_self[i] < required_sum);
                }

                // Step8: Create index
                std::vector<size_t> index(index_mask.size(), 0);
                for (size_t i = 0; i < index_mask.size(); i++) {
                    if (index_mask[i]) {
                        if (i == 0)
                            index[i] = 0;
                        else if (i == 1)
                            index[i] = diagonal_k;
                        else if (i - 2 < other_values.size())
                            index[i] = other_values[i - 2].second;
                        else
                            index[i] = 0;
                    }
                }

                // Step9: Get retval
                for (size_t i = 0; i < index.size(); i++) {
                    size_t k_block_idx = index[i];
                    if (index_mask[i] && k_block_idx < k_block_num) {
                        mask[head_idx][q_block_idx][k_block_idx] = true;
                    }
                }
                for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
                    if (mask[head_idx][q_block_idx][k_block_idx])
                        retval[head_idx].insert({q_block_idx, k_block_idx});
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
     * @param chunk_size The length of each chunk used for padding and block computation. Query and key sequences are
     * padded as needed to make their lengths multiples of this chunk size, before reshaping and computing XAttention.
     * @return A vector of size `num_heads` of sets, each set containing pairs of block indices (.first is the block
     * index along the query dimension, .second - along the key). Each set is the head-specific subset of blocks that
     * must be preserved in the sparse attention computation. Indices are given in units of XAttention-specific
     * `block_size` (as configured), which may differ from the block size in the paged attention implementation.
     */
    XAttentionRetainedBlockIndicesForAllHeads select_blocks(const T* query_data,
                                                            const Shape& query_shape,
                                                            const T* key_data,
                                                            const Shape& key_shape,
                                                            int chunk_size = -1) {
        OPENVINO_ASSERT(query_shape.size() == 3 && key_shape.size() == 3);
        OPENVINO_ASSERT(query_shape[0] == key_shape[0] && query_shape[2] == key_shape[2]);
        OPENVINO_ASSERT(query_shape[1] % m_stride == 0 && key_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(query_shape[1] % m_block_size == 0 && key_shape[1] % m_block_size == 0);

        const size_t num_heads = query_shape[0];
        const size_t q_len = query_shape[1];
        const size_t k_len = key_shape[1];
        const size_t head_dim = query_shape[2];
        if (chunk_size == -1) chunk_size = q_len;

        auto pad_seq = [&](const T* src_data, size_t seq_len) {
            size_t num_to_pad = ((seq_len + chunk_size - 1) / chunk_size) * chunk_size - seq_len;
            Shape pad_shape = {num_heads, seq_len + num_to_pad, head_dim};
            auto buf = allocate_buf(pad_shape);

            for (size_t h = 0; h < num_heads; ++h) {
                size_t src_off = h * seq_len * head_dim;
                size_t dst_off = h * (seq_len + num_to_pad) * head_dim;
                std::memcpy(buf.get() + dst_off, src_data + src_off, seq_len * head_dim * sizeof(T));
                if (num_to_pad)
                    std::fill(buf.get() + dst_off + seq_len * head_dim,
                              buf.get() + dst_off + (seq_len + num_to_pad) * head_dim,
                              T(0));
            }
            return std::make_pair(std::move(buf), pad_shape);
        };

        // ======== Pad Query & Key ========
        auto [pad_query_buf, pad_query_shape] = pad_seq(query_data, q_len);
        auto [pad_key_buf, pad_key_shape] = pad_seq(key_data, k_len);

        // ======== Diagonal Reshape ========
        const size_t reshaped_q_len = pad_query_shape[1] / m_stride;
        const size_t reshaped_k_len = pad_key_shape[1] / m_stride;
        Shape q_shape_r = {num_heads, reshaped_q_len, head_dim * m_stride};
        Shape k_shape_r = {num_heads, reshaped_k_len, head_dim * m_stride};

        auto q_buf = allocate_buf(q_shape_r);
        auto k_buf = allocate_buf(k_shape_r);
        diagonal_reshape(pad_query_buf.get(), pad_query_shape, q_buf.get(), q_shape_r, true);
        diagonal_reshape(pad_key_buf.get(), pad_key_shape, k_buf.get(), k_shape_r, false);
        pad_query_buf.reset();
        pad_key_buf.reset();

        // ======== QK^T + scale ========
        Shape qk_shape = {num_heads, reshaped_q_len, reshaped_k_len};
        auto qk_buf = allocate_buf(qk_shape);
        transpose_matmul_scale(q_buf.get(), k_buf.get(), q_shape_r, k_shape_r, qk_buf.get(), qk_shape);
        q_buf.reset();
        k_buf.reset();

        // ======== Causal Mask ========
        auto causal_mask_buf = allocate_buf(qk_shape);
        std::fill(causal_mask_buf.get(), causal_mask_buf.get() + ov::shape_size(qk_shape), T(0));
        const size_t reshaped_chunk_size = q_len / m_stride;
        const size_t k_chunk_num = (k_len + ((k_len + chunk_size - 1) / chunk_size * chunk_size - k_len)) / q_len;
        const size_t k_reshaped_seq_len = pad_key_shape[1] / m_stride;
        const size_t k_reshaped_num_to_pad = pad_key_shape[1] / m_stride - k_len / m_stride;
        const size_t chunk_start = (k_chunk_num - 1) * reshaped_chunk_size;
        const size_t chunk_end = chunk_start + reshaped_chunk_size;
        const T neg_inf = std::numeric_limits<T>::lowest();

        for (size_t h = 0; h < num_heads; ++h) {
            for (size_t q = 0; q < reshaped_chunk_size; ++q) {
                size_t base = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) +
                              q * (reshaped_chunk_size * k_chunk_num);

                for (size_t k = k_reshaped_seq_len - k_reshaped_num_to_pad; k < k_reshaped_seq_len; ++k)
                    causal_mask_buf.get()[base + k] = neg_inf;
                for (size_t k = q + 1; k < reshaped_chunk_size; ++k)
                    causal_mask_buf.get()[base + chunk_start + k] = neg_inf;
                for (size_t k = chunk_end; k < reshaped_chunk_size * k_chunk_num; ++k)
                    causal_mask_buf.get()[base + k] = neg_inf;
            }
        }
        // ======== qk += mask ========
        for (size_t i = 0; i < ov::shape_size(qk_shape); ++i)
            qk_buf.get()[i] += causal_mask_buf.get()[i];
        causal_mask_buf.reset();

        // ======== softmax ========
        auto attn_score_buf = allocate_buf(qk_shape);
        softmax(qk_buf.get(), qk_shape, attn_score_buf.get(), qk_shape);
        qk_buf.reset();

        // ======== block sum + select ========
        const size_t blocks_per_axis = m_block_size / m_stride;
        Shape block_sum_shape = {num_heads, reshaped_q_len / blocks_per_axis, reshaped_k_len / blocks_per_axis};
        auto block_sum_buf = allocate_buf(block_sum_shape);
        block_sum_attention_scores(attn_score_buf.get(), qk_shape, block_sum_buf.get(), block_sum_shape);
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
