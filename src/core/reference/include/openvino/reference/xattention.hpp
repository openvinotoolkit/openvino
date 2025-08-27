// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/runtime/tensor.hpp"
#include "openvino/reference/transpose.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/softmax.hpp"
#include "openvino/reference/divide.hpp"
#include <memory>
#include <queue>

namespace ov::reference {

using XAttentionBlockIndex = std::pair<size_t, size_t>; // .first is the *query* dimension block index, .second is *key*
using XAttentionRetainedBlockIndices = std::set<XAttentionBlockIndex>;
using XAttentionRetainedBlockIndicesForAllHeads = std::vector<XAttentionRetainedBlockIndices>;

template<typename T>
class XAttentionBlockSelector {
public:
    XAttentionBlockSelector(double threshold, size_t block_size, size_t stride): m_threshold(threshold), m_block_size(block_size), m_stride(stride) {
        OPENVINO_ASSERT(m_block_size % m_stride == 0);
    }
    void diagonal_reshape(const T* input_data, const Shape& input_shape, T* output_data, const Shape& out_shape, bool is_antidiagonal) {
        OPENVINO_ASSERT(input_shape.size() == 3); // [num_heads, num_tokens, head_size]
        OPENVINO_ASSERT(out_shape.size() == 3);
        OPENVINO_ASSERT(input_shape[0] == out_shape[0]);
        OPENVINO_ASSERT(input_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(input_shape[1] / m_stride == out_shape[1]);
        OPENVINO_ASSERT(input_shape[2] * m_stride == out_shape[2]);

        size_t num_stride_steps = input_shape[1] / m_stride;
        for (size_t head_idx = 0; head_idx < input_shape[0]; head_idx++) {
            size_t head_offset = head_idx * input_shape[1] * input_shape[2];
            for (size_t slice_idx = 0; slice_idx < m_stride; slice_idx++) {
                for (size_t stride_idx = 0; stride_idx < num_stride_steps; stride_idx++) {
                    size_t input_offset = head_offset;
                    size_t output_offset = head_offset + stride_idx * out_shape[2] + slice_idx * input_shape[2];
                    if (is_antidiagonal) {
                        input_offset += (m_stride - 1 - slice_idx + stride_idx * m_stride) * input_shape[2] ;
                    } else {
                        input_offset += (slice_idx + stride_idx * m_stride) * input_shape[2];
                    }
                    std::memcpy(output_data + output_offset, input_data + input_offset, input_shape[2] * sizeof(T));
                }
            }
        }
    }

    void transpose_matmul_scale(const T* reshaped_query_data, const T* reshaped_key_data, const Shape& reshaped_query_shape, const Shape& reshaped_key_shape, T* out, const Shape& out_shape) {
        OPENVINO_ASSERT(reshaped_key_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_query_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_query_shape[0] == reshaped_key_shape[0]);
        OPENVINO_ASSERT(reshaped_query_shape[2] == reshaped_key_shape[2]);

        OPENVINO_ASSERT(out_shape.size() == 3);
        OPENVINO_ASSERT(out_shape[0] == reshaped_query_shape[0]);
        OPENVINO_ASSERT(out_shape[1] == reshaped_query_shape[1]);
        OPENVINO_ASSERT(out_shape[2] == reshaped_key_shape[1]);

        // ov::Tensor key_transposed(ov::element::from<T>(), {reshaped_query_shape[0], reshaped_query_shape[2], reshaped_query_shape[1]});
        // ov::reference::transpose(reshaped_key_data, key_transposed.data<char>(), reshaped_key_shape, sizeof(T), {0, 2, 1}, key_transposed.get_shape());
        ov::reference::matmul(reshaped_query_data, reshaped_key_data, out, reshaped_query_shape, reshaped_key_shape, out_shape, /* transpose_arg0 = */ false, /* transpose_arg1 = */ true);

        size_t out_size = out_shape[0] * out_shape[1] * out_shape[2];

        for (size_t i = 0; i < out_size; i++) {
            out[i] = out[i] / (std::sqrt(reshaped_query_shape[2]) * m_stride);
        }
    }

    void softmax(const T* reshaped_qk_product_data, const Shape& reshaped_qk_product_shape, T* out, const Shape& out_shape) {
        OPENVINO_ASSERT(reshaped_qk_product_shape.size() == 3);
        OPENVINO_ASSERT(reshaped_qk_product_shape == out_shape);
        ov::reference::softmax(reshaped_qk_product_data, out, reshaped_qk_product_shape, {2});
    }

    void block_sum_attention_scores(const T* attention_scores_data, const Shape& attention_scores_shape, T* out, const Shape& out_shape) {
        OPENVINO_ASSERT(attention_scores_shape.size() == 3);
        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        OPENVINO_ASSERT(attention_scores_shape[1] % antidiagonals_per_xattention_block == 0);
        OPENVINO_ASSERT(attention_scores_shape[2] % antidiagonals_per_xattention_block == 0);

        OPENVINO_ASSERT(out_shape[0] == attention_scores_shape[0]);
        OPENVINO_ASSERT(out_shape[1] == attention_scores_shape[1] / antidiagonals_per_xattention_block); // query length, blocked
        OPENVINO_ASSERT(out_shape[2] == attention_scores_shape[2] / antidiagonals_per_xattention_block); // key length, blocked

        std::memset(out, 0, out_shape[0] * out_shape[1] * out_shape[2]);

        for (size_t head_idx = 0; head_idx < attention_scores_shape[0]; head_idx++) {
            size_t in_head_offset = head_idx * attention_scores_shape[1] * attention_scores_shape[2];
            size_t out_head_offset = head_idx * out_shape[1] * out_shape[2];
            for (size_t query_len_idx = 0; query_len_idx < attention_scores_shape[1]; query_len_idx++) {
                for (size_t key_len_idx = 0; key_len_idx < attention_scores_shape[2]; key_len_idx++) {
                    size_t query_block_idx = query_len_idx / m_block_size;
                    size_t key_block_idx = key_len_idx / m_block_size;
                    auto target_block_sum_ptr = out + out_head_offset + query_block_idx * out_shape[2] + key_block_idx;
                    *target_block_sum_ptr += *(attention_scores_data + in_head_offset + query_len_idx * attention_scores_shape[2] + key_len_idx);
                }
            }
        }
    }


    XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data, const Shape& blocked_attention_scores_shape) {
        OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

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
            std::priority_queue<IndexAndScore> indices_and_scores_queue;
            for (size_t q_block_idx = 0; q_block_idx < blocked_attention_scores_shape[1]; q_block_idx++) {
                for (size_t k_block_idx = 0; k_block_idx < blocked_attention_scores_shape[2]; k_block_idx++) {
                    size_t target_offset = head_offset + blocked_attention_scores_shape[2] * q_block_idx + k_block_idx;
                    indices_and_scores_queue.push({ {q_block_idx, k_block_idx}, *(blocked_attention_scores_data + target_offset) });
                }
            }
            double cumsum = 0.0;
            double total_sum = 0.0;
            while (cumsum < m_threshold && !indices_and_scores_queue.empty()) {
                auto index_and_largest_score = indices_and_scores_queue.top();
                indices_and_scores_queue.pop();
                cumsum += index_and_largest_score.score;
                total_sum += index_and_largest_score.score;
                retval[head_idx].insert(index_and_largest_score.idx);
            }
            while (!indices_and_scores_queue.empty()) {
                auto index_and_largest_score = indices_and_scores_queue.top();
                indices_and_scores_queue.pop();
                total_sum += index_and_largest_score.score;
            }
            std::cout << "VSHAMPOR: head " << head_idx << " cumsum: " << cumsum << " total_sum: " << total_sum << std::endl;
        }
        return retval;
    }

    XAttentionRetainedBlockIndicesForAllHeads select_blocks(const T* query_data, const Shape& query_shape, const T* key_data, const Shape& key_shape) {
        OPENVINO_ASSERT(query_shape.size() == 3); // [num_heads, query_token_len, head_dim]
        OPENVINO_ASSERT(key_shape.size() == 3);   // [num_heads, key_token_len, head_dim]

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
        transpose_matmul_scale(q_buf.get(), k_buf.get(), reshaped_query_shape, reshaped_key_shape, qk_buf.get(), transpose_matmul_scaled_shape);
        q_buf.reset();
        k_buf.reset();

        Shape attention_scores_shape = transpose_matmul_scaled_shape;
        auto attn_score_buf = allocate_buf(attention_scores_shape);
        softmax(qk_buf.get(), transpose_matmul_scaled_shape, attn_score_buf.get(), attention_scores_shape);
        qk_buf.reset();

        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        Shape block_sum_shape = {attention_scores_shape[0], attention_scores_shape[1] / antidiagonals_per_xattention_block, attention_scores_shape[2] / antidiagonals_per_xattention_block};
        auto block_sum_buf = allocate_buf(block_sum_shape);
        auto selected_block_indices = get_block_indices_to_keep(block_sum_buf.get(), block_sum_shape);
        block_sum_buf.reset();

        return selected_block_indices;
    }

    std::shared_ptr<T[]> allocate_buf(const Shape& shape) {
        return std::shared_ptr<T[]>(new T[ov::shape_size(shape)]);
    }

    size_t pad_to_block(size_t token_length) {
        return (token_length + m_block_size - 1) / m_block_size * m_block_size;
    }

    template<typename KEY_TYPE>
    void cpu_gather_key_cache_from_blocks_and_pad(const KEY_TYPE* key_cache_data, const Shape& key_cache_shape, T* out_data, const Shape& out_shape, std::vector<size_t> physical_block_indices, size_t key_length_in_tokens) {
        // num_k_heads and num_q_heads may differ to accomodated grouped-query mechanism
        OPENVINO_ASSERT(key_cache_shape.size() == 4); // [cache_size_in_blocks, num_k_heads, cb_block_size, head_dim]
        OPENVINO_ASSERT(out_shape.size() == 3); // [num_q_heads, pad(key_length_in_tokens, m_block_size), head_dim]

        OPENVINO_ASSERT(out_shape[0] % key_cache_shape[1] == 0);
        OPENVINO_ASSERT(out_shape[1] >= key_length_in_tokens);
        OPENVINO_ASSERT(out_shape[1] % m_block_size == 0);
        OPENVINO_ASSERT(out_shape[2] == key_cache_shape[3]);

        size_t num_query_heads_per_key_head = out_shape[0] / key_cache_shape[1];
        size_t cb_block_size = key_cache_shape[2];

        if (key_length_in_tokens % cb_block_size == 0) {
            OPENVINO_ASSERT(key_length_in_tokens / cb_block_size == physical_block_indices.size());
        } else {
            OPENVINO_ASSERT(key_length_in_tokens / cb_block_size + 1 == physical_block_indices.size());
        }

        for (size_t query_head_idx = 0; query_head_idx < out_shape[0]; query_head_idx++) {
            size_t in_head_offset = (query_head_idx / num_query_heads_per_key_head) * key_cache_shape[2] * key_cache_shape[3];
            size_t out_head_offset = query_head_idx * out_shape[1] * out_shape[2];

            size_t num_tokens_processed = 0;
            size_t out_key_token_len_offset = 0;
            for (auto phys_block_id : physical_block_indices) {
                size_t in_block_offset = phys_block_id * key_cache_shape[1] * key_cache_shape[2] * key_cache_shape[3];
                size_t num_tokens_to_copy = cb_block_size;
                if (key_length_in_tokens - num_tokens_processed < cb_block_size) {
                    num_tokens_to_copy = key_length_in_tokens - num_tokens_processed;
                }
                size_t num_elts_to_copy = num_tokens_to_copy * key_cache_shape[3];
                for (size_t elt_idx = 0; elt_idx < num_elts_to_copy; elt_idx++) {
                    out_data[out_head_offset + out_key_token_len_offset + elt_idx] = static_cast<T>(key_cache_data[in_block_offset + in_head_offset + elt_idx]);
                }
                num_tokens_processed += num_tokens_to_copy;
                out_key_token_len_offset += num_elts_to_copy;
            }
            OPENVINO_ASSERT(num_tokens_processed == key_length_in_tokens);
            if (key_length_in_tokens < out_shape[1]) {
                size_t num_tokens_to_pad = out_shape[1] - key_length_in_tokens;
                std::memset(out_data + out_head_offset + out_key_token_len_offset, 0, sizeof(T) * num_tokens_to_pad * out_shape[2]);
            }
        }
    }

    void cpu_gather_query_and_pad(const T* query_data, const Shape& query_shape, T* out_data, const Shape& out_shape, size_t subsequence_begin, size_t subsequence_length) {
        OPENVINO_ASSERT(query_shape.size() == 4); // [num_tokens_for_all_seqs, num_heads, 1, head_dim]
        OPENVINO_ASSERT(query_shape[2] == 1);

        OPENVINO_ASSERT(out_shape.size() == 3); // [num_heads, pad(num_tokens_for_this_seq, m_block_size), head_dim]
        OPENVINO_ASSERT(query_shape[1] == out_shape[0]);
        OPENVINO_ASSERT(query_shape[3] == out_shape[2]);

        OPENVINO_ASSERT(query_shape[0] >= subsequence_begin + subsequence_length);

        OPENVINO_ASSERT(out_shape[1] >= subsequence_length); // will pad with zeroes if token length is not a multiple of m_block_size
        OPENVINO_ASSERT(out_shape[1] % m_block_size == 0);

        for (size_t head_idx = 0; head_idx < out_shape[0]; head_idx++) {
            size_t out_head_offset = head_idx * out_shape[1] * out_shape[2];
            size_t in_head_offset = head_idx * query_shape[2] * query_shape[3];
            for (size_t token_idx = 0; token_idx < out_shape[1]; token_idx++) {
                size_t in_token_offset = (subsequence_begin + token_idx) * query_shape[1] * query_shape[2] * query_shape[3];
                size_t out_token_offset = token_idx * out_shape[2];
                if (token_idx < subsequence_length) {
                    std::memcpy(out_data + out_head_offset + out_token_offset, query_data + in_token_offset + in_head_offset, sizeof(T) * out_shape[2]);
                } else { std::memset(out_data + out_head_offset + out_token_offset, 0, sizeof(T) * out_shape[2]); }
            }
        }
    }

    double m_threshold;
    size_t m_block_size;
    size_t m_stride;
};

}  // namespace ov::reference
