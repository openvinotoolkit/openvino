// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <queue>

#include "openvino/core/type/element_type_traits.hpp"
#include "openvino/reference/divide.hpp"
#include "openvino/reference/matmul.hpp"
#include "openvino/reference/softmax.hpp"
#include "openvino/reference/transpose.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::reference {

using Shape = std::vector<size_t>;

using XAttentionBlockIndex =
    std::pair<size_t, size_t>;  // .first is the *query* dimension block index, .second is *key*
using XAttentionRetainedBlockIndices = std::set<XAttentionBlockIndex>;
using XAttentionRetainedBlockIndicesForAllHeads = std::vector<XAttentionRetainedBlockIndices>;

/** @brief Reference implementation of the XAttention sparse attention prefill mechanism
 *[](https://arxiv.org/abs/2503.16428) */
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

    void diagonal_reshape(const T* input_data,
                          const Shape& input_shape,
                          T* output_data,
                          const Shape& out_shape,
                          bool is_antidiagonal) {
        OPENVINO_ASSERT(input_shape.size() == 3);
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

void diagonal_reshape_kdb1_no_batch(
    const T* input_data,            // 原始 query buffer
    const std::vector<size_t>& input_shape,  // [H, Q_orig, dim]
    T* output_data,                 // 输出 q_buf
    const std::vector<size_t>& output_shape)
{

    size_t H = input_shape[0];
    size_t Q_orig = input_shape[1];
    size_t dim = input_shape[2];
    size_t Q_new = output_shape[1];


    for (size_t h = 0; h < H; ++h) {
        size_t head_in_offset = h * Q_orig * dim;
        size_t head_out_offset = h * Q_new * m_stride * dim;

        for (size_t s = 0; s < m_stride; ++s) {
            for (size_t q = 0; q < Q_new; ++q) {
                size_t in_idx = head_in_offset + (m_stride - 1 - s + q * m_stride) * dim;
                size_t out_idx = head_out_offset + q * m_stride * dim + s * dim;
                std::memcpy(output_data + out_idx, input_data + in_idx, dim * sizeof(T));
            }
        }
    }
}
    void diagonal_reshape_q(const T* input_data,
                            const Shape& input_shape,
                            T* output_data,
                            const Shape& out_shape,
                            bool is_antidiagonal) {
        size_t B = 1;
        size_t H = input_shape[0];
        int Q = input_shape[1];
        int dim = input_shape[2];
        for (size_t b = 0; b < B; ++b) {
            for (size_t h = 0; h < H; ++h) {
                size_t head_offset_in = b * H * Q * dim + h * Q * dim;
                size_t head_offset_out = b * H * Q * dim * m_stride + h * Q * dim * m_stride;
                for (size_t q = 0; q < Q / m_stride; ++q) {
                    for (size_t s = 0; s < m_stride; ++s) {
                        size_t in_idx = head_offset_in + (Q / m_stride) * s + q;          // 交错取值
                        size_t out_idx = head_offset_out + q * m_stride * dim + s * dim;  // 拼接到最后维度
                        std::memcpy(output_data + out_idx, input_data + in_idx * dim, dim * sizeof(T));
                    }
                }
            }
        }
    }

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
                              false,
                              true);

        size_t out_size = out_shape[0] * out_shape[1] * out_shape[2];

        for (size_t i = 0; i < out_size; i++) {
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

    void block_sum_attention_scores(const T* attention_scores_data,
                                    const Shape& attention_scores_shape,
                                    T* out,
                                    const Shape& out_shape) {
        OPENVINO_ASSERT(attention_scores_shape.size() == 3);
        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        OPENVINO_ASSERT(attention_scores_shape[1] % antidiagonals_per_xattention_block == 0);
        OPENVINO_ASSERT(attention_scores_shape[2] % antidiagonals_per_xattention_block == 0);

        OPENVINO_ASSERT(out_shape[0] == attention_scores_shape[0]);
        OPENVINO_ASSERT(out_shape[1] == attention_scores_shape[1] / antidiagonals_per_xattention_block);
        OPENVINO_ASSERT(out_shape[2] == attention_scores_shape[2] / antidiagonals_per_xattention_block);

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
// template <typename T>
// void print_blocked_attention_scores(const T* data,
//                                     size_t num_heads,
//                                     size_t num_q_blocks,
//                                     size_t num_k_blocks) {
//     std::cout << "blocked_attention_scores shape: ["
//               << num_heads << ", " << num_q_blocks << ", " << num_k_blocks << "]\n";

//     for (size_t h = 0; h < num_heads; ++h) {
//         std::cout << "Head " << h << ":\n";
//         std::cout << std::setw(8) << "";
//         for (size_t k = 0; k < num_k_blocks; ++k) {
//             std::cout << std::setw(12) << ("K" + std::to_string(k));
//         }
//         std::cout << "\n";

//         for (size_t q = 0; q < num_q_blocks; ++q) {
//             std::cout << std::setw(6) << ("Q" + std::to_string(q)) << " ";
//             double row_sum = 0.0;
//             for (size_t k = 0; k < num_k_blocks; ++k) {
//                 size_t idx = h * (num_q_blocks * num_k_blocks) + q * num_k_blocks + k;
//                 double v = static_cast<double>(static_cast<float>(*(data + idx)));
//                 row_sum += v;
//                 std::cout << std::setw(12) << std::fixed << std::setprecision(6) << v;
//             }
//             std::cout << "   sum=" << std::fixed << std::setprecision(6) << row_sum << "\n";
//         }
//         std::cout << std::flush;
//     }
// }
// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(
//     const std::vector<double>& input_tensor,  // flattened [batch, head, q_block_num, k_block_num]
//     size_t batch_size,
//     size_t num_heads,
//     size_t q_block_num,
//     size_t k_block_num,
//     double threshold,
//     size_t block_size,
//     size_t stride,
//     bool causal = true) {

//     XAttentionRetainedBlockIndicesForAllHeads retained_blocks(num_heads);

//     for (size_t b = 0; b < batch_size; ++b) {
//         for (size_t h = 0; h < num_heads; ++h) {
//             auto& retained = retained_blocks[h];
//             const size_t base_offset = ((b * num_heads + h) * q_block_num) * k_block_num;

//             for (size_t q_block_idx = 0; q_block_idx < q_block_num; ++q_block_idx) {
//                 size_t diagonal_k = q_block_idx;
//                 std::vector<std::pair<double, size_t>> others;

//                 // 1. 收集当前 query block 对所有 key block 的分数
//                 double row_sum = 0.0;
//                 for (size_t k_block_idx = 0; k_block_idx < k_block_num; ++k_block_idx) {
//                     double score = input_tensor[base_offset + q_block_idx * k_block_num + k_block_idx];
//                     if (std::isnan(score) || std::isinf(score))
//                         score = 0.0;
//                     row_sum += score;
//                     if (k_block_idx != 0 && k_block_idx != diagonal_k) {
//                         others.emplace_back(score, k_block_idx);
//                     }
//                 }

//                 // Debug: 打印 row_sum 和 q_block_idx
//                 /*
//                 if (h == 0)
//                     std::cout << "[Debug] q=" << q_block_idx
//                               << " row_sum=" << row_sum << " others=" << others.size() << "\n";
//                 */

//                 if (row_sum <= 0.0)
//                     continue;

//                 // 2. 强制保留 (q, 0) 和 diagonal
//                 retained.insert({q_block_idx, 0});
//                 retained.insert({q_block_idx, diagonal_k});

//                 // 3. 按分数降序排列 others
//                 std::sort(others.begin(), others.end(),
//                           [](const auto& a, const auto& b) { return a.first > b.first; });

//                 // 4. 计算累计阈值
//                 double required_sum = threshold * row_sum;
//                 double cumsum = 0.0;

//                 std::priority_queue<IndexAndScore> pq;

//                 // ✅ 修复点：原代码用了 others.size() - 2，导致丢项。应当 push 全部候选。
//                 for (size_t i = 0; i < others.size(); ++i) {
//                     pq.push({others[i].second, others[i].first});
//                 }

//                 // Debug: 打印 top 若干项
//                 /*
//                 if (h == 0 && (q_block_idx == 6 || q_block_idx == 7)) {
//                     std::cout << "[Debug] q=" << q_block_idx << " others(sorted): ";
//                     for (size_t i = 0; i < std::min<size_t>(others.size(), 8); ++i)
//                         std::cout << "(" << others[i].second << "," << std::fixed << std::setprecision(3)
//                                   << others[i].first << ") ";
//                     std::cout << "\n";
//                 }
//                 */

//                 // 5. 从大到小取，直到累计到阈值
//                 while (!pq.empty() && cumsum < required_sum) {
//                     auto top = pq.top();
//                     pq.pop();
//                     cumsum += top.score;
//                     retained.insert({q_block_idx, top.index});
//                 }

//                 // Debug: 打印累计结果
//                 /*
//                 if (h == 0 && (q_block_idx == 6 || q_block_idx == 7)) {
//                     std::cout << "[Debug] q=" << q_block_idx
//                               << " required=" << required_sum
//                               << " cumsum=" << cumsum
//                               << " retained=" << retained.size() << "\n";
//                 }
//                 */

//                 // 6. causal mask：只保留 k <= q
//                 if (causal) {
//                     std::set<std::pair<size_t, size_t>> causal_retained;
//                     for (auto& kv : retained) {
//                         if (kv.second <= kv.first)
//                             causal_retained.insert(kv);
//                     }
//                     retained = std::move(causal_retained);
//                 }
//             }
//         }
//     }

//     return retained_blocks;
// }


//     XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
//                                                                         const Shape& blocked_attention_scores_shape) {
//         OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

//         auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//         struct IndexAndScore {
//             size_t k_block_idx;
//             double score;
//             bool operator<(const IndexAndScore& rhs) const {
//                 return score < rhs.score;
//             }
//         };

//         size_t q_block_num = blocked_attention_scores_shape[1];
//         size_t k_block_num = blocked_attention_scores_shape[2];
//         size_t current_index = k_block_num - q_block_num;

//         for (size_t head_idx = 0; head_idx < blocked_attention_scores_shape[0]; head_idx++) {
//             auto& retained = retval[head_idx];
//             for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
//                 double row_sum = 0.0;
//                 for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
//                     size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                     row_sum += static_cast<double>(blocked_attention_scores_data[offset]);
//                 }

//                 double required_sum = m_threshold * row_sum;
//                 double cumsum = 0.0;
//                 // Force include first
//                 size_t k_block_idx = 0;
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 double score = static_cast<double>(blocked_attention_scores_data[offset]);
//                 cumsum += score;
//                 retained.insert({q_block_idx, k_block_idx});
//                 // Force include diagonal
//                 size_t diagonal_k = current_index + q_block_idx;
//                 offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + diagonal_k;
//                 score = static_cast<double>(blocked_attention_scores_data[offset]);
//                 cumsum += score;
//                 retained.insert({q_block_idx, diagonal_k});
//                 // Others

//                 std::vector<std::pair<double, size_t>> others;
//                 for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
//                     if (k_block_idx == 0 || k_block_idx == diagonal_k)
//                         continue;
//                     offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                     double sc = static_cast<double>(blocked_attention_scores_data[offset]);
//                     others.emplace_back(sc, k_block_idx);
//                 }

//                 std::sort(others.begin(), others.end(), [](const auto& a, const auto& b) {
//                     return a.first > b.first;
//                 });

//                 std::priority_queue<IndexAndScore> indices_and_scores_queue;

//                 for (size_t i = 0; i < others.size() - 2; i++) {
//                     if (i >= others.size())
//                         break;

//                     indices_and_scores_queue.push({others[i].second, others[i].first});
//                 }

//                 while (cumsum < required_sum && !indices_and_scores_queue.empty()) {
//                     auto index_and_largest_score = indices_and_scores_queue.top();

//                     indices_and_scores_queue.pop();

//                     cumsum += index_and_largest_score.score;

//                     retained.insert({q_block_idx, index_and_largest_score.k_block_idx});
//                 }
//             }

//             // Enforce causal

//             auto it = retained.begin();

//             while (it != retained.end()) {
//                 size_t q = it->first;

//                 size_t k = it->second;

//                 if (k >= current_index && (k - current_index) > q) {
//                     it = retained.erase(it);

//                 } else {
//                     ++it;
//                 }
//             }
//         }

//         return retval;
//     }

// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
//                                                                     const Shape& blocked_attention_scores_shape) {
//     OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

//     auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//     size_t num_heads = blocked_attention_scores_shape[0];
//     size_t q_block_num = blocked_attention_scores_shape[1];
//     size_t k_block_num = blocked_attention_scores_shape[2];

//     // keep the same current_index computation as original C++ (matches Python caller behavior)
//     size_t current_index = k_block_num - q_block_num;

//     for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//         auto& retained = retval[head_idx];

//         for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
//             // --- 1) 读一行（q_block_idx）并计算 row_sum
//             std::vector<double> row(k_block_num);
//             double row_sum = 0.0;
//             for (size_t k_block_idx = 0; k_block_idx < k_block_num; ++k_block_idx) {
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 double v = static_cast<double>(blocked_attention_scores_data[offset]);
//                 if (std::isnan(v) || std::isinf(v))
//                     v = 0.0;
//                 row[k_block_idx] = v;
//                 row_sum += v;
//             }

//             double required_sum = m_threshold * row_sum;

//             // --- 2) 构造 forced mask（与 Python 中 mask 一致：k==0 与 diagonal_k）
//             std::vector<char> forced(k_block_num, 0);
//             forced[0] = 1;
//             size_t diagonal_k = current_index + q_block_idx;
//             if (diagonal_k < k_block_num)
//                 forced[diagonal_k] = 1;

//             // --- 3) 计算 forced_sum（就是 torch.where(mask, input_tensor, 0).sum(...)）
//             double forced_sum = 0.0;
//             for (size_t k = 0; k < k_block_num; ++k)
//                 if (forced[k])
//                     forced_sum += row[k];

//             // --- 4) 构造 other_values = masked_fill(mask, 0) 并做降序排序（保留索引）
//             std::vector<std::pair<double, size_t>> other_pairs;  // (value, k_idx)
//             other_pairs.reserve(k_block_num);
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 double val = forced[k] ? 0.0 : row[k];
//                 other_pairs.emplace_back(val, k);
//             }
//             std::sort(other_pairs.begin(), other_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 5) 按 Python: 构造 sorted_values_final = [0, forced_sum, other_pairs[0..-3]] (即 sorted_values[:-2])
//             //      这样 final length == k_block_num（相同长度）
//             std::vector<double> sorted_values_cat;
//             sorted_values_cat.reserve(k_block_num);
//             sorted_values_cat.push_back(0.0);
//             sorted_values_cat.push_back(forced_sum);
//             size_t take = 0;
//             if (k_block_num >= 2) {
//                 // other_pairs.size() == k_block_num
//                 // we need to append other_pairs[0 .. k_block_num-3]  => count = k_block_num - 2
//                 // but slice is other_pairs[:-2] -> indices [0 .. k_block_num-3] (count k_block_num-2)
//                 take = (k_block_num >= 2) ? (k_block_num - 2) : 0;
//             }
//             for (size_t i = 0; i < take; ++i) {
//                 sorted_values_cat.push_back(other_pairs[i].first);
//             }
//             // safety: if for some reason sizes mismatch, pad zeros to reach length k_block_num
//             while (sorted_values_cat.size() < k_block_num)
//                 sorted_values_cat.push_back(0.0);

//             // --- 6) 构造 index_order == argsort(descending) of where(mask, BIG*(1+row), row)
//             std::vector<std::pair<double, size_t>> index_pairs;
//             index_pairs.reserve(k_block_num);
//             const double BIG = 100000.0;  // mirrors Python 100000*(1 + input_tensor)
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 double key = forced[k] ? (BIG * (1.0 + row[k])) : row[k];
//                 index_pairs.emplace_back(key, k);
//             }
//             std::sort(index_pairs.begin(), index_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 7) 计算 cumulative_sum_without_self == cumsum( [0] + sorted_values_cat[0:-1] )
//             //     即 cumsum_before[pos] = sum(sorted_values_cat[0 .. pos-1])
//             std::vector<double> cumsum_before(k_block_num, 0.0);
//             double acc = 0.0;
//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 cumsum_before[pos] = acc;
//                 acc += sorted_values_cat[pos];
//             }

//             // --- 8) 构造 index 掩码： index[pos] = index_pairs[pos].second  if cumsum_before[pos] < required_sum else 0
//             //     然后把 index[pos] 对应的 k 插入 retained（等价于 python 的 fancy assignment）
//             //     先强制包含 (align with original C++)
//             retained.insert({q_block_idx, 0});
//             if (diagonal_k < k_block_num)
//                 retained.insert({q_block_idx, diagonal_k});

//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 if (cumsum_before[pos] < required_sum) {
//                     size_t sel_k = index_pairs[pos].second;
//                     retained.insert({q_block_idx, sel_k});
//                 } else {
//                     // python uses 0 where mask false; but we already inserted 0 above
//                 }
//             }

//             // --- Note: we intentionally do NOT add any ad-hoc "neighbor extension" here.
//             //     The above faithfully reproduces Python's selection (including the "[:-2]" trimming).
//             //     Debug printing (commented):
//             if (head_idx == 0 && (q_block_idx == 6 || q_block_idx == 7)) {
//                 std::cout << "[DBG] q=" << q_block_idx
//                           << " row_sum=" << row_sum
//                           << " required=" << required_sum
//                           << " forced_sum=" << forced_sum
//                           << " cumsum_before(last)=" << cumsum_before.back()
//                           << " retained_count=" << retained.size() << std::endl;
//                 std::cout << "  index_order: ";
//                 for (size_t i = 0; i < index_pairs.size(); ++i) std::cout << index_pairs[i].second << " ";
//                 std::cout << std::endl;
//                 std::cout << "  sorted_values_cat: ";
//                 for (size_t i = 0; i < sorted_values_cat.size(); ++i) std::cout << sorted_values_cat[i] << " ";
//                 std::cout << std::endl;
//             }
//         } // q_block loop

//         // --- Enforce causal (keep original style/condition)
//         auto it = retained.begin();
//         while (it != retained.end()) {
//             size_t q = it->first;
//             size_t k = it->second;
//             if (k >= current_index && (k - current_index) > q) {
//                 it = retained.erase(it);
//             } else {
//                 ++it;
//             }
//         }
//     } // head loop

//     return retval;
// }
// template <typename T>
// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(
//     const T* blocked_attention_scores_data,
//     const Shape& blocked_attention_scores_shape) {
    
//     OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

//     auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//     size_t num_heads = blocked_attention_scores_shape[0];
//     size_t q_block_num = blocked_attention_scores_shape[1];
//     size_t k_block_num = blocked_attention_scores_shape[2];

//     // 当前索引保持与原始 C++ 一致，匹配 Python caller
//     size_t current_index = k_block_num - q_block_num;

//     for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//         auto& retained = retval[head_idx];

//         for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
//             // --- 1) 读一行（q_block_idx）并计算 row_sum
//             std::vector<double> row(k_block_num);
//             double row_sum = 0.0;
//             for (size_t k_block_idx = 0; k_block_idx < k_block_num; ++k_block_idx) {
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 double v = static_cast<double>(blocked_attention_scores_data[offset]);
//                 if (std::isnan(v) || std::isinf(v))
//                     v = 0.0;
//                 row[k_block_idx] = v;
//                 row_sum += v;
//             }

//             double required_sum = m_threshold * row_sum;

//             // --- 2) 构造 forced mask（k==0 与 diagonal_k）
//             std::vector<char> forced(k_block_num, 0);
//             forced[0] = 1;
//             size_t diagonal_k = current_index + q_block_idx;
//             if (diagonal_k < k_block_num)
//                 forced[diagonal_k] = 1;

//             // --- 3) 计算 forced_sum
//             double forced_sum = 0.0;
//             for (size_t k = 0; k < k_block_num; ++k)
//                 if (forced[k])
//                     forced_sum += row[k];

//             // --- 4) 构造 other_values = masked_fill(mask,0) 并降序排序
//             std::vector<std::pair<double, size_t>> other_pairs;
//             other_pairs.reserve(k_block_num);
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 double val = forced[k] ? 0.0 : row[k];
//                 other_pairs.emplace_back(val, k);
//             }
//             std::sort(other_pairs.begin(), other_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 5) 构造 sorted_values_cat
//             std::vector<double> sorted_values_cat;
//             sorted_values_cat.reserve(k_block_num);
//             sorted_values_cat.push_back(0.0);
//             sorted_values_cat.push_back(forced_sum);
//             size_t take = (k_block_num >= 2) ? (k_block_num - 2) : 0;
//             for (size_t i = 0; i < take; ++i) {
//                 sorted_values_cat.push_back(other_pairs[i].first);
//             }
//             while (sorted_values_cat.size() < k_block_num)
//                 sorted_values_cat.push_back(0.0);

//             // --- 6) 构造 index_order
//             std::vector<std::pair<double, size_t>> index_pairs;
//             index_pairs.reserve(k_block_num);
//             const double BIG = 100000.0;
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 double key = forced[k] ? (BIG * (1.0 + row[k])) : row[k];
//                 index_pairs.emplace_back(key, k);
//             }
//             std::sort(index_pairs.begin(), index_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 7) 构造 cumulative_sum_without_self
//             std::vector<double> cumsum_before(k_block_num, 0.0);
//             double acc = 0.0;
//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 cumsum_before[pos] = acc;
//                 acc += sorted_values_cat[pos];
//             }

//             // // --- 8) 累加保留逻辑，严格对应 Python
//             // retained.insert({q_block_idx, 0});
//             // if (diagonal_k < k_block_num)
//             //     retained.insert({q_block_idx, diagonal_k});

//             // for (size_t pos = 0; pos < k_block_num; ++pos) {
//             //     if (cumsum_before[pos] < required_sum) {
//             //         size_t sel_k = index_pairs[pos].second;
//             //         retained.insert({q_block_idx, sel_k});
//             //     } else {
//             //         break; // <-- 关键修改，停止累加，避免多保留 (7,6)
//             //     }
//             // }

//             // --- 8) 累加保留逻辑，严格对应 Python
//             retained.insert({q_block_idx, 0});
//             if (diagonal_k < k_block_num)
//                 retained.insert({q_block_idx, diagonal_k});

//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 size_t sel_k = index_pairs[pos].second;
//                 if (!forced[sel_k] && cumsum_before[pos] >= required_sum) {
//                     // Python 对应 torch.where(index_mask, index, 0)
//                     continue; // 不保留非强制位置
//                 }
//                 retained.insert({q_block_idx, sel_k});
//             }



//             // --- debug 打印（可注释）
//             /*
//             if (head_idx == 0 && (q_block_idx == 6 || q_block_idx == 7)) {
//                 std::cout << "[DBG] q=" << q_block_idx
//                           << " row_sum=" << row_sum
//                           << " required=" << required_sum
//                           << " forced_sum=" << forced_sum
//                           << " cumsum_before(last)=" << cumsum_before.back()
//                           << " retained_count=" << retained.size() << std::endl;
//                 std::cout << "  index_order: ";
//                 for (size_t i = 0; i < index_pairs.size(); ++i) std::cout << index_pairs[i].second << " ";
//                 std::cout << std::endl;
//                 std::cout << "  sorted_values_cat: ";
//                 for (size_t i = 0; i < sorted_values_cat.size(); ++i) std::cout << sorted_values_cat[i] << " ";
//                 std::cout << std::endl;
//             }
//             */
//         }

//         // --- Enforce causal
//         auto it = retained.begin();
//         while (it != retained.end()) {
//             size_t q = it->first;
//             size_t k = it->second;
//             if (k >= current_index && (k - current_index) > q) {
//                 it = retained.erase(it);
//             } else {
//                 ++it;
//             }
//         }
//     }

//     return retval;
// }

    XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
                                                                        const Shape& blocked_attention_scores_shape) {
        OPENVINO_ASSERT(blocked_attention_scores_shape.size() ==
                        3);  // [num_heads, num_blocks_in_query, num_blocks_in_key]
                             //
        OPENVINO_ASSERT(blocked_attention_scores_shape[1] <= blocked_attention_scores_shape[2]);

//     auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//     size_t num_heads = blocked_attention_scores_shape[0];
//     size_t q_block_num = blocked_attention_scores_shape[1];
//     size_t k_block_num = blocked_attention_scores_shape[2];

//     // 与 Python 对齐
//     size_t current_index = k_block_num - q_block_num;

//     for (size_t head_idx = 0; head_idx < num_heads; ++head_idx) {
//         auto& retained = retval[head_idx];

//         for (size_t q_block_idx = 0; q_block_idx < q_block_num; ++q_block_idx) {
//             // --- 1) 读取一行
//             std::vector<double> row(k_block_num);
//             double row_sum = 0.0;
//             for (size_t k_block_idx = 0; k_block_idx < k_block_num; ++k_block_idx) {
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 double v = static_cast<double>(blocked_attention_scores_data[offset]);
//                 if (std::isnan(v) || std::isinf(v)) v = 0.0;
//                 row[k_block_idx] = v;
//                 row_sum += v;
//             }

//             double required_sum = m_threshold * row_sum;

//             // --- 2) 强制保留位置
//             std::vector<char> forced(k_block_num, 0);
//             forced[0] = 1;
//             size_t diagonal_k = current_index + q_block_idx;
//             if (diagonal_k < k_block_num) forced[diagonal_k] = 1;

//             double forced_sum = 0.0;
//             for (size_t k = 0; k < k_block_num; ++k)
//                 if (forced[k]) forced_sum += row[k];

//             // --- 3) 其他值排序
//             std::vector<std::pair<double, size_t>> other_pairs; // (value, k_idx)
//             for (size_t k = 0; k < k_block_num; ++k)
//                 other_pairs.emplace_back(forced[k] ? 0.0 : row[k], k);
//             std::sort(other_pairs.begin(), other_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 4) 构造 sorted_values_cat
//             std::vector<double> sorted_values_cat;
//             sorted_values_cat.push_back(0.0);
//             sorted_values_cat.push_back(forced_sum);
//             size_t take = k_block_num >= 2 ? k_block_num - 2 : 0;
//             for (size_t i = 0; i < take; ++i) sorted_values_cat.push_back(other_pairs[i].first);
//             while (sorted_values_cat.size() < k_block_num) sorted_values_cat.push_back(0.0);

//             // --- 5) 构造 index_pairs (argsort desc)
//             std::vector<std::pair<double, size_t>> index_pairs;
//             const double BIG = 100000.0;
//             for (size_t k = 0; k < k_block_num; ++k)
//                 index_pairs.emplace_back(forced[k] ? (BIG * (1.0 + row[k])) : row[k], k);
//             std::sort(index_pairs.begin(), index_pairs.end(), [](const auto& a, const auto& b) {
//                 return a.first > b.first;
//             });

//             // --- 6) cumsum_before
//             std::vector<double> cumsum_before(k_block_num, 0.0);
//             double acc = 0.0;
//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 cumsum_before[pos] = acc;
//                 acc += sorted_values_cat[pos];
//             }

//             // --- 7) 强制保留
//             retained.insert({q_block_idx, 0});
//             if (diagonal_k < k_block_num) retained.insert({q_block_idx, diagonal_k});

//             // --- 8) 按 Python 逻辑选择
//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 if (cumsum_before[pos] < required_sum) {
//                     size_t sel_k = index_pairs[pos].second;
//                     retained.insert({q_block_idx, sel_k});
//                 }
//             }

//             // --- 9) 完整 debug 打印
//             std::cout << "[DBG] head=" << head_idx << " q=" << q_block_idx
//                       << " row_sum=" << row_sum
//                       << " required=" << required_sum
//                       << " forced_sum=" << forced_sum
//                       << " cumsum_before(last)=" << cumsum_before.back()
//                       << " retained_count=" << retained.size() << std::endl;

//             std::cout << "  row: ";
//             for (auto v : row) std::cout << v << " ";
//             std::cout << std::endl;

//             std::cout << "  forced: ";
//             for (auto f : forced) std::cout << (int)f << " ";
//             std::cout << std::endl;

//             std::cout << "  other_pairs: ";
//             for (auto& p : other_pairs) std::cout << "(" << p.first << "," << p.second << ") ";
//             std::cout << std::endl;

//             std::cout << "  sorted_values_cat: ";
//             for (auto v : sorted_values_cat) std::cout << v << " ";
//             std::cout << std::endl;

//             std::cout << "  index_pairs: ";
//             for (auto& p : index_pairs) std::cout << "(" << p.first << "," << p.second << ") ";
//             std::cout << std::endl;

//             std::cout << "  cumsum_before: ";
//             for (auto v : cumsum_before) std::cout << v << " ";
//             std::cout << std::endl;

//             std::cout << "  retained before causal: ";
//             for (auto& p : retained) std::cout << "(" << p.first << "," << p.second << ") ";
//             std::cout << std::endl;
//         } // q_block loop

//         // --- 10) enforce causal
//         auto it = retained.begin();
//         while (it != retained.end()) {
//             size_t q = it->first;
//             size_t k = it->second;
//             if (k >= current_index && (k - current_index) > q)
//                 it = retained.erase(it);
//             else
//                 ++it;
//         }

//         // --- 11) 打印 causal 后 retained
//         std::cout << "[DBG] head=" << head_idx << " retained after causal: ";
//         for (auto& p : retained) std::cout << "(" << p.first << "," << p.second << ") ";
//         std::cout << std::endl;
//     } // head loop

//     return retval;
// }


// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
//                                                                           const Shape& blocked_attention_scores_shape) {
//     OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

//     auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//     size_t num_heads = blocked_attention_scores_shape[0];
//     size_t q_block_num = blocked_attention_scores_shape[1];
//     size_t k_block_num = blocked_attention_scores_shape[2];

//     size_t current_index = k_block_num - q_block_num; // Python caller behavior

//     const double BIG = 100000.0;

//     for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//         auto& retained = retval[head_idx];

//         for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
//             // --- 1) row
//             std::vector<double> row(k_block_num);
//             double row_sum = 0.0;
//             for (size_t k_block_idx = 0; k_block_idx < k_block_num; ++k_block_idx) {
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 double v = static_cast<double>(blocked_attention_scores_data[offset]);
//                 if (std::isnan(v) || std::isinf(v)) v = 0.0;
//                 row[k_block_idx] = v;
//                 row_sum += v;
//             }
//             double required_sum = m_threshold * row_sum;

//             // --- 2) forced mask
//             std::vector<char> forced(k_block_num, 0);
//             forced[0] = 1;
//             size_t diagonal_k = current_index + q_block_idx;
//             if (diagonal_k < k_block_num)
//                 forced[diagonal_k] = 1;

//             // --- 3) forced sum
//             double forced_sum = 0.0;
//             for (size_t k = 0; k < k_block_num; ++k)
//                 if (forced[k]) forced_sum += row[k];

//             // --- 4) other values
//             std::vector<std::pair<double, size_t>> other_pairs; // value, k
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 if (!forced[k]) other_pairs.emplace_back(row[k], k);
//             }
//             std::sort(other_pairs.begin(), other_pairs.end(),
//                       [](const auto& a, const auto& b) { return a.first > b.first; });

//             // --- 5) sorted_values_cat
//             std::vector<double> sorted_values_cat;
//             sorted_values_cat.push_back(0.0);
//             sorted_values_cat.push_back(forced_sum);
//             size_t take_count = (other_pairs.size() >= 2) ? other_pairs.size() - 2 : other_pairs.size();
//             for (size_t i = 0; i < take_count; ++i) sorted_values_cat.push_back(other_pairs[i].first);
//             while (sorted_values_cat.size() < k_block_num) sorted_values_cat.push_back(0.0);

//             // --- 6) index pairs (argsort)
//             std::vector<std::pair<double, size_t>> index_pairs;
//             for (size_t k = 0; k < k_block_num; ++k) {
//                 double key = forced[k] ? BIG * (1.0 + row[k]) : row[k];
//                 index_pairs.emplace_back(key, k);
//             }
//             std::sort(index_pairs.begin(), index_pairs.end(),
//                       [](const auto& a, const auto& b) { return a.first > b.first; });

//             // --- 7) cumsum_before
//             std::vector<double> cumsum_before(k_block_num, 0.0);
//             double acc = 0.0;
//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 cumsum_before[pos] = acc;
//                 acc += sorted_values_cat[pos];
//             }

//             // --- 8) insert into retained
//             // force include 0 and diagonal
//             retained.insert({q_block_idx, 0});
//             if (diagonal_k < k_block_num) retained.insert({q_block_idx, diagonal_k});

//             for (size_t pos = 0; pos < k_block_num; ++pos) {
//                 if (cumsum_before[pos] < required_sum) {
//                     size_t sel_k = index_pairs[pos].second;
//                     retained.insert({q_block_idx, sel_k});
//                 }
//             }

//             // --- debug print
//             std::cout << "[DBG] head=" << head_idx << " q=" << q_block_idx
//                       << " row_sum=" << row_sum
//                       << " required=" << required_sum
//                       << " forced_sum=" << forced_sum
//                       << " cumsum_before(last)=" << cumsum_before.back()
//                       << " retained_count=" << retained.size() << "\n";
//             std::cout << "  row: ";
//             for (auto v : row) std::cout << v << " ";
//             std::cout << "\n  forced: ";
//             for (auto f : forced) std::cout << int(f) << " ";
//             std::cout << "\n  other_pairs: ";
//             for (auto& p : other_pairs) std::cout << "(" << p.first << "," << p.second << ") ";
//             std::cout << "\n  sorted_values_cat: ";
//             for (auto v : sorted_values_cat) std::cout << v << " ";
//             std::cout << "\n  index_pairs: ";
//             for (auto& p : index_pairs) std::cout << "(" << p.first << "," << p.second << ") ";
//             std::cout << "\n  cumsum_before: ";
//             for (auto v : cumsum_before) std::cout << v << " ";
//             std::cout << "\n  retained before causal: ";
//             for (auto& x : retained) std::cout << "(" << x.first << "," << x.second << ") ";
//             std::cout << "\n";
//         }

//         // --- 9) causal mask
//         auto it = retained.begin();
//         while (it != retained.end()) {
//             size_t q = it->first;
//             size_t k = it->second;
//             if (k >= current_index && (k - current_index) > q) {
//                 it = retained.erase(it);
//             } else {
//                 ++it;
//             }
//         }

//         // --- debug retained after causal
//         std::cout << "[DBG] head=" << head_idx << " retained after causal: ";
//         for (auto& x : retained) std::cout << "(" << x.first << "," << x.second << ") ";
//         std::cout << "\n";
//     }

//     return retval;
// }

void print_blocked_attention_scores(const T* blocked_attention_scores_data,
                                    size_t num_heads,
                                    size_t q_block_num,
                                    size_t k_block_num) {
    std::cout << "=== blocked_attention_scores_data ===\n";
    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t q = 0; q < q_block_num; ++q) {
            std::cout << " q_block " << q << ": ";
            for (size_t k = 0; k < k_block_num; ++k) {
                size_t offset = h * q_block_num * k_block_num + q * k_block_num + k;
                std::cout << std::fixed << std::setprecision(6)
                          << blocked_attention_scores_data[offset] << " ";
            }
            std::cout << "\n";
        }
        std::cout << "\n";
    }
}

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
    }

    return retval;
}

void print_attn_score_buf_with_shape(const std::shared_ptr<T[]>& buf,
                                     size_t num_heads,
                                     size_t rows,    // 实际 buf 的第2维长度
                                     size_t cols,    // 实际 buf 的第3维长度
                                     size_t show_first_n_cols = 0) { // 0 表示显示全部
    std::cout << "=== Debug: attn_score_buf (shape = [" << num_heads << ", " << rows << ", " << cols << "]) ===\n";
    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t r = 0; r < rows; ++r) {
            std::cout << std::setw(3) << r << ": ";
            size_t nonzero_count = 0;
            size_t limit = (show_first_n_cols == 0) ? cols : std::min(cols, (size_t)show_first_n_cols);
            for (size_t c = 0; c < limit; ++c) {
                size_t idx = h * rows * cols + r * cols + c;
                double v = static_cast<double>(buf[idx]);
                if (std::fabs(v) > 1e-12) ++nonzero_count;
                std::cout << std::fixed << std::setprecision(6) << v << " ";
            }
            if (limit < cols) std::cout << "...";
            std::cout << "  (nonzero=" << nonzero_count << ")\n";
        }
        // 打印非零掩码行（帮助看 pattern）
        std::cout << "Nonzero mask per row: ";
        for (size_t r = 0; r < rows; ++r) {
            size_t nonzero = 0;
            for (size_t c = 0; c < cols; ++c) {
                size_t idx = h * rows * cols + r * cols + c;
                if (std::fabs(static_cast<double>(buf[idx])) > 1e-12) {
                    nonzero = 1;
                    break;
                }
            }
            std::cout << nonzero;
        }
        std::cout << "\n\n";
    }
    std::cout << "=== End attn_score_buf ===\n";
}

void print_qk_buf(const std::shared_ptr<T[]>& qk_buf,
                  size_t num_heads,
                  size_t q_block_num,
                  size_t k_block_num,
                  size_t show_first_n_cols = 0) {
    std::cout << "\n=== Debug: qk_buf (shape = ["
              << num_heads << ", " << q_block_num << ", " << k_block_num << "]) ==="
              << std::endl;

    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t q = 0; q < q_block_num; ++q) {
            std::cout << std::setw(3) << q << ": ";
            size_t limit = (show_first_n_cols == 0)
                               ? k_block_num
                               : std::min(k_block_num, (size_t)show_first_n_cols);
            size_t nonzero_count = 0;
            for (size_t k = 0; k < limit; ++k) {
                size_t idx = h * q_block_num * k_block_num + q * k_block_num + k;
                double val = static_cast<double>(qk_buf[idx]);
                if (std::fabs(val) > 1e-12)
                    ++nonzero_count;
                std::cout << std::fixed << std::setprecision(6) << val << " ";
            }
            if (limit < k_block_num)
                std::cout << "...";
            std::cout << " (nonzero=" << nonzero_count << ")\n";
        }

        // 打印每行是否含非零的简单掩码
        std::cout << "Nonzero mask per row: ";
        for (size_t q = 0; q < q_block_num; ++q) {
            bool nonzero = false;
            for (size_t k = 0; k < k_block_num; ++k) {
                size_t idx = h * q_block_num * k_block_num + q * k_block_num + k;
                if (std::fabs(static_cast<double>(qk_buf[idx])) > 1e-12) {
                    nonzero = true;
                    break;
                }
            }
            std::cout << (nonzero ? "1" : "0");
        }
        std::cout << "\n\n";
    }

    std::cout << "=== End of qk_buf ===\n" << std::endl;
}

void assign_qk_buf(std::shared_ptr<T[]>& qk_buf,
                   size_t num_heads,
                   size_t q_block_num,
                   size_t k_block_num) {
    std::vector<float> data = {
        0.1953, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.1914, 0.2695, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.2305, -0.1211, -0.1211, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        0.0703, -0.0859, 0.2148, -0.1367, -65504.0, -65504.0, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.1367, -0.4766, -0.0039, 0.0273, 0.2031, -65504.0, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.4414, 0.0703, 0.3477, 0.4102, 0.2891, 0.4453, -65504.0, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.2266, -0.1797, 0.1992, 0.1523, 0.0586, 0.5234, -0.2070, -65504.0,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.3164, -0.0117, 0.0312, 0.2422, 0.3047, 0.1562, -0.1172, 0.0820,
        -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        0.4648, -0.0117, 0.1680, -0.3086, -0.2695, 0.3906, -0.1641, -0.1406,
        -0.1211, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        0.3086, -0.0156, 0.0430, -0.0938, -0.1484, 0.2773, -0.2812, 0.0039,
        -0.1133, -0.2656, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        0.5078, -0.0664, -0.2266, -0.6055, -0.2383, -0.1719, -0.0195, 0.2461,
        0.0859, -0.1680, 0.1875, -65504.0, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.4922, 0.4258, 0.2578, 0.4219, 0.0820, 0.3711, 0.4688, -0.5859,
        -0.1328, 0.4102, -0.2266, 0.2695, -65504.0, -65504.0, -65504.0, -65504.0,

        -0.5586, 0.5430, 0.1211, 0.3359, -0.0859, -0.3477, 0.2500, 0.0391,
        -0.1797, 0.5430, -0.2109, 0.7695, 0.1484, -65504.0, -65504.0, -65504.0,

        0.0859, -0.1406, 0.0430, -0.1406, -0.0938, -0.2539, -0.0781, -0.0273,
        -0.0820, -0.2578, 0.0469, -0.0781, -0.2227, -0.2969, -65504.0, -65504.0,

        -0.2109, -0.2539, 0.3086, 0.7109, 0.2695, 0.5547, -0.0977, -0.5430,
        -0.1953, -0.3242, -0.1289, -0.0156, -0.0547, -0.5391, 0.1133, -65504.0,

        0.0742, 0.1758, 0.2344, -0.1523, -0.2109, -0.0508, 0.0859, -0.1953,
        -0.1562, 0.1680, 0.3242, 0.0195, -0.4141, -0.3164, -0.1133, 0.2383
    };

    size_t total = num_heads * q_block_num * k_block_num;
    if (data.size() != total) {
        std::cerr << "Error: expected total=" << total << " but data.size=" << data.size() << std::endl;
        return;
    }

    // qk_buf = std::shared_ptr<float[]>(new float[total]);
    std::copy(data.begin(), data.end(), qk_buf.get());
}

void print_causal_mask_buf(const std::shared_ptr<T[]>& causal_mask_buf,
                           size_t num_heads,
                           size_t q_block_num,
                           size_t k_block_num) {
    std::cout << "=== Debug: causal_mask_buf ===" << std::endl;

    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t q = 0; q < q_block_num; ++q) {
            for (size_t k = 0; k < k_block_num; ++k) {
                size_t idx = h * q_block_num * k_block_num + q * k_block_num + k;
                auto val = static_cast<double>(causal_mask_buf[idx]);
                std::cout << std::setw(6) << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== End of causal_mask_buf ===" << std::endl;
}

void print_q_buf(const std::shared_ptr<T[]>& q_buf,
                 size_t num_heads,
                 size_t q_block_num,
                 size_t head_dim) {
    std::cout << "=== Debug: q_buf ===" << std::endl;

    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t q = 0; q < q_block_num; ++q) {
            std::cout << "Q" << std::setw(2) << q << ": ";
            for (size_t d = 0; d < head_dim; ++d) {
                size_t idx = h * q_block_num * head_dim + q * head_dim + d;
                auto val = static_cast<double>(q_buf[idx]);
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== End of q_buf ===" << std::endl;
}

void print_k_buf(const std::shared_ptr<T[]>& k_buf,
                 size_t num_heads,
                 size_t q_block_num,
                 size_t head_dim) {
    std::cout << "=== Debug: k_buf ===" << std::endl;

    for (size_t h = 0; h < num_heads; ++h) {
        std::cout << "Head " << h << ":\n";
        for (size_t q = 0; q < q_block_num; ++q) {
            std::cout << "Q" << std::setw(2) << q << ": ";
            for (size_t d = 0; d < head_dim; ++d) {
                size_t idx = h * q_block_num * head_dim + q * head_dim + d;
                auto val = static_cast<double>(k_buf[idx]);
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << val << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    std::cout << "=== End of q_buf ===" << std::endl;
}

void print_query_data(const T* data, const std::vector<size_t>& shape, const std::string& name = "query_data") {
    if (!data) {
        std::cout << name << " is nullptr\n";
        return;
    }

    std::cout << "=== " << name << " ===\n";

    if (shape.size() == 3) {  // [num_heads, q_block_num, k_block_num]
        size_t H = shape[0];
        size_t Q = shape[1];
        size_t K = shape[2];

        for (size_t h = 0; h < H; ++h) {
            std::cout << "Head " << h << ":\n";
            for (size_t q = 0; q < Q; ++q) {
                for (size_t k = 0; k < K; ++k) {
                    size_t idx = h * Q * K + q * K + k;
                    std::cout << std::fixed << std::setprecision(4)
                              << static_cast<float>(data[idx]) << " ";
                }
                std::cout << "\n";
            }
            std::cout << "\n";
        }
    } else if (shape.size() == 4) {  // [B, H, Q, K]
        size_t B = shape[0];
        size_t H = shape[1];
        size_t Q = shape[2];
        size_t K = shape[3];

        for (size_t b = 0; b < B; ++b) {
            std::cout << "Batch " << b << ":\n";
            for (size_t h = 0; h < H; ++h) {
                std::cout << "  Head " << h << ":\n";
                for (size_t q = 0; q < Q; ++q) {
                    std::cout << "    ";
                    for (size_t k = 0; k < K; ++k) {
                        size_t idx = b * H * Q * K + h * Q * K + q * K + k;
                        std::cout << std::fixed << std::setprecision(4)
                                  << static_cast<float>(data[idx]) << " ";
                    }
                    std::cout << "\n";
                }
                std::cout << "\n";
            }
        }
    } else {
        std::cout << "Unsupported shape size=" << shape.size() << "\n";
    }

    std::cout << "=== End of " << name << " ===\n";
}

void set_q_buf(std::shared_ptr<T[]> &q_buf) {
    const size_t B = 1;
    const size_t H = 1;
    const size_t Q = 32;
    const size_t dim = 4;

    // tmp_data 用 float 填写你的 chunked_query 数据
    float tmp_data[B*H*Q*dim] = {
        -0.3750,  1.0000, -0.2500,  0.2500, -1.0000, -0.5000, -0.1250, 0.0000,
        -0.6250, -0.2500,  0.7500,  0.7500, -0.2500,  0.3750, -0.3750, -0.3750,
        -0.6250, -0.7500,  0.1250,  0.1250,  1.0000,  0.7500, -0.8750, 0.1250,
        0.3750,  0.8750, -0.1250, -0.2500,  1.0000,  0.7500,  0.2500, -0.2500,
        0.1250,  0.8750, -0.8750, -0.3750,  0.6250, -0.3750, -0.1250, -1.0000,
        -0.3750,  0.7500,  0.0000,  0.8750,  0.7500,  0.2500,  0.6250, -0.6250,
        0.8750, -0.2500, -0.1250,  0.7500,  0.2500,  0.3750, -0.6250, -0.7500,
        -0.7500,  0.0000, -0.2500,  0.6250, -1.0000, -0.5000, -0.6250, -1.0000,
        0.8750,  0.2500,  0.5000, -0.6250, -0.1250,  0.7500, -0.7500, -0.5000,
        1.0000, -0.3750,  0.6250,  0.3750,  0.2500,  0.5000, -0.5000, 0.7500,
        0.1250,  0.0000,  0.0000, -1.0000,  0.2500,  0.6250, -0.5000, 0.8750,
        -0.7500, -0.6250,  0.8750,  0.7500,  1.0000,  0.7500,  0.7500, 0.1250,
        -0.5000, -1.0000,  0.0000,  0.7500, -0.8750, -0.1250,  1.0000, -0.1250,
        0.7500,  0.7500, -0.7500, -0.1250,  0.1250, -0.1250,  0.6250, 0.1250,
        0.7500,  0.6250,  0.5000,  0.8750,  1.0000, -0.6250,  0.5000, -0.6250,
        0.3750,  0.6250, -0.2500, -0.3750, -0.3750,  0.3750,  0.5000, -0.6250
    };

    for (size_t idx = 0; idx < B*H*Q*dim; ++idx) {
        q_buf[idx] = ov::float16(tmp_data[idx]);
    }
}

    XAttentionRetainedBlockIndicesForAllHeads select_blocks(const T* query_data,
                                                            const Shape& query_shape,
                                                            const T* key_data,
                                                            const Shape& key_shape) {
        OPENVINO_ASSERT(query_shape.size() == 3);
        OPENVINO_ASSERT(key_shape.size() == 3);
        OPENVINO_ASSERT(key_shape[0] == query_shape[0]);
        OPENVINO_ASSERT(key_shape[2] == query_shape[2]);
        OPENVINO_ASSERT(query_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(key_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(query_shape[1] % m_block_size == 0);
        OPENVINO_ASSERT(key_shape[1] % m_block_size == 0);
        // print_query_data(query_data, {1, 32, 4});

        size_t chunk_size = query_shape[1];
        size_t k_len = key_shape[1];
        size_t head_dim = query_shape[2];
        size_t num_heads = query_shape[0];
        size_t k_num_to_pad = ((k_len + chunk_size - 1) / chunk_size) * chunk_size - k_len;
        Shape pad_key_shape = {num_heads, k_len + k_num_to_pad, head_dim};
        auto pad_key_buf = allocate_buf(pad_key_shape);

        for (size_t h = 0; h < num_heads; h++)
            for (size_t t = 0; t < k_len; t++)
                for (size_t d = 0; d < head_dim; d++) {
                    size_t offset = h * (k_len + k_num_to_pad) * head_dim + t * head_dim + d;
                    size_t original_offset = h * k_len * head_dim + t * head_dim + d;
                    pad_key_buf.get()[offset] = key_data[original_offset];
                }

        size_t k_chunk_num = (k_len + k_num_to_pad) / chunk_size;
        size_t offset_token_chunk_num = k_chunk_num - 1;
        size_t reshaped_chunk_size = chunk_size / m_stride;
        // size_t reshaped_block_size = m_block_size / m_stride;
        size_t k_reshaped_num_to_pad = k_num_to_pad / m_stride;
        size_t k_reshaped_seq_len = (k_len + k_num_to_pad) / m_stride;

        // size_t num_blocks_per_chunk = reshaped_chunk_size / reshaped_block_size;

        // size_t q_block_num = chunk_size / m_block_size;

        // size_t k_block_num = (k_len + k_num_to_pad) / m_block_size;

        Shape reshaped_query_shape = {num_heads, query_shape[1] / m_stride, head_dim * m_stride};
        auto q_buf = allocate_buf(reshaped_query_shape);
        diagonal_reshape_kdb1_no_batch(query_data, query_shape, q_buf.get(), reshaped_query_shape);
        Shape reshaped_key_shape = {num_heads, pad_key_shape[1] / m_stride, head_dim * m_stride};
        auto k_buf = allocate_buf(reshaped_key_shape);
        diagonal_reshape(pad_key_buf.get(), pad_key_shape, k_buf.get(), reshaped_key_shape, false);
        Shape transpose_matmul_scaled_shape = {num_heads, query_shape[1] / m_stride, pad_key_shape[1] / m_stride};
        std::cout << "transpose_matmul_scaled_shape: \n";
        for (auto ii : transpose_matmul_scaled_shape) {
            std::cout << ii << " ";
        }
        std::cout << std::endl;
        auto qk_buf = allocate_buf(transpose_matmul_scaled_shape);


        // print_q_buf(q_buf, num_heads, query_shape[1] / m_stride, head_dim * m_stride);
        // set_q_buf(q_buf);
        // print_q_buf(q_buf, num_heads, query_shape[1] / m_stride, head_dim * m_stride);
        // print_k_buf(k_buf, num_heads, pad_key_shape[1] / m_stride, head_dim * m_stride);
        transpose_matmul_scale(q_buf.get(),
                               k_buf.get(),
                               reshaped_query_shape,
                               reshaped_key_shape,
                               qk_buf.get(),
                               transpose_matmul_scaled_shape);
        // print_qk_buf(qk_buf, num_heads, 16, 16);

        q_buf.reset();
        k_buf.reset();
        Shape causal_mask_shape = {num_heads, reshaped_chunk_size, reshaped_chunk_size * k_chunk_num};
        auto causal_mask_buf = allocate_buf(causal_mask_shape);
        std::fill(causal_mask_buf.get(), causal_mask_buf.get() + ov::shape_size(causal_mask_shape), T(0));
        if (k_reshaped_num_to_pad) {
            for (size_t h = 0; h < num_heads; h++)
                for (size_t q = 0; q < reshaped_chunk_size; q++)
                    for (size_t k = k_reshaped_seq_len - k_reshaped_num_to_pad; k < k_reshaped_seq_len; k++) {
                        size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) +
                                        q * (reshaped_chunk_size * k_chunk_num) + k;

                        causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                    }
        }

        size_t chunk_start = offset_token_chunk_num * reshaped_chunk_size;

        size_t chunk_end = chunk_start + reshaped_chunk_size;

        for (size_t h = 0; h < num_heads; h++)
            for (size_t q = 0; q < reshaped_chunk_size; q++)
                for (size_t k = q + 1; k < reshaped_chunk_size; k++) {
                    size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) +
                                    q * (reshaped_chunk_size * k_chunk_num) + chunk_start + k;

                    causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                }

        for (size_t h = 0; h < num_heads; h++)
            for (size_t q = 0; q < reshaped_chunk_size; q++)
                for (size_t k = chunk_end; k < reshaped_chunk_size * k_chunk_num; k++) {
                    size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) +
                                    q * (reshaped_chunk_size * k_chunk_num) + k;

                    causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                }

        // slice [: , : , 0 ::1 , : ] since kdb=1

        size_t out_size =
            transpose_matmul_scaled_shape[0] * transpose_matmul_scaled_shape[1] * transpose_matmul_scaled_shape[2];


        // print_causal_mask_buf(causal_mask_buf, num_heads, reshaped_chunk_size, reshaped_chunk_size * k_chunk_num);

        for (size_t i = 0; i < out_size; i++) {
            qk_buf.get()[i] += causal_mask_buf.get()[i];
        }


        
        causal_mask_buf.reset();

        apply_causal_mask_(qk_buf.get(), transpose_matmul_scaled_shape);

        Shape attention_scores_shape = transpose_matmul_scaled_shape;

        auto attn_score_buf = allocate_buf(attention_scores_shape);

        // print_qk_buf(qk_buf, num_heads, 16, 16);
        // assign_qk_buf(qk_buf, num_heads, 16, 16);
        // print_qk_buf(qk_buf, num_heads, 16, 16);


        softmax(qk_buf.get(), transpose_matmul_scaled_shape, attn_score_buf.get(), attention_scores_shape);

        qk_buf.reset();

        // print_attn_score_buf_with_shape(attn_score_buf, 
        //                         transpose_matmul_scaled_shape[0],
        //                         transpose_matmul_scaled_shape[1],
        //                         transpose_matmul_scaled_shape[2]);


        

        size_t antidiagonals_per_xattention_block = m_block_size / m_stride;
        Shape block_sum_shape = {attention_scores_shape[0],
                                 attention_scores_shape[1] / antidiagonals_per_xattention_block,
                                 attention_scores_shape[2] / antidiagonals_per_xattention_block};

        auto block_sum_buf = allocate_buf(block_sum_shape);
        block_sum_attention_scores(attn_score_buf.get(), attention_scores_shape, block_sum_buf.get(), block_sum_shape);
        attn_score_buf.reset();
        auto selected_block_indices = get_block_indices_to_keep(block_sum_buf.get(), block_sum_shape);
        block_sum_buf.reset();

        // The Python has the tril on the last q_block_num

        // So, to match, the simple_masks [: , : , -q_block_num : , -q_block_num : ] = where (tril, simple_masks, False)

        // But since the return is the set, we can do in the retained, erase the upper

        // Yes, already has.

        return selected_block_indices;
    }

    std::shared_ptr<T[]> allocate_buf(const Shape& shape) {
        return std::shared_ptr<T[]>(new T[ov::shape_size(shape)]);
    }

    size_t pad_to_block(size_t token_length) {
        return (token_length + m_block_size - 1) / m_block_size * m_block_size;
    }

    double m_threshold;

    size_t m_block_size;

    size_t m_stride;
};

}  // namespace ov::reference
