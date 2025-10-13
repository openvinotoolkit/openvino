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

        size_t num_stride_steps = input_shape[1] / m_stride;
        for (size_t head_idx = 0; head_idx < input_shape[0]; head_idx++) {
            size_t head_offset = head_idx * input_shape[1] * input_shape[2];
            for (size_t slice_idx = 0; slice_idx < m_stride; slice_idx++) {
                for (size_t stride_idx = 0; stride_idx < num_stride_steps; stride_idx++) {
                    size_t input_offset = head_offset;
                    size_t output_offset = head_offset + stride_idx * out_shape[2] + slice_idx * input_shape[2];
                    if (is_antidiagonal) {
                        input_offset += (input_shape[1] - 1 - slice_idx - stride_idx * m_stride) * input_shape[2];
                    } else {
                        input_offset += (slice_idx + stride_idx * m_stride) * input_shape[2];
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

void softmax_ww(const T* reshaped_qk_product_data,
             const Shape& reshaped_qk_product_shape,
             T* out,
             const Shape& out_shape) {
    OPENVINO_ASSERT(reshaped_qk_product_shape.size() == 3);
    OPENVINO_ASSERT(reshaped_qk_product_shape == out_shape);

    size_t num_heads = reshaped_qk_product_shape[0];
    size_t q_blocks  = reshaped_qk_product_shape[1];
    size_t k_blocks  = reshaped_qk_product_shape[2];

    std::vector<float> temp_in(q_blocks * k_blocks);
    std::vector<float> temp_out(q_blocks * k_blocks);

    for (size_t h = 0; h < num_heads; ++h) {
        for (size_t q = 0; q < q_blocks; ++q) {
            // 将输入从 half 转为 float
            for (size_t k = 0; k < k_blocks; ++k) {
                size_t idx = h * q_blocks * k_blocks + q * k_blocks + k;
                temp_in[k] = static_cast<float>(reshaped_qk_product_data[idx]);
            }

            // 数值稳定 softmax: 先减去最大值
            float max_val = *std::max_element(temp_in.begin(), temp_in.end());
            float sum_exp = 0.f;
            for (size_t k = 0; k < k_blocks; ++k) {
                temp_out[k] = std::exp(temp_in[k] - max_val);
                sum_exp += temp_out[k];
            }

            // 归一化
            float inv_sum = 1.f / (sum_exp + 1e-12f);
            for (size_t k = 0; k < k_blocks; ++k) {
                size_t idx = h * q_blocks * k_blocks + q * k_blocks + k;
                out[idx] = static_cast<T>(temp_out[k] * inv_sum);
            }
        }
    }
}

void softmax_fp32(const T* input, const Shape& shape, T* output, const Shape& out_shape) {
    OPENVINO_ASSERT(shape.size() == 3);
    size_t dim0 = shape[0], dim1 = shape[1], dim2 = shape[2];

    std::vector<float> temp(dim2);
    for (size_t i = 0; i < dim0 * dim1; ++i) {
        size_t offset = i * dim2;

        // 1. 转为 float32
        for (size_t j = 0; j < dim2; ++j)
            temp[j] = static_cast<float>(input[offset + j]);

        // 2. 稳定 softmax
        float max_val = *std::max_element(temp.begin(), temp.end());
        float sum_exp = 0.f;
        for (float& v : temp) {
            v = std::exp(v - max_val);
            sum_exp += v;
        }

        // 3. 写回
        for (size_t j = 0; j < dim2; ++j)
            output[offset + j] = static_cast<T>(temp[j] / sum_exp);
    }
}

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

// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(const T* blocked_attention_scores_data,
//                                                                     const Shape& blocked_attention_scores_shape) {
//     OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

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

void print_retained_blocks(const XAttentionRetainedBlockIndicesForAllHeads& retained_blocks) {
    for (size_t head = 0; head < retained_blocks.size(); ++head) {
        std::cout << "[Head " << head << "] retained blocks: ";
        for (const auto& p : retained_blocks[head]) {
            std::cout << "(" << p.first << "," << p.second << ") ";
        }
        std::cout << std::endl;
    }
}

void print_scores(const std::vector<std::pair<double, size_t>>& scores) {
    std::cout << "[Scores] ";
    for (const auto& p : scores) {
        std::cout << "(" << p.first << ", " << p.second << ") ";
    }
    std::cout << std::endl;
}



// XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(
//     T* blocked_attention_scores_data,
//     const Shape& blocked_attention_scores_shape) {
//     OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3);

//     auto retval = XAttentionRetainedBlockIndicesForAllHeads(blocked_attention_scores_shape[0]);

//     size_t num_heads = blocked_attention_scores_shape[0];
//     size_t q_block_num = blocked_attention_scores_shape[1];
//     size_t k_block_num = blocked_attention_scores_shape[2];

//     print_blocked_attention_scores(blocked_attention_scores_data,
//                                num_heads, q_block_num, k_block_num);


//     float blocked_attention_scores_values[q_block_num * k_block_num] = {
//         2.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
//         1.1399f, 0.8601f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
//         0.5426f, 0.8147f, 0.6427f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
//         0.4169f, 0.5852f, 0.6589f, 0.3390f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
//         0.5131f, 0.4026f, 0.4603f, 0.3615f, 0.2625f, 0.0000f, 0.0000f, 0.0000f,
//         0.3882f, 0.3218f, 0.3278f, 0.3583f, 0.3449f, 0.2589f, 0.0000f, 0.0000f,
//         0.3030f, 0.3146f, 0.2382f, 0.3002f, 0.2992f, 0.3479f, 0.1969f, 0.0000f,
//         0.2431f, 0.3503f, 0.3054f, 0.2146f, 0.2261f, 0.2692f, 0.1847f, 0.2065f
//     };

//     // 分配可写的 ov::float16 buffer
//     // ov::float16* blocked_attention_scores_data = new ov::float16[num_heads * q_block_num * k_block_num];

//     // 逐元素赋值
//     for (int i = 0; i < 64; ++i) {
//         blocked_attention_scores_data[i] = ov::float16(blocked_attention_scores_values[i]);
//     }

//     print_blocked_attention_scores(blocked_attention_scores_data,
//                                num_heads, q_block_num, k_block_num);

//     // ✅ Python 中没有 current_index 偏移的逻辑
//     // 原逻辑引入 offset 导致 diagonal 错位
//     // 如果确实需要 offset，可通过参数控制，但这里保持与 Python 一致
//     // size_t current_index = 0;

//     for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
//         auto& retained = retval[head_idx];

//         for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
//             std::cout << "**************************\n";
//             // 1️⃣ 累加整行分数
//             double row_sum = 0.0;
//             for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
//                 size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//                 row_sum += static_cast<double>(blocked_attention_scores_data[offset]);
//             }

//             double required_sum = m_threshold * row_sum;
//             std::cout << "required_sum: " << required_sum << std::endl;
//             double cumsum = 0.0;

//             // // 2️⃣ 强制保留 diagonal 块
//             // size_t diagonal_k = q_block_idx;
//             // size_t offset_diag = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + diagonal_k;
//             // double diag_score = static_cast<double>(blocked_attention_scores_data[offset_diag]);
//             // std::cout << "diag_score: " << diag_score << std::endl;
//             // cumsum += diag_score;
//             // retained.insert({q_block_idx, diagonal_k});

//             // print_retained_blocks(retval);

//             // // 3️⃣ 收集所有候选块
//             // std::vector<std::pair<double, size_t>> scores;
//             // scores.reserve(k_block_num);
//             // for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
//             //     if (k_block_idx == diagonal_k)
//             //         continue;
//             //     if (k_block_idx == 0) continue; 
//             //     size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//             //     scores.emplace_back(static_cast<double>(blocked_attention_scores_data[offset]), k_block_idx);
//             // }

//             // print_scores(scores);

//             // // 4️⃣ 降序排序（高分优先）
//             // std::sort(scores.begin(), scores.end(),
//             //           [](const auto& a, const auto& b) { return a.first > b.first; });

//             // // 5️⃣ 从高到低选取直到累积超过阈值
//             // for (auto& [score, k_block_idx] : scores) {
//             //     if (cumsum >= required_sum)
//             //         break;
//             //     cumsum += score;
//             //     retained.insert({q_block_idx, k_block_idx});
//             // }


//         // 2️⃣ 强制保留 diagonal 块
//         size_t diagonal_k = q_block_idx;
//         size_t offset_diag = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + diagonal_k;
//         double diag_score = static_cast<double>(blocked_attention_scores_data[offset_diag]);
//         cumsum += diag_score;
//         retained.insert({q_block_idx, diagonal_k});

//         // 2️⃣.1️⃣ 额外：强制保留首列块 (k=0)，与 Python mask[:, :, :, 0] = 1 一致
//         if (k_block_num > 0 && q_block_idx != 0) {
//             size_t offset_first = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + 0;
//             double first_col_score = static_cast<double>(blocked_attention_scores_data[offset_first]);
//             cumsum += first_col_score;
//             retained.insert({q_block_idx, 0});
//         }

//         // 3️⃣ 收集其他候选块（去掉 diagonal 和首列）
//         std::vector<std::pair<double, size_t>> scores;
//         scores.reserve(k_block_num);
//         for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
//             if (k_block_idx == diagonal_k || k_block_idx == 0)
//                 continue;
//             size_t offset = head_idx * q_block_num * k_block_num + q_block_idx * k_block_num + k_block_idx;
//             scores.emplace_back(static_cast<double>(blocked_attention_scores_data[offset]), k_block_idx);
//         }

//         // 4️⃣ 降序排序（高分优先）
//         std::sort(scores.begin(), scores.end(),
//                 [](const auto& a, const auto& b) { return a.first > b.first; });

//         // 5️⃣ 从高到低选取直到累积超过阈值
//         for (auto& [score, k_block_idx] : scores) {
//             if (cumsum >= required_sum)
//                 break;
//             cumsum += score;
//             retained.insert({q_block_idx, k_block_idx});
//         }


//             // 6️⃣ 保证左侧（k <= q）邻域不被裁掉
//             // （Python 行为是保留对角线及左侧邻近块）
//             for (int s = 1; s <= 2; s++) {  // stride=2 可根据外部参数替换
//                 if (q_block_idx >= static_cast<size_t>(s))
//                     retained.insert({q_block_idx, q_block_idx - s});
//             }

//             // 7️⃣ 保证对角块右邻域（但受 causal 约束）
//             for (int s = 1; s <= 2; s++) {
//                 size_t right = q_block_idx + s;
//                 if (right < k_block_num)
//                     retained.insert({q_block_idx, right});
//             }

//             // 调试打印（默认注释）
//             // std::cout << "[Head " << head_idx << "] Q=" << q_block_idx
//             //           << " required_sum=" << required_sum << " cumsum=" << cumsum
//             //           << " diag_score=" << diag_score << " retained=" << retained.size()
//             //           << std::endl;
//         }

//         // 8️⃣ 修正 causal mask（与 Python 一致：禁止未来块）
//         auto it = retained.begin();
//         while (it != retained.end()) {
//             size_t q = it->first;
//             size_t k = it->second;
//             if (k > q) {  // ✅ Python 中严格排除未来块
//                 it = retained.erase(it);
//             } else {
//                 ++it;
//             }
//         }

//         // 调试打印（默认注释）
//         // std::cout << "Head " << head_idx << " selected blocks:";
//         // for (auto [a, b] : retained)
//         //     std::cout << " (" << a << "," << b << ")";
//         // std::cout << std::endl;
//     }

//     return retval;
// }

XAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(
    T* blocked_attention_scores_data,
    const Shape& blocked_attention_scores_shape) {

    OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3,
                    "Expected shape [num_heads, q_block_num, k_block_num]");

    size_t num_heads = blocked_attention_scores_shape[0];
    size_t q_block_num = blocked_attention_scores_shape[1];
    size_t k_block_num = blocked_attention_scores_shape[2];

    // float blocked_attention_scores_values[q_block_num * k_block_num] = {
    //     2.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    //     1.1399f, 0.8601f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    //     0.5426f, 0.8147f, 0.6427f, 0.0000f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    //     0.4169f, 0.5852f, 0.6589f, 0.3390f, 0.0000f, 0.0000f, 0.0000f, 0.0000f,
    //     0.5131f, 0.4026f, 0.4603f, 0.3615f, 0.2625f, 0.0000f, 0.0000f, 0.0000f,
    //     0.3882f, 0.3218f, 0.3278f, 0.3583f, 0.3449f, 0.2589f, 0.0000f, 0.0000f,
    //     0.3030f, 0.3146f, 0.2382f, 0.3002f, 0.2992f, 0.3479f, 0.1969f, 0.0000f,
    //     0.2431f, 0.3503f, 0.3054f, 0.2146f, 0.2261f, 0.2692f, 0.1847f, 0.2065f
    // };

    // // 分配可写的 ov::float16 buffer
    // // ov::float16* blocked_attention_scores_data = new ov::float16[num_heads * q_block_num * k_block_num];

    // // 逐元素赋值
    // for (int i = 0; i < 64; ++i) {
    //     blocked_attention_scores_data[i] = ov::float16(blocked_attention_scores_values[i]);
    // }

    // std::vector<float> blocked_attention_scores_f32(num_heads * q_block_num * k_block_num);
    // for (size_t i = 0; i < blocked_attention_scores_f32.size(); ++i) {
    //     blocked_attention_scores_f32[i] = static_cast<float>(blocked_attention_scores_data[i]);
    // }

    // print_blocked_attention_scores(blocked_attention_scores_data,
    //                            num_heads, q_block_num, k_block_num);

    // 返回结果，每个 head 一个 set 存储 (q_block_idx, k_block_idx)
    XAttentionRetainedBlockIndicesForAllHeads retval(num_heads);

    // 临时 mask 矩阵，用于模拟 Python mask
    std::vector<std::vector<std::vector<bool>>> mask(
        num_heads, std::vector<std::vector<bool>>(
            q_block_num, std::vector<bool>(k_block_num, false)));

    for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
        for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
            // Step0: diagonal 保留
            size_t diagonal_k = q_block_idx;
            if (diagonal_k < k_block_num) {
                mask[head_idx][q_block_idx][diagonal_k] = true;
            }
            // Step1: 首列保留
            mask[head_idx][q_block_idx][0] = true;

            // Step2: 构建 other_values（masked_fill）
            std::vector<std::pair<float, size_t>> other_values;
            for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
                if (mask[head_idx][q_block_idx][k_block_idx])
                    continue;
                size_t offset = head_idx * q_block_num * k_block_num
                                + q_block_idx * k_block_num
                                + k_block_idx;
                other_values.emplace_back(static_cast<float>(blocked_attention_scores_data[offset]), k_block_idx);
            }

            // // Step4: 打印 other_values
            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] other_values:\n";
            // for (auto& [score, k_block_idx] : other_values) {
            //     std::cout << "(" << k_block_idx << ", " << score << ") ";
            // }
            // std::cout << std::endl;

            // Step3: 对 other_values 降序排序
            std::sort(other_values.begin(), other_values.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

            // Step4: 构建 cumulative_sum_without_self，cat([0, diagonal_sum, sorted_values[:-1]])
            std::vector<float> sorted_scores;
            sorted_scores.push_back(0.0);  // 前置0
            // diagonal + 首列分数
            size_t offset_diag = head_idx * q_block_num * k_block_num
                                 + q_block_idx * k_block_num
                                 + diagonal_k;
            float diag_score = static_cast<float>(blocked_attention_scores_data[offset_diag]);
            float first_col_score = 0.0;
            if (diagonal_k != 0) {
                size_t offset_first = head_idx * q_block_num * k_block_num
                                      + q_block_idx * k_block_num
                                      + 0;
                first_col_score = static_cast<float>(blocked_attention_scores_data[offset_first]);
            }
            std::cout << diag_score << " " << diag_score << " " << first_col_score << " " << diag_score + first_col_score << std::endl;
            sorted_scores.push_back(diag_score + first_col_score);

            // for (size_t i = 0; i + 1 < other_values.size(); i++) {
            //     sorted_scores.push_back(other_values[i].first);
            // }
            for (auto& p : other_values) {
                sorted_scores.push_back(p.first);
            }
            if (q_block_idx == 0) {
                sorted_scores.pop_back();
            }
            // // Step4.1: 打印 sorted_scores
            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] sorted_scores: ";
            // for (size_t i = 0; i < sorted_scores.size(); i++) {
            //     std::cout << sorted_scores[i] << " ";
            // }
            // std::cout << std::endl;
            


            // Step5: 计算 cumsum_without_self: cumsum of right-shifted sorted_scores
            std::vector<float> cumsum_without_self(sorted_scores.size(), 0.0);
            float running = 0.0;
            for (size_t i = 0; i < sorted_scores.size(); ++i) {
                cumsum_without_self[i] = running;   // 等价于 Python 的 cat([0, ...]) then cumsum, i.e. previous sum
                running += sorted_scores[i];
            }

            // // 打印 cumsum_without_self（调试用）
            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] cumsum: ";
            // for (size_t i = 0; i < cumsum_without_self.size(); i++) {
            //     std::cout << cumsum_without_self[i] << " ";
            // }
            // std::cout << std::endl;

            // Step6: 生成 required_sum（基于整行）
            size_t offset_row_start = head_idx * q_block_num * k_block_num
                                      + q_block_idx * k_block_num;
            float row_sum = 0.0;
            for (size_t k = 0; k < k_block_num; k++) {
                row_sum += static_cast<float>(blocked_attention_scores_data[offset_row_start + k]);
            }
            float required_sum = row_sum * m_threshold;
            std::cout << "required_sum: " << required_sum << std::endl;


            // Step7: 构建 index_mask
            std::vector<bool> index_mask(cumsum_without_self.size(), false);
            for (size_t i = 0; i < cumsum_without_self.size(); i++) {
                index_mask[i] = (cumsum_without_self[i] < required_sum);
            }

            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] index_mask: ";
            // for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
            //     std::cout << (index_mask[head_idx][q_block_idx][k_block_idx] ? "1 " : "0 ");
            // }
            // std::cout << std::endl;


            // Step8: 构建 index 向量（torch.where(index_mask, index, 0)）
            std::vector<size_t> index(index_mask.size(), 0);
            for (size_t i = 0; i < index_mask.size(); i++) {
                if (index_mask[i]) {
                    // 索引来源：sorted_scores[0], [1], ... 对应哪些 k_block？
                    // 前两个为 [0:padding], [1:diag+col0], 后续对应 other_values
                    if (i == 0) index[i] = 0;          // dummy
                    else if (i == 1) index[i] = diagonal_k;
                    else if (i - 2 < other_values.size())
                        index[i] = other_values[i - 2].second;
                    else
                        index[i] = 0;
                }
            }

            // Step9: 模拟 Python mask[:, torch.arange(...), index] = True
            // 即对每个 (head_idx, q_block_idx)，将 index[i] 对应的 k_block 置 True
            for (size_t i = 0; i < index.size(); i++) {
                size_t k_block_idx = index[i];
                if (index_mask[i] && k_block_idx < k_block_num) {
                    mask[head_idx][q_block_idx][k_block_idx] = true;
                }
            }


            // 打印 cumsum_without_self（调试用）
            std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] required_sum: " << required_sum << std::endl;

            // Step7: 根据 index_mask 更新 mask
            // 注意：sorted_scores 带有两个前缀项，因此 other_values 对应的 sorted_scores 索引从 2 开始
            // but we must only iterate the number of other_values actually included in sorted_scores.
            // size_t included_count = 0;
            // if (sorted_scores.size() > 2) {
            //     included_count = sorted_scores.size() - 2;
            // } else {
            //     included_count = 0;
            // }


            // // 🔹 Step10.1: 打印当前 head、q_block 的 mask
            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] mask: ";
            // for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
            //     std::cout << (mask[head_idx][q_block_idx][k_block_idx] ? "1 " : "0 ");
            // }
            // std::cout << std::endl;

            // for (size_t i = 0; i < included_count; ++i) {
            //     size_t idx_in_sorted = 2 + i;
            //     // 安全检查（通常应该不越界）
            //     if (idx_in_sorted < cumsum_without_self.size()) {
            //         if (cumsum_without_self[idx_in_sorted] < required_sum) {
            //             size_t k_block_idx = other_values[i].second;
            //             mask[head_idx][q_block_idx][k_block_idx] = true;
            //         }
            //     } else {
            //         // 如果发生越界，输出调试信息（不抛异常以便继续调试）
            //         std::cerr << "Debug: idx_in_sorted out of range: " << idx_in_sorted
            //                   << " cumsum_size=" << cumsum_without_self.size()
            //                   << " other_values.size()=" << other_values.size()
            //                   << " sorted_scores.size()=" << sorted_scores.size() << std::endl;
            //     }
            // }

            // // Step8: 保留左侧邻域（stride=2）
            // for (int s = 1; s <= 2; s++) {
            //     if (q_block_idx >= static_cast<size_t>(s)) {
            //         std::cout << head_idx << " " << q_block_idx << " " << q_block_idx - s << std::endl;
            //         mask[head_idx][q_block_idx][q_block_idx - s] = true;
            //     }
            // }

            // // Step9: 保留右侧邻域（受 causal 约束）
            // for (int s = 1; s <= 2; s++) {
            //     size_t right = q_block_idx + s;
            //     if (right < k_block_num) {
            //         std::cout << head_idx << " " << q_block_idx << " " << right << std::endl;
            //         mask[head_idx][q_block_idx][right] = true;
            //     }
            // }

            // // Step10: causal mask，删除未来块
            // for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
            //     if (k_block_idx > q_block_idx)
            //         mask[head_idx][q_block_idx][k_block_idx] = false;
            // }

            // 🔹 Step10.1: 打印当前 head、q_block 的 mask
            // std::cout << "[Head " << head_idx << " Q=" << q_block_idx << "] mask: ";
            // for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
            //     std::cout << (mask[head_idx][q_block_idx][k_block_idx] ? "1 " : "0 ");
            // }
            // std::cout << std::endl;

            // Step11: 收集 mask 为 true 的块到 retval
            for (size_t k_block_idx = 0; k_block_idx < k_block_num; k_block_idx++) {
                if (mask[head_idx][q_block_idx][k_block_idx])
                    retval[head_idx].insert({q_block_idx, k_block_idx});
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