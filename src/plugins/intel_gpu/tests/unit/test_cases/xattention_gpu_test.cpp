// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <intel_gpu/primitives/activation.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/gemm.hpp>
#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/paged_attention.hpp>
#include <intel_gpu/primitives/permute.hpp>
#include <intel_gpu/primitives/reorder.hpp>
#include <intel_gpu/primitives/softmax.hpp>

#include "paged_attention_gpu_test.hpp"
#include "random_generator.hpp"
#include "test_utils.h"

using namespace cldnn;
using namespace ov::intel_gpu;
using namespace ::tests;

using Shape = std::vector<size_t>;

using CMXAttentionBlockIndex = std::pair<size_t, size_t>;  // .first is the *query* dimension block index, .second is *key*
using CMXAttentionRetainedBlockIndices = std::set<CMXAttentionBlockIndex>;
using CMXAttentionRetainedBlockIndicesForAllHeads = std::vector<CMXAttentionRetainedBlockIndices>;

template <typename T>
class CMXAttentionBlockSelector {
public:
    CMXAttentionBlockSelector(double threshold, size_t block_size, size_t stride) : m_threshold(threshold), m_block_size(block_size), m_stride(stride) {
        OPENVINO_ASSERT(m_block_size % m_stride == 0);
    }

    void diagonal_reshape(const T* input_data, const Shape& input_shape, T* output_data, const Shape& out_shape, bool is_antidiagonal) {
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

    void diagonal_reshape_kdb1_no_batch(const T* input_data,
                                        const std::vector<size_t>& input_shape,  // [H, Q_orig, dim]
                                        T* output_data,
                                        const std::vector<size_t>& output_shape) {
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

        ov::reference::matmul(reshaped_query_data, reshaped_key_data, out, reshaped_query_shape, reshaped_key_shape, out_shape, false, true);

        size_t out_size = out_shape[0] * out_shape[1] * out_shape[2];

        for (size_t i = 0; i < out_size; i++) {
            out[i] = out[i] / std::sqrt(reshaped_query_shape[2] * m_stride);
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
                    *target_block_sum_ptr += *(attention_scores_data + in_head_offset + query_len_idx * attention_scores_shape[2] + key_len_idx);
                }
            }
        }
    }

    CMXAttentionRetainedBlockIndicesForAllHeads get_block_indices_to_keep(T* blocked_attention_scores_data, const Shape& blocked_attention_scores_shape) {
        OPENVINO_ASSERT(blocked_attention_scores_shape.size() == 3, "Expected shape [num_heads, q_block_num, k_block_num]");

        size_t num_heads = blocked_attention_scores_shape[0];
        size_t q_block_num = blocked_attention_scores_shape[1];
        size_t k_block_num = blocked_attention_scores_shape[2];

        CMXAttentionRetainedBlockIndicesForAllHeads retval(num_heads);

        std::vector<std::vector<std::vector<bool>>> mask(num_heads, std::vector<std::vector<bool>>(q_block_num, std::vector<bool>(k_block_num, false)));

        for (size_t head_idx = 0; head_idx < num_heads; head_idx++) {
            for (size_t q_block_idx = 0; q_block_idx < q_block_num; q_block_idx++) {
                size_t diagonal_k = q_block_idx;
                if (diagonal_k < k_block_num) {
                    mask[head_idx][q_block_idx][diagonal_k] = true;
                }
                // Step1: Keep the first column
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

                // Step8: Ceate index
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

    CMXAttentionRetainedBlockIndicesForAllHeads select_blocks(const T* query_data, const Shape& query_shape, const T* key_data, const Shape& key_shape) {
        OPENVINO_ASSERT(query_shape.size() == 3);
        OPENVINO_ASSERT(key_shape.size() == 3);
        OPENVINO_ASSERT(key_shape[0] == query_shape[0]);
        OPENVINO_ASSERT(key_shape[2] == query_shape[2]);
        OPENVINO_ASSERT(query_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(key_shape[1] % m_stride == 0);
        OPENVINO_ASSERT(query_shape[1] % m_block_size == 0);
        OPENVINO_ASSERT(key_shape[1] % m_block_size == 0);

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
        size_t k_reshaped_num_to_pad = k_num_to_pad / m_stride;
        size_t k_reshaped_seq_len = (k_len + k_num_to_pad) / m_stride;

        Shape reshaped_query_shape = {num_heads, query_shape[1] / m_stride, head_dim * m_stride};
        auto q_buf = allocate_buf(reshaped_query_shape);
        diagonal_reshape_kdb1_no_batch(query_data, query_shape, q_buf.get(), reshaped_query_shape);
        Shape reshaped_key_shape = {num_heads, pad_key_shape[1] / m_stride, head_dim * m_stride};
        auto k_buf = allocate_buf(reshaped_key_shape);
        diagonal_reshape(pad_key_buf.get(), pad_key_shape, k_buf.get(), reshaped_key_shape, false);
        Shape transpose_matmul_scaled_shape = {num_heads, query_shape[1] / m_stride, pad_key_shape[1] / m_stride};
        auto qk_buf = allocate_buf(transpose_matmul_scaled_shape);

        transpose_matmul_scale(q_buf.get(), k_buf.get(), reshaped_query_shape, reshaped_key_shape, qk_buf.get(), transpose_matmul_scaled_shape);
        q_buf.reset();
        k_buf.reset();

        Shape causal_mask_shape = {num_heads, reshaped_chunk_size, reshaped_chunk_size * k_chunk_num};
        auto causal_mask_buf = allocate_buf(causal_mask_shape);
        std::fill(causal_mask_buf.get(), causal_mask_buf.get() + ov::shape_size(causal_mask_shape), T(0));
        if (k_reshaped_num_to_pad) {
            for (size_t h = 0; h < num_heads; h++)
                for (size_t q = 0; q < reshaped_chunk_size; q++)
                    for (size_t k = k_reshaped_seq_len - k_reshaped_num_to_pad; k < k_reshaped_seq_len; k++) {
                        size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) + q * (reshaped_chunk_size * k_chunk_num) + k;

                        causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                    }
        }

        size_t chunk_start = offset_token_chunk_num * reshaped_chunk_size;
        size_t chunk_end = chunk_start + reshaped_chunk_size;

        for (size_t h = 0; h < num_heads; h++) {
            for (size_t q = 0; q < reshaped_chunk_size; q++) {
                for (size_t k = q + 1; k < reshaped_chunk_size; k++) {
                    size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) + q * (reshaped_chunk_size * k_chunk_num) + chunk_start + k;
                    causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                }
            }
        }

        for (size_t h = 0; h < num_heads; h++) {
            for (size_t q = 0; q < reshaped_chunk_size; q++) {
                for (size_t k = chunk_end; k < reshaped_chunk_size * k_chunk_num; k++) {
                    size_t offset = h * reshaped_chunk_size * (reshaped_chunk_size * k_chunk_num) + q * (reshaped_chunk_size * k_chunk_num) + k;
                    causal_mask_buf.get()[offset] = std::numeric_limits<T>::lowest();
                }
            }
        }
        size_t out_size = transpose_matmul_scaled_shape[0] * transpose_matmul_scaled_shape[1] * transpose_matmul_scaled_shape[2];

        for (size_t i = 0; i < out_size; i++) {
            qk_buf.get()[i] += causal_mask_buf.get()[i];
        }

        causal_mask_buf.reset();
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

struct xAttentionReference {
    xAttentionReference(PagedAttentionManager& pam) : pam(pam), test_engine(pam.test_engine), test_stream(pam.test_stream) {}

    std::pair<std::vector<ov::float16>, std::vector<ov::float16>> get_reference() {
        std::vector<ov::float16> ref_data_output;
        std::vector<ov::float16> ref_scores_output;

        for (size_t i = 0; i < pam.subsequence_descs.size(); i++) {
            const auto& subsequence_desc = pam.subsequence_descs[i];
            const auto kv_seq_len = subsequence_desc.num_tokens + subsequence_desc.past_len;

            auto key_data = pam.key_data[i];
            if (pam.rotation_config.apply_rotation) {
                auto blocks_start = pam.block_indices_begins[i];
                auto blocks_end = pam.block_indices_begins[i + 1];

                std::vector<int> block_indices(pam.block_indices.begin() + blocks_start, pam.block_indices.begin() + blocks_end);

                for (const auto& block_idx : block_indices) {
                    auto it = std::find(pam.rotated_block_indices.begin(), pam.rotated_block_indices.end(), block_idx);
                    if (it != pam.rotated_block_indices.end()) {
                        int index = std::distance(pam.rotated_block_indices.begin(), it);
                        int subsequence_rotated_block_idx = *it - blocks_start;

                        rotate_block(key_data,
                                     pam.rotation_deltas,
                                     pam.rotation_trig_lut,
                                     index,
                                     subsequence_rotated_block_idx,
                                     pam.num_heads,
                                     pam.k_head_size,
                                     pam.block_size,
                                     pam.rotation_config.per_block);
                    }
                }
            }

            auto window_size = pam.has_score_aggregation ? pam.score_aggregation[i] : 1;

            auto subsequence_ref_results = run_reference(pam.query_data[i],
                                                         key_data,
                                                         pam.value_data[i],
                                                         subsequence_desc.num_tokens,
                                                         kv_seq_len,
                                                         pam.num_heads,
                                                         pam.k_head_size,
                                                         pam.v_head_size,
                                                         window_size,
                                                         pam.sliding_window_size,
                                                         pam.get_default_scale());

            // concatenate all subsequences into one vector
            ref_data_output.insert(ref_data_output.end(), subsequence_ref_results.first.begin(), subsequence_ref_results.first.end());
            ref_scores_output.insert(ref_scores_output.end(), subsequence_ref_results.second.begin(), subsequence_ref_results.second.end());
        }

        return {ref_data_output, ref_scores_output};
    }

private:
    std::pair<std::vector<ov::float16>, std::vector<ov::float16>> run_reference(const std::vector<ov::float16>& query_data,
                                                                                   const std::vector<ov::float16>& key_data,
                                                                                   const std::vector<ov::float16>& value_data,
                                                                                   int num_queries,
                                                                                   int num_keys,
                                                                                   int num_heads,
                                                                                   int k_head_size,
                                                                                   int v_head_size,
                                                                                   int window_size,
                                                                                   int sliding_window_size,
                                                                                   float scale,
                                                                                   double threshold = 0.9,
                                                                                   size_t block_size = 128,
                                                                                   size_t stride = 16) {
        auto query_shape_bfyx = ov::PartialShape{1, num_queries, num_heads, k_head_size};
        auto key_shape_bfyx = ov::PartialShape{1, num_keys, num_heads, k_head_size};
        auto value_shape_bfyx = ov::PartialShape{1, num_keys, num_heads, v_head_size};

        auto query_layout = layout{query_shape_bfyx, data_types::f16, format::bfyx};
        auto key_layout = layout{key_shape_bfyx, data_types::f16, format::bfyx};
        auto value_layout = layout{value_shape_bfyx, data_types::f16, format::bfyx};

        OPENVINO_ASSERT(query_layout.count() == query_data.size());
        OPENVINO_ASSERT(key_layout.count() == key_data.size());
        OPENVINO_ASSERT(value_layout.count() == value_data.size());

        auto query_mem = test_engine.allocate_memory(query_layout);
        auto key_mem = test_engine.allocate_memory(key_layout);
        auto value_mem = test_engine.allocate_memory(value_layout);

        set_values(query_mem, query_data);
        set_values(key_mem, key_data);
        set_values(value_mem, value_data);

        std::vector<ov::float16> query_data_3d(num_heads * num_queries * k_head_size);
        std::vector<ov::float16> key_data_3d(num_heads * num_keys * k_head_size);

        for (int h = 0; h < num_heads; h++) {
            for (int q = 0; q < num_queries; q++) {
                for (int d = 0; d < k_head_size; d++) {
                    query_data_3d[h * num_queries * k_head_size + q * k_head_size + d] = query_data[q * num_heads * k_head_size + h * k_head_size + d];
                }
            }
        }

        for (int h = 0; h < num_heads; h++) {
            for (int k = 0; k < num_keys; k++) {
                for (int d = 0; d < k_head_size; d++) {
                    key_data_3d[h * num_keys * k_head_size + k * k_head_size + d] = key_data[k * num_heads * k_head_size + h * k_head_size + d];
                }
            }
        }

        ov::Shape query_shape_3d = {static_cast<size_t>(num_heads), static_cast<size_t>(num_queries), static_cast<size_t>(k_head_size)};
        ov::Shape key_shape_3d = {static_cast<size_t>(num_heads), static_cast<size_t>(num_keys), static_cast<size_t>(k_head_size)};

        CMXAttentionRetainedBlockIndicesForAllHeads retained_blocks;
    if (num_queries >= static_cast<int>(block_size)) {
        size_t padded_q = ((num_queries + block_size - 1) / block_size) * block_size;
        size_t padded_k = ((num_keys + block_size - 1) / block_size) * block_size;
        std::vector<ov::float16> query_padded(num_heads * padded_q * k_head_size, ov::float16(0));
        std::vector<ov::float16> key_padded(num_heads * padded_k * k_head_size, ov::float16(0));

        for (int h = 0; h < num_heads; ++h) {
            std::copy_n(&query_data_3d[h * num_queries * k_head_size],
                        num_queries * k_head_size,
                        &query_padded[h * padded_q * k_head_size]);
            std::copy_n(&key_data_3d[h * num_keys * k_head_size],
                        num_keys * k_head_size,
                        &key_padded[h * padded_k * k_head_size]);
        }

        ov::Shape query_shape_padded = {static_cast<size_t>(num_heads), padded_q, static_cast<size_t>(k_head_size)};
        ov::Shape key_shape_padded = {static_cast<size_t>(num_heads), padded_k, static_cast<size_t>(k_head_size)};

        std::vector<float> query_padded_f32(query_padded.size());
        std::vector<float> key_padded_f32(key_padded.size());
        for (size_t i = 0; i < query_padded.size(); ++i)
            query_padded_f32[i] = static_cast<float>(query_padded[i]);
        for (size_t i = 0; i < key_padded.size(); ++i)
            key_padded_f32[i] = static_cast<float>(key_padded[i]);

        CMXAttentionBlockSelector<float> selector(threshold, block_size, stride);
        retained_blocks = selector.select_blocks(query_padded_f32.data(), query_shape_padded,
                                                 key_padded_f32.data(), key_shape_padded);
    }
        auto mask_mem = get_mask_mem_combined_multi_head(num_queries, num_keys, num_heads, sliding_window_size, retained_blocks, block_size);

        topology topology;
        topology.add(input_layout("query", query_layout),
                     input_layout("key", key_layout),
                     input_layout("value", value_layout),
                     data("mask", mask_mem),
                     permute("query_transposed", input_info("query"), {0, 2, 1, 3}),
                     permute("key_transposed", input_info("key"), {0, 2, 1, 3}),
                     permute("value_transposed", input_info("value"), {0, 2, 1, 3}),
                     gemm("qk_gemm", {input_info("query_transposed"), input_info("key_transposed")}, data_types::f16, false, true, scale),
                     eltwise("eltwise", {input_info("qk_gemm"), input_info("mask")}, eltwise_mode::sum),
                     softmax("softmax", input_info("eltwise"), -1),
                     gemm("qkv_gemm", {input_info("softmax"), input_info("value_transposed")}, data_types::f16, false, false),
                     permute("qkv_gemm_transposed", input_info("qkv_gemm"), {0, 2, 1, 3}),
                     reorder("output_data", input_info("qkv_gemm_transposed"), format::bfyx, data_types::f16),
                     reorder("scores_data", input_info("softmax"), format::bfyx, data_types::f16));

        ExecutionConfig config = get_test_default_config(test_engine);
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));

        network::ptr network = get_network(test_engine, topology, config, get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);

        auto outputs = network->execute();

        auto output_data_mem = outputs.at("output_data").get_memory();
        auto output_scores_mem = outputs.at("scores_data").get_memory();

        return {get_output_data_vec(output_data_mem, num_queries, v_head_size, num_heads),
                get_output_scores_vec(output_scores_mem, window_size, num_queries, num_keys, num_heads)};
    }

    std::vector<ov::float16> get_output_scores_vec(memory::ptr scores_output, int window_size, int num_queries, int num_keys, int num_heads) {
        OPENVINO_ASSERT(scores_output->count() == static_cast<size_t>(num_heads * num_queries * num_keys));

        std::vector<ov::float16> output_scores(num_keys, 0);
        mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output, test_stream);
        for (int row_idx = 0; row_idx < window_size; row_idx++) {
            for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                for (int score_idx = 0; score_idx < num_keys; score_idx++) {
                    auto scores_offset = head_idx * num_queries * num_keys + (num_queries - window_size + row_idx) * num_keys + score_idx;
                    output_scores[score_idx] += mem_ptr[scores_offset];
                }
            }
        }

        return output_scores;
    }

    std::vector<ov::float16> get_output_data_vec(memory::ptr data_output, int num_queries, int k_head_size, int num_heads) {
        OPENVINO_ASSERT(data_output->count() == static_cast<size_t>(num_queries * num_heads * k_head_size));

        std::vector<ov::float16> output_data(data_output->count());
        mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output, test_stream);
        for (size_t i = 0; i < data_output->count(); i++)
            output_data[i] = mem_ptr[i];

        return output_data;
    }

    memory::ptr get_mask_mem_combined_multi_head(int num_queries,
                                                 int num_keys,
                                                 int num_heads,
                                                 int sliding_window_size,
                                                 const CMXAttentionRetainedBlockIndicesForAllHeads& retained_blocks,
                                                 int block_size) {
        auto mask_shape = ov::PartialShape{1, num_heads, num_queries, num_keys};
        auto mask_layout = layout{mask_shape, data_types::f16, format::bfyx};
        auto mask_mem = test_engine.allocate_memory(mask_layout);

        mem_lock<ov::float16> mem_ptr(mask_mem, test_stream);

        for (int h = 0; h < num_heads; h++) {
            if (retained_blocks.empty() || retained_blocks[h].empty()) {
                for (int i = 0; i < num_queries; i++) {
                    for (int j = 0; j < num_keys; j++) {
                        ov::float16 value = ov::float16(0.f);
                        if (sliding_window_size == 0) {
                            int past_len = num_keys - num_queries + 1;
                            if (j >= past_len + i)
                                value = std::numeric_limits<ov::float16>::lowest();
                        } else {
                            int sliding_left = num_keys - num_queries - sliding_window_size + 1;
                            int past_len = num_keys - num_queries + 1;
                            bool is_min;
                            if (num_queries == num_keys) {
                                is_min = (j >= sliding_left + i) && (j <= i) ? 0 : 1;
                            } else {
                                is_min = (j >= sliding_left + i) && (j < past_len + i) ? 0 : 1;
                            }
                            if (is_min)
                                value = std::numeric_limits<ov::float16>::lowest();
                        }
                        mem_ptr[h * num_queries * num_keys + i * num_keys + j] = value;
                    }
                }
                continue;
            }

            for (int i = 0; i < num_queries; i++) {
                for (int j = 0; j < num_keys; j++) {
                    mem_ptr[h * num_queries * num_keys + i * num_keys + j] = std::numeric_limits<ov::float16>::lowest();
                }
            }

            for (int i = 0; i < num_queries; i++) {
                int left_idx = 0;
                int right_idx = 0;

                if (sliding_window_size == 0) {
                    int past_len = num_keys - num_queries + 1;
                    right_idx = past_len + i - 1;
                    left_idx = 0;
                } else {
                    int sliding_left = num_keys - num_queries - sliding_window_size + 1;
                    int past_len = num_keys - num_queries + 1;
                    if (num_queries == num_keys) {
                        left_idx = sliding_left + i;
                        right_idx = i;
                    } else {
                        left_idx = sliding_left + i;
                        right_idx = past_len + i - 1;
                    }
                }

                left_idx = std::max(0, left_idx);
                right_idx = std::min(num_keys - 1, right_idx);

                for (const auto& [q_block_idx, k_block_idx] : retained_blocks[h]) {
                    int q_start = q_block_idx * block_size;
                    int q_end = std::min(q_start + block_size, num_queries);
                    int k_start = k_block_idx * block_size;
                    int k_end = std::min(k_start + block_size, num_keys);

                    if (i < q_start || i >= q_end)
                        continue;

                    for (int j = k_start; j < k_end; j++) {
                        if (j >= left_idx && j <= right_idx) {
                            mem_ptr[h * num_queries * num_keys + i * num_keys + j] = ov::float16(0.f);
                        }
                    }
                }
            }
        }

        return mask_mem;
    }

    void rotate_block(std::vector<ov::float16>& cache_data,
                      std::vector<int> rotation_deltas,
                      std::vector<ov::float16> rotation_trig_lut_mem,
                      int rotated_block_idx,
                      int subsequence_rotated_block_idx,
                      int num_heads,
                      int k_head_size,
                      int block_size,
                      bool per_block) {
        // cache_data shape: [1, num_tokens, num_heads, k_head_size]
        int start_token_idx = subsequence_rotated_block_idx * block_size;

        for (int token_idx = 0; token_idx < block_size; token_idx++) {
            auto rotation_deltas_offset = per_block ? rotated_block_idx : rotated_block_idx * block_size + token_idx;
            auto rotation_trig_lut_idx = rotation_deltas[rotation_deltas_offset];
            for (int head_idx = 0; head_idx < num_heads; head_idx++) {
                for (int k_head_size_idx = 0; k_head_size_idx < k_head_size / 2; k_head_size_idx++) {
                    auto input_offset = (start_token_idx + token_idx) * num_heads * k_head_size + head_idx * k_head_size + k_head_size_idx;

                    auto cache_value_0 = cache_data[input_offset];
                    auto cache_value_1 = cache_data[input_offset + k_head_size / 2];

                    ov::float16 rotation_value_cos = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx];
                    ov::float16 rotation_value_sin = rotation_trig_lut_mem[rotation_trig_lut_idx * k_head_size + k_head_size_idx + k_head_size / 2];

                    cache_data[input_offset] = cache_value_0 * rotation_value_cos - cache_value_1 * rotation_value_sin;
                    cache_data[input_offset + k_head_size / 2] = cache_value_0 * rotation_value_sin + cache_value_1 * rotation_value_cos;
                }
            }
        }
    }

    PagedAttentionManager& pam;
    cldnn::engine& test_engine;
    cldnn::stream& test_stream;
};

template <typename T>
struct xAttentionTest : public ::testing::TestWithParam<T> {
public:
    random_generator rg;
    cldnn::engine& engine = get_test_engine();
    float tolerance = 2e-3;

    void SetUp() override {
        rg.set_seed(GET_SUITE_NAME);
    }

    void execute(T& p) {
        PagedAttentionManager pam(rg,
                                  get_test_engine(),
                                  get_test_stream(),
                                  p.subsequences,
                                  p.num_heads,
                                  p.k_head_size,
                                  p.v_head_size,
                                  p.block_size,
                                  p.sliding_window_size,
                                  p.kv_cache_compression,
                                  p.key_cache_quant_mode,
                                  p.scores_mode == ScoresMode::SNAPKV,
                                  p.rotation_config);

        if (p.kv_cache_compression)
            tolerance = 25e-3;

        auto query_mem = pam.get_query_memory();
        auto key_mem = pam.get_key_memory();
        auto value_mem = pam.get_value_memory();

        auto key_cache_mem = pam.get_key_cache_memory();
        auto value_cache_mem = pam.get_value_cache_memory();

        auto past_lens_mem = pam.get_past_lens_memory();
        auto subsequence_begins_mem = pam.get_subsequence_begins_memory();
        auto block_indices_mem = pam.get_block_indices_memory();
        auto block_indices_begins_mem = pam.get_block_indices_begins_memory();

        auto scale_mem = pam.get_scale_memory();
        auto sliding_window_mem = pam.get_sliding_window_memory();
        auto alibi_mem = pam.get_alibi_memory();
        auto max_context_len_mem = pam.get_max_context_len_memory();

        // scores calculation related memory buffers
        auto score_aggregation_mem = pam.get_score_aggregation();

        // cache rotation related memory buffers
        auto rotated_block_indices_mem = pam.get_rotated_block_indices_memory();
        auto rotation_deltas_mem = pam.get_rotation_deltas_memory();
        auto rotation_trig_lut_mem = pam.get_rotation_trig_lut_memory();

        auto xattention_threshold_mem = pam.get_xattention_threshold_memory();
        auto xattention_block_size_mem = pam.get_xattention_block_size_memory();
        auto xattention_stride_mem = pam.get_xattention_stride_memory();

        auto query_layout = query_mem->get_layout();
        auto key_layout = key_mem->get_layout();
        auto value_layout = value_mem->get_layout();
        auto key_cache_layout = key_cache_mem->get_layout();
        auto value_cache_layout = value_cache_mem->get_layout();
        auto past_lens_layout = past_lens_mem->get_layout();
        auto subsequence_begins_layout = subsequence_begins_mem->get_layout();
        auto block_indices_layout = block_indices_mem->get_layout();
        auto block_indices_begins_layout = block_indices_begins_mem->get_layout();
        auto scale_layout = scale_mem->get_layout();
        auto sliding_window_layout = sliding_window_mem->get_layout();
        auto alibi_layout = alibi_mem->get_layout();
        auto max_context_len_layout = max_context_len_mem->get_layout();
        auto score_aggregation_window_layout = score_aggregation_mem->get_layout();
        auto rotated_block_indices_layout = rotated_block_indices_mem->get_layout();
        auto rotation_deltas_layout = rotation_deltas_mem->get_layout();
        auto rotation_trig_lut_layout = rotation_trig_lut_mem->get_layout();
        auto xattention_threshold_layout = xattention_threshold_mem->get_layout();
        auto xattention_block_size_layout = xattention_block_size_mem->get_layout();
        auto xattention_stride_layout = xattention_stride_mem->get_layout();

        // make layouts dynamic
        query_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
        key_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.k_head_size });
        value_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads * p.v_head_size });
#if ENABLE_PA_CM_PATH
        key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.block_size, p.k_head_size });
#else
        key_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.k_head_size, p.block_size });
#endif
        value_cache_layout.set_partial_shape(ov::PartialShape{ -1, p.num_heads, p.block_size, p.v_head_size });
        past_lens_layout.set_partial_shape(ov::PartialShape{ -1 });
        subsequence_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        block_indices_begins_layout.set_partial_shape(ov::PartialShape{ -1 });
        score_aggregation_window_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotated_block_indices_layout.set_partial_shape(ov::PartialShape{ -1 });
        rotation_deltas_layout.set_partial_shape(ov::PartialShape{ -1, -1 });
        rotation_trig_lut_layout.set_partial_shape(ov::PartialShape{ -1, p.k_head_size });
        xattention_threshold_layout.set_partial_shape(ov::PartialShape{ -1 });

        if (p.dynamic_paddings) {
            const auto padding_axis = 1;
            const auto pad_before = p.k_head_size;
            const auto pad_after = p.k_head_size * 2;

            query_layout.data_padding._dynamic_dims_mask[padding_axis] = 1;

            auto query_data_layout = query_mem->get_layout();
            auto padded_query_data_layout = query_data_layout;
            padded_query_data_layout.data_padding._lower_size[padding_axis] = pad_before;
            padded_query_data_layout.data_padding._upper_size[padding_axis] = pad_after;

            auto new_query_memory = get_test_engine().allocate_memory(padded_query_data_layout, false);

            mem_lock<ov::float16> query_mem_lock(query_mem, get_test_stream());
            mem_lock<ov::float16> new_query_mem_lock(new_query_memory, get_test_stream());

            auto query_data_shape = query_data_layout.get_shape();
            for (size_t b = 0; b < query_data_shape[0]; b++) {
                for (size_t f = 0; f < query_data_shape[1]; f++) {
                    auto input_offset =
                        query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));
                    auto output_offset =
                        padded_query_data_layout.get_linear_offset(cldnn::tensor(static_cast<int32_t>(b), static_cast<int32_t>(f), 0, 0, 0, 0));

                    new_query_mem_lock[output_offset] = query_mem_lock[input_offset];
                }
            }
            query_mem = new_query_memory;
        }

        std::vector<input_info> pa_inputs = {
            input_info("query"),
            input_info("key"),
            input_info("value"),
            input_info("key_cache"),
            input_info("value_cache"),
            input_info("past_lens"),
            input_info("subsequence_begins"),
            input_info("block_indices"),
            input_info("block_indices_begins"),
            input_info("scale"),
            input_info("sliding_window"),
            input_info("alibi"),
            input_info("max_context_len"),
            input_info("score_aggregation_window"),
            input_info("rotated_block_indices"),
            input_info("rotation_deltas"),
            input_info("rotation_trig_lut_modified"),
            input_info("xattention_threshold"),
            input_info("xattention_block_size"),
            input_info("xattention_stride"),
        };

        auto pa_prim = paged_attention("paged_attention", pa_inputs);

        pa_prim.k_head_size = p.k_head_size;
        pa_prim.v_head_size = p.v_head_size;
        pa_prim.kv_heads_num = p.num_heads;
        pa_prim.heads_num = p.num_heads;
        pa_prim.scale_val = pam.get_default_scale();
        pa_prim.has_alibi = false;
        pa_prim.num_outputs = p.scores_mode == ScoresMode::DISABLED ? 1 : 2;
        pa_prim.has_rotated_blocks = p.rotation_config.apply_rotation;
        pa_prim.has_score_aggregation = p.scores_mode == ScoresMode::SNAPKV;
        pa_prim.sliding_window = p.sliding_window_size;
        pa_prim.is_key_by_channel = (p.key_cache_quant_mode == ov::internal::CacheQuantMode::BY_CHANNEL);

        topology topology;

        topology.add(
            input_layout("query", query_layout),
            input_layout("key", key_layout),
            input_layout("value", value_layout),
            input_layout("key_cache", key_cache_layout),
            input_layout("value_cache", value_cache_layout),
            input_layout("past_lens", past_lens_layout),
            input_layout("subsequence_begins", subsequence_begins_layout),
            input_layout("block_indices", block_indices_layout),
            input_layout("block_indices_begins", block_indices_begins_layout),
            input_layout("scale", scale_layout),
            input_layout("sliding_window", sliding_window_layout),
            input_layout("alibi", alibi_layout),
            input_layout("max_context_len", max_context_len_layout),
            input_layout("score_aggregation_window", score_aggregation_window_layout),
            pa_prim,
            reorder("output_data", input_info("paged_attention", 0), format::bfyx, data_types::f16)
        );

        if (p.scores_mode != ScoresMode::DISABLED) {
            topology.add(reorder("output_scores", input_info("paged_attention", 1), format::bfyx, data_types::f16));
        }

        {
            topology.add(input_layout("rotated_block_indices", rotated_block_indices_layout));
            topology.add(input_layout("rotation_deltas", rotation_deltas_layout));
            topology.add(input_layout("rotation_trig_lut", rotation_trig_lut_layout));

            // add dummy activation operation to simulate an empty PA `rotation_trig_lut` buffer for shapes like [0, k_head_size]
            topology.add(activation("rotation_trig_lut_modified", input_info("rotation_trig_lut"), activation_func::none));

            topology.add(input_layout("xattention_threshold", xattention_threshold_layout));
            topology.add(input_layout("xattention_block_size", xattention_block_size_layout));
            topology.add(input_layout("xattention_stride", xattention_stride_layout));
        }

        ExecutionConfig config = get_test_default_config(get_test_engine());
        config.set_property(ov::intel_gpu::optimize_data(true));
        config.set_property(ov::intel_gpu::allow_new_shape_infer(true));
        // FlashAttn v1 or v2?
        config.set_property(ov::intel_gpu::could_use_flashattn_v2(p.disable_flashattn_v2));
        config.set_property(ov::internal::key_cache_quant_mode(p.key_cache_quant_mode));
        network::ptr network = get_network(get_test_engine(), topology, config, get_test_stream_ptr(), false);
        network->set_input_data("query", query_mem);
        network->set_input_data("key", key_mem);
        network->set_input_data("value", value_mem);
        network->set_input_data("key_cache", key_cache_mem);
        network->set_input_data("value_cache", value_cache_mem);
        network->set_input_data("past_lens", past_lens_mem);
        network->set_input_data("subsequence_begins", subsequence_begins_mem);
        network->set_input_data("block_indices", block_indices_mem);
        network->set_input_data("block_indices_begins", block_indices_begins_mem);
        network->set_input_data("scale", scale_mem);
        network->set_input_data("sliding_window", sliding_window_mem);
        network->set_input_data("alibi", alibi_mem);
        network->set_input_data("max_context_len", max_context_len_mem);
        network->set_input_data("score_aggregation_window", score_aggregation_mem);
        network->set_input_data("rotated_block_indices", rotated_block_indices_mem);
        network->set_input_data("rotation_deltas", rotation_deltas_mem);
        network->set_input_data("rotation_trig_lut", rotation_trig_lut_mem);
        network->set_input_data("xattention_threshold", xattention_threshold_mem);
        network->set_input_data("xattention_block_size", xattention_block_size_mem);
        network->set_input_data("xattention_stride", xattention_stride_mem);

        auto outputs = network->execute();

        cldnn::memory::ptr output_data_mem = nullptr;
        cldnn::memory::ptr output_scores_mem = nullptr;

        output_data_mem = outputs.at("output_data").get_memory();
        if (p.scores_mode != ScoresMode::DISABLED) {
            output_scores_mem = outputs.at("output_scores").get_memory();
        }
        auto ref_data = xAttentionReference(pam).get_reference();
        compare(output_data_mem, output_scores_mem, ref_data);
    }

    void compare(memory::ptr data_output_mem, memory::ptr scores_output_mem, std::pair<std::vector<ov::float16>, std::vector<ov::float16>> ref_data) {
        if (data_output_mem) {
            ASSERT_EQ(data_output_mem->count(), ref_data.first.size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(data_output_mem, get_test_stream());
            int mismatch_count = 0;
            for (size_t i = 0; i < data_output_mem->count(); i++) {
                if (std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(ref_data.first[i])) > tolerance) {
                    mismatch_count++;
                }
            }
            EXPECT_LE(mismatch_count, int(data_output_mem->count() * 0.02));
        }

        if (scores_output_mem) {
            ASSERT_EQ(scores_output_mem->count(), ref_data.second.size());
            mem_lock<ov::float16, mem_lock_type::read> mem_ptr(scores_output_mem, get_test_stream());
            int mismatch_count = 0;
            for (size_t i = 0; i < scores_output_mem->count(); i++) {
                if (std::fabs(static_cast<float>(mem_ptr[i]) - static_cast<float>(ref_data.second[i])) > tolerance) {
                    mismatch_count++;
                }
            }
            EXPECT_LE(mismatch_count, int(scores_output_mem->count() * 0.02));
        }
    }
};

struct xattention_test_params {
    std::vector<SubsequenceDescriptor> subsequences;
    int num_heads;
    int k_head_size;
    int v_head_size;
    int block_size;
    int sliding_window_size;
    bool kv_cache_compression;
    ov::internal::CacheQuantMode key_cache_quant_mode;
    bool dynamic_paddings;
    ScoresMode scores_mode;
    CacheRotationDescriptor rotation_config;
    bool disable_flashattn_v2;
};

class xattention_test : public xAttentionTest<xattention_test_params> {};
TEST_P(xattention_test, basic) {
    auto p = GetParam();

    execute(p);
}

const auto ENABLE_CACHE_COMPRESSION = true;
const auto DISABLE_CACHE_COMPRESSION = false;
const auto DISABLE_SCORES = ScoresMode::DISABLED;
const auto ENABLE_SCORES = ScoresMode::LAST_TOKEN;
const auto ENABLE_SCORES_SNAPKV = ScoresMode::SNAPKV;
const auto PER_BLOCK_ROTATION = CacheRotationDescriptor{true, true};
const auto PER_TOKEN_ROTATION = CacheRotationDescriptor{true, false};
const auto DISABLE_ROTATION = CacheRotationDescriptor{false, false};
const auto STATIC_INPUT_PAD = false;
const auto DYNAMIC_INPUT_PAD = true;
const auto ENABLE_FA_V2 = false;
const auto DISABLE_FA_V2 = true;

INSTANTIATE_TEST_SUITE_P(smoke_cm_xattention,
                         xattention_test,
                         ::testing::ValuesIn(std::vector<xattention_test_params>{

#if ENABLE_PA_CM_PATH
    /* without scores output, static input query paddings, single sequence, disable KV cache compression, k_head_size==v_head_size,
    token_size>=32, disable_mix_mode */
    xattention_test_params{ {{32, 0}},   2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token
    xattention_test_params{ {{4096, 0}},   2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, ENABLE_FA_V2 }, // 1st token

    xattention_test_params{ {{1, 31}},   2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
    xattention_test_params{ {{1, 32}},   2, 64, 64, 256, 0, DISABLE_CACHE_COMPRESSION, ov::internal::CacheQuantMode::BY_TOKEN, STATIC_INPUT_PAD, DISABLE_SCORES, DISABLE_ROTATION, DISABLE_FA_V2 }, // 2nd token
#endif
}));
