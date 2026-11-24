// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/qsa_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/qsa_prepare.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sparse_attention_indexer.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "ov_ops/rms.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "utils/precision_support.h"

namespace ov::test {

namespace {

constexpr float kEpsilon = 1e-6F;

// The model inputs, in the order of the Parameters.
enum InputIdx : size_t {
    kQuery,           // [B, L, H_iq, D]
    kRawKeys,         // [B, L, D]
    kRopeCos,         // [B, L, rotary_dim]
    kRopeSin,         // [B, L, rotary_dim]
    kKeyLength,       // [B]
    kQNormWeight,     // [D]
    kKNormWeight,     // [D]
    kBlockKeysCache,  // [B, blocks of the previous call, D]
    kPastRawKeys,     // [B, key_block_size - 1, D]
    kPastRope,        // [B, 2 * rotary_dim]
    kInputCount
};

// The model outputs, in the order of the Results.
enum OutputIdx : size_t {
    kSelectedIndices,  // [B, 1, L, block_topk]
    kSelectedCount,    // [B, 1, L]
    kBlockKeysOut,     // [B, blocks of the longest row, D]
    kPastRawKeysOut,   // [B, key_block_size - 1, D]
    kPastRopeOut       // [B, 2 * rotary_dim]
};

const char* scenario_name(QSAIndexerScenario scenario) {
    switch (scenario) {
    case QSAIndexerScenario::kReference:
        return "Reference";
    case QSAIndexerScenario::kNaNScores:
        return "NaNScores";
    case QSAIndexerScenario::kOverflowScores:
        return "OverflowScores";
    case QSAIndexerScenario::kTinyScores:
        return "TinyScores";
    case QSAIndexerScenario::kStateBatchMismatch:
        return "StateBatchMismatch";
    }
    return "";
}

std::vector<std::string> runtime_layer_types(const ov::CompiledModel& compiled_model) {
    std::vector<std::string> types;
    for (const auto& op : compiled_model.get_runtime_model()->get_ops()) {
        const auto& info = op->get_rt_info();
        const auto layer_type = info.find(ov::exec_model_info::LAYER_TYPE);
        if (layer_type != info.end()) {
            types.push_back(layer_type->second.as<std::string>());
        }
    }
    return types;
}

std::string vec_to_string(const std::vector<int32_t>& values) {
    std::ostringstream out;
    for (size_t i = 0; i < values.size(); i++) {
        out << (i > 0 ? "x" : "") << values[i];
    }
    return out.str();
}

ov::Tensor make_zero_tensor(const ov::element::Type& precision, const ov::Shape& shape) {
    ov::Tensor t(precision, shape);
    if (t.get_byte_size() != 0) {
        std::memset(t.data(), 0, t.get_byte_size());
    }
    return t;
}

// Host-vector reference of one call, from the QSAPrepare / SparseAttentionIndexer op specs: all the outputs follow from
// the inputs of the call, the state included (the previous call's outputs).
std::vector<ov::Tensor> calculate_qsa_indexer_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                                                   const std::shared_ptr<ov::Model>& function,
                                                   size_t block_topk,
                                                   bool query_norm,
                                                   bool query_rope) {
    const auto& params = function->get_parameters();
    const auto to_f32 = [&](size_t input) {
        return read_tensor_to_f32(inputs.at(params[input]));
    };
    const auto& query_tensor = inputs.at(params[kQuery]);
    const auto& query_shape = query_tensor.get_shape();
    const size_t batch = query_shape[0];
    const size_t L = query_shape[1];
    const size_t h_iq = query_shape[2];
    const size_t d_idx = query_shape[3];
    const size_t rotary_dim = inputs.at(params[kRopeCos]).get_shape()[2];
    const auto r = static_cast<int64_t>(inputs.at(params[kPastRawKeys]).get_shape()[1]) + 1;
    const auto data_type = query_tensor.get_element_type();
    const auto length = static_cast<int64_t>(L);
    const size_t h_ik = 1;
    const size_t h_sel = 1;
    const float eps = kEpsilon;
    const float scale = 1.0F / std::sqrt(static_cast<float>(d_idx));

    const auto* key_length_data = inputs.at(params[kKeyLength]).data<const int32_t>();
    const std::vector<int32_t> key_length(key_length_data, key_length_data + batch);
    const auto q_idx = to_f32(kQuery);
    const auto k_idx = to_f32(kRawKeys);
    const auto rope_cos = to_f32(kRopeCos);
    const auto rope_sin = to_f32(kRopeSin);
    const auto q_norm_weight = to_f32(kQNormWeight);
    const auto k_norm_weight = to_f32(kKNormWeight);

    // The state: the block keys of the previous calls with room for this call's blocks, the incomplete block.
    const size_t cached_blocks = inputs.at(params[kBlockKeysCache]).get_shape()[1];
    const size_t summary_cap =
        std::max(cached_blocks, static_cast<size_t>(*std::max_element(key_length.begin(), key_length.end()) / r));
    std::vector<float> summary_cache(batch * summary_cap * d_idx, 0.0F);
    const auto cached_keys = to_f32(kBlockKeysCache);
    for (size_t b = 0; b < batch; b++) {
        std::copy_n(cached_keys.begin() + b * cached_blocks * d_idx,
                    cached_blocks * d_idx,
                    summary_cache.begin() + b * summary_cap * d_idx);
    }
    auto past_raw_k = to_f32(kPastRawKeys);
    auto past_rope = to_f32(kPastRope);

    std::vector<int32_t> sel_indices_data(batch * h_sel * L * block_topk, -1);
    std::vector<int32_t> sel_count_data(batch * h_sel * L, 0);
    const size_t group = h_iq / h_ik;
    std::vector<size_t> kv_of_head(h_iq), sel_of_head(h_iq);
    for (size_t h = 0; h < h_iq; h++) {
        kv_of_head[h] = h / group;
        sel_of_head[h] = (kv_of_head[h] * h_sel) / h_ik;
    }

    sel_indices_data.assign(batch * h_sel * static_cast<size_t>(length) * block_topk, -1);
    sel_count_data.assign(batch * h_sel * static_cast<size_t>(length), 0);

    for (size_t b = 0; b < batch; b++) {
        const int64_t key_len = key_length[b];
        const int64_t past_n = std::max<int64_t>(0, key_len - length);
        const int64_t current_n = key_len - past_n;
        const int64_t num_padding_n = length - current_n;
        const int64_t c_n = past_n % r;
        const int64_t num_cached_blocks_n = past_n / r;

        std::vector<float> q_normed(h_iq * d_idx);
        std::vector<float> rope_cos_buf(rotary_dim);
        std::vector<float> rope_sin_buf(rotary_dim);
        std::vector<float> block_ref_cos_sin(2 * rotary_dim, 0.0F);
        std::vector<float> pooled(d_idx);
        std::vector<float> summary_buf(d_idx);

        std::vector<float> pending_rows(static_cast<size_t>(r) * h_ik * d_idx, 0.0F);
        auto pending_ptr = [&](int64_t pos_in_block, size_t h) {
            return pending_rows.data() + (static_cast<size_t>(pos_in_block) * h_ik + h) * d_idx;
        };
        auto past_raw_k_ptr = [&](size_t h, int64_t p) {
            return past_raw_k.data() + ((b * h_ik + h) * static_cast<size_t>(r - 1) + static_cast<size_t>(p)) * d_idx;
        };
        auto summary_ptr = [&](size_t h, int64_t block) {
            return summary_cache.data() + ((b * h_ik + h) * summary_cap + static_cast<size_t>(block)) * d_idx;
        };

        for (int64_t p = 0; p < c_n; p++) {
            for (size_t h = 0; h < h_ik; h++) {
                std::copy_n(past_raw_k_ptr(h, p), d_idx, pending_ptr(p, h));
            }
        }
        if (c_n > 0) {
            std::copy_n(past_rope.data() + b * 2 * rotary_dim, 2 * rotary_dim, block_ref_cos_sin.data());
        }

        int64_t running_block_idx = num_cached_blocks_n;

        for (int64_t l = num_padding_n; l < length; l++) {
            const int64_t v = l - num_padding_n;
            const int64_t pending_pos = c_n + v;
            const int64_t pos_in_block = pending_pos % r;
            const size_t lu = static_cast<size_t>(l);

            std::copy_n(&rope_cos[(b * L + lu) * rotary_dim], rotary_dim, rope_cos_buf.data());
            std::copy_n(&rope_sin[(b * L + lu) * rotary_dim], rotary_dim, rope_sin_buf.data());
            for (size_t h = 0; h < h_iq; h++) {
                const float* src = &q_idx[((b * L + lu) * h_iq + h) * d_idx];
                query_row_ref(src,
                              &q_normed[h * d_idx],
                              q_norm_weight.data(),
                              d_idx,
                              rotary_dim,
                              eps,
                              rope_cos_buf.data(),
                              rope_sin_buf.data(),
                              query_norm,
                              query_rope);
            }

            if (pos_in_block == 0) {
                std::copy(rope_cos_buf.begin(), rope_cos_buf.end(), block_ref_cos_sin.begin());
                std::copy(rope_sin_buf.begin(),
                          rope_sin_buf.end(),
                          block_ref_cos_sin.begin() + static_cast<ptrdiff_t>(rotary_dim));
            }
            for (size_t h = 0; h < h_ik; h++) {
                std::copy_n(&k_idx[((b * L + lu) * h_ik + h) * d_idx], d_idx, pending_ptr(pos_in_block, h));
            }

            if (pos_in_block + 1 == r) {
                for (size_t h = 0; h < h_ik; h++) {
                    std::fill(pooled.begin(), pooled.end(), 0.0F);
                    for (int64_t p = 0; p < r; p++) {
                        const float* src = pending_ptr(p, h);
                        for (size_t d = 0; d < d_idx; d++) {
                            pooled[d] += src[d];
                        }
                    }
                    const float inv_r = 1.0F / static_cast<float>(r);
                    for (float& val : pooled) {
                        val *= inv_r;
                    }
                    query_row_ref(pooled.data(),
                                  summary_buf.data(),
                                  k_norm_weight.data(),
                                  d_idx,
                                  rotary_dim,
                                  eps,
                                  block_ref_cos_sin.data(),
                                  block_ref_cos_sin.data() + rotary_dim,
                                  true,
                                  true);
                    std::copy(summary_buf.begin(), summary_buf.end(), summary_ptr(h, running_block_idx));
                }
                running_block_idx += 1;
            }

            const int64_t n_complete = running_block_idx;
            for (size_t sh = 0; sh < h_sel; sh++) {
                size_t k = 0;
                std::vector<int32_t> chosen;
                if (n_complete > 0) {
                    std::vector<std::pair<float, int32_t>> scores(static_cast<size_t>(n_complete));
                    for (int64_t bidx = 0; bidx < n_complete; bidx++) {
                        float score = 0.0F;
                        for (size_t h = 0; h < h_iq; h++) {
                            if (sel_of_head[h] != sh) {
                                continue;
                            }
                            const float* summary = summary_ptr(kv_of_head[h], bidx);
                            float dot = 0.0F;
                            for (size_t d = 0; d < d_idx; d++) {
                                dot += q_normed[h * d_idx + d] * summary[d];
                            }
                            score += std::max(0.0F, dot);
                        }
                        scores[static_cast<size_t>(bidx)] = {score * scale, static_cast<int32_t>(bidx)};
                    }
                    k = std::min(block_topk, static_cast<size_t>(n_complete));
                    std::partial_sort(scores.begin(),
                                      scores.begin() + static_cast<ptrdiff_t>(k),
                                      scores.end(),
                                      [](const auto& a, const auto& bb) {
                                          if (a.first != bb.first) {
                                              return a.first > bb.first;
                                          }
                                          return a.second < bb.second;
                                      });
                    chosen.resize(k);
                    for (size_t i = 0; i < k; i++) {
                        chosen[i] = scores[i].second;
                    }
                    std::sort(chosen.begin(), chosen.end());
                }
                auto* out_row = &sel_indices_data[((b * h_sel + sh) * L + lu) * block_topk];
                for (size_t i = 0; i < k; i++) {
                    out_row[i] = chosen[i];
                }
                sel_count_data[(b * h_sel + sh) * L + lu] = static_cast<int32_t>(k);
            }
        }

        const int64_t rest = (c_n + current_n) - (running_block_idx - num_cached_blocks_n) * r;
        for (int64_t p = 0; p < r - 1; p++) {
            for (size_t h = 0; h < h_ik; h++) {
                const float* src = p < rest ? pending_ptr(p, h) : nullptr;
                if (src != nullptr) {
                    std::copy_n(src, d_idx, past_raw_k_ptr(h, p));
                } else {
                    std::fill_n(past_raw_k_ptr(h, p), d_idx, 0.0F);
                }
            }
        }
        if (rest == 0) {
            std::fill(block_ref_cos_sin.begin(), block_ref_cos_sin.end(), 0.0F);
        }
        std::copy(block_ref_cos_sin.begin(), block_ref_cos_sin.end(), past_rope.data() + b * 2 * rotary_dim);
    }

    ov::Tensor sel_indices(ov::element::i32, ov::Shape{batch, h_sel, L, block_topk});
    ov::Tensor sel_count(ov::element::i32, ov::Shape{batch, h_sel, L});
    std::copy(sel_indices_data.begin(), sel_indices_data.end(), sel_indices.data<int32_t>());
    std::copy(sel_count_data.begin(), sel_count_data.end(), sel_count.data<int32_t>());

    // The output holds the blocks of the longest row: max(key_length) / key_block_size.
    const auto num_blocks = static_cast<size_t>(*std::max_element(key_length.begin(), key_length.end()) / r);
    ov::Tensor block_keys(data_type, ov::Shape{batch, num_blocks, d_idx});
    std::vector<float> block_keys_data(batch * num_blocks * d_idx);
    for (size_t b = 0; b < batch; b++) {
        std::copy_n(summary_cache.begin() + b * summary_cap * d_idx,
                    num_blocks * d_idx,
                    block_keys_data.begin() + b * num_blocks * d_idx);
    }
    write_f32_to_tensor(block_keys, block_keys_data);
    ov::Tensor past_raw_keys(data_type, inputs.at(params[kPastRawKeys]).get_shape());
    ov::Tensor past_rope_tensor(data_type, inputs.at(params[kPastRope]).get_shape());
    write_f32_to_tensor(past_raw_keys, past_raw_k);
    write_f32_to_tensor(past_rope_tensor, past_rope);
    return {sel_indices, sel_count, block_keys, past_raw_keys, past_rope_tensor};
}

}  // namespace

std::string QSAIndexerSubgraphTest::getTestCaseName(const testing::TestParamInfo<QSAIndexerSubgraphParams>& obj) {
    const auto& [h_iq,
                 batch,
                 indexer_head_dim,
                 rotary_dim,
                 key_block_size,
                 block_topk,
                 new_tokens,
                 key_lengths,
                 query_fusion_type,
                 scenario,
                 data_precision,
                 device] = obj.param;
    std::ostringstream result;
    result << "Hiq=" << h_iq << "_Batch=" << batch << "_Didx=" << indexer_head_dim << "_RotaryDim=" << rotary_dim
           << "_R=" << key_block_size << "_BlockTopk=" << block_topk << "_NewTokens=";
    for (size_t i = 0; i < new_tokens.second.size(); i++) {
        result << (i > 0 ? "-" : "") << new_tokens.second[i][0];
    }
    result << "_KeyLengths=";
    for (size_t i = 0; i < key_lengths.size(); i++) {
        result << (i > 0 ? "-" : "") << vec_to_string(key_lengths[i]);
    }
    result << "_QueryFusionType=" << static_cast<int>(query_fusion_type) << "_Scenario=" << scenario_name(scenario)
           << "_DataPrecision=" << data_precision << "_Target=" << device;
    return result.str();
}

void QSAIndexerSubgraphTest::SetUp() {
    const auto& [h_iq,
                 batch,
                 indexer_head_dim,
                 rotary_dim,
                 key_block_size,
                 block_topk,
                 new_tokens,
                 key_lengths,
                 query_fusion_type,
                 scenario,
                 data_precision,
                 device] = GetParam();
    // Without hardware support the plugin runs a bf16 / f16 model in f32 and converts its parameters.
    if (!ov::intel_cpu::hasHardwareSupport(data_precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << data_precision << " data on this system.";
    }
    targetDevice = device;
    m_data_precision = data_precision;
    m_h_iq = h_iq;
    m_batch = batch;
    m_indexer_head_dim = indexer_head_dim;
    m_rotary_dim = rotary_dim;
    m_key_block_size = key_block_size;
    m_block_topk = block_topk;
    m_scenario = scenario;
    m_key_lengths = key_lengths;
    m_query_norm = query_fusion_type == QSAIndexerQueryFusionType::kNormAndRope ||
                   query_fusion_type == QSAIndexerQueryFusionType::kNormOnly;
    m_query_rope = query_fusion_type == QSAIndexerQueryFusionType::kNormAndRope ||
                   query_fusion_type == QSAIndexerQueryFusionType::kRopeOnly;
    m_mismatch_share =
        query_fusion_type != QSAIndexerQueryFusionType::kNormAndRope && data_precision != ov::element::f32 ? 0.1 : 0.0;
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
    }
    if (data_precision != ov::element::f32) {
        configuration.insert(ov::hint::inference_precision(data_precision));
    }

    if (scenario == QSAIndexerScenario::kStateBatchMismatch) {
        // 2 threads: with 1 thread and 1 stream the plugin drops a node's exception (CPUStreamsExecutor::execute()).
        configuration[ov::inference_num_threads.name()] = 2;
    }

    OPENVINO_ASSERT(key_block_size > 1 && rotary_dim <= indexer_head_dim && rotary_dim % 2 == 0);
    OPENVINO_ASSERT(key_lengths.size() == new_tokens.second.size(), "key_lengths of every call");
    m_calls.clear();
    // Blocks of the previous call: the cache input of a call.
    std::vector<size_t> past_blocks;
    size_t blocks = 0;
    for (size_t call = 0; call < key_lengths.size(); call++) {
        OPENVINO_ASSERT(key_lengths[call].size() == batch, "a key length for every row");
        const auto length = static_cast<int32_t>(new_tokens.second[call][0]);
        m_calls.push_back(length);
        past_blocks.push_back(blocks);
        blocks =
            static_cast<size_t>(*std::max_element(key_lengths[call].begin(), key_lengths[call].end())) / key_block_size;
    }

    const auto dim = [](size_t value) {
        return static_cast<ov::Dimension::value_type>(value);
    };
    // Batch and tokens are dynamic, as in a real model: a static batch turns the cos / sin Unsqueeze into a Reshape
    // that FuseQSAIndexerQueryRoPE does not match.
    const auto batch_dim = ov::Dimension::dynamic();
    const auto tokens = ov::Dimension::dynamic();
    const ov::PartialShape query_shape{batch_dim, tokens, dim(h_iq), dim(indexer_head_dim)};
    const ov::PartialShape keys_shape{batch_dim, tokens, dim(indexer_head_dim)};
    const ov::PartialShape rope_shape{batch_dim, tokens, dim(rotary_dim)};
    const ov::PartialShape blocks_shape{batch_dim, tokens, dim(indexer_head_dim)};
    const std::vector<ov::PartialShape> partial_shapes{
        query_shape,
        keys_shape,
        rope_shape,
        rope_shape,
        ov::PartialShape{batch_dim},
        ov::PartialShape{dim(indexer_head_dim)},
        ov::PartialShape{dim(indexer_head_dim)},
        blocks_shape,
        ov::PartialShape{batch_dim, dim(key_block_size - 1), dim(indexer_head_dim)},
        ov::PartialShape{batch_dim, dim(2 * rotary_dim)}};

    // One target shape per call: the token inputs follow the tokens of the call, the cache the blocks of the previous
    // call, the others do not change.
    std::vector<InputShape> input_shapes;
    for (size_t input = 0; input < kInputCount; input++) {
        input_shapes.emplace_back(partial_shapes[input], std::vector<ov::Shape>{});
    }
    for (size_t call = 0; call < m_calls.size(); call++) {
        const auto length = static_cast<size_t>(m_calls[call]);
        const std::vector<ov::Shape> call_shapes{{batch, length, h_iq, indexer_head_dim},
                                                 {batch, length, indexer_head_dim},
                                                 {batch, length, rotary_dim},
                                                 {batch, length, rotary_dim},
                                                 {batch},
                                                 {indexer_head_dim},
                                                 {indexer_head_dim},
                                                 {batch, past_blocks[call], indexer_head_dim},
                                                 {batch, key_block_size - 1, indexer_head_dim},
                                                 {batch, 2 * rotary_dim}};
        for (size_t input = 0; input < kInputCount; input++) {
            input_shapes[input].second.push_back(call_shapes[input]);
        }
    }
    init_input_shapes(input_shapes);

    ov::ParameterVector params;
    for (size_t input = 0; input < kInputCount; input++) {
        const auto type = input == kKeyLength ? ov::element::i32 : data_precision;
        params.push_back(std::make_shared<ov::op::v0::Parameter>(type, partial_shapes[input]));
    }

    // The query as exported: the RMSNorm and the rotate-half RoPE of the [B, L, H, D] activations, then the head-first
    // layout of SparseAttentionIndexer.
    const auto head_transpose = [](const ov::Output<ov::Node>& data) {
        return std::make_shared<ov::op::v1::Transpose>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3}));
    };
    ov::Output<ov::Node> query = params[kQuery];
    if (m_query_norm) {
        query = std::make_shared<ov::op::internal::RMS>(query, params[kQNormWeight], kEpsilon, data_precision, -1);
    }
    if (m_query_rope) {
        ov::op::internal::RoPE::Config rope_config;
        rope_config.rotary_ndims = rotary_dim;
        rope_config.input_trans0213 = true;
        const auto head_axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        query = std::make_shared<ov::op::internal::RoPE>(
            ov::OutputVector{query,
                             std::make_shared<ov::op::v0::Unsqueeze>(params[kRopeCos], head_axis),
                             std::make_shared<ov::op::v0::Unsqueeze>(params[kRopeSin], head_axis)},
            rope_config);
    } else {
        query = head_transpose(query);
    }

    const auto prepare = std::make_shared<ov::op::v17::QSAPrepare>(params[kRawKeys],
                                                                   params[kBlockKeysCache],
                                                                   params[kKeyLength],
                                                                   params[kRopeCos],
                                                                   params[kRopeSin],
                                                                   params[kKNormWeight],
                                                                   params[kPastRawKeys],
                                                                   params[kPastRope],
                                                                   static_cast<int64_t>(key_block_size),
                                                                   kEpsilon);
    const auto indexer =
        std::make_shared<ov::op::v17::SparseAttentionIndexer>(query,
                                                              prepare->output(0),
                                                              params[kKeyLength],
                                                              static_cast<int64_t>(key_block_size),
                                                              static_cast<int64_t>(block_topk * key_block_size),
                                                              /*num_selection_heads=*/1);

    function = std::make_shared<ov::Model>(ov::ResultVector{std::make_shared<ov::op::v0::Result>(indexer->output(0)),
                                                            std::make_shared<ov::op::v0::Result>(indexer->output(1)),
                                                            std::make_shared<ov::op::v0::Result>(prepare->output(0)),
                                                            std::make_shared<ov::op::v0::Result>(prepare->output(1)),
                                                            std::make_shared<ov::op::v0::Result>(prepare->output(2))},
                                           params,
                                           "QSAIndexerSubgraph");
}

void QSAIndexerSubgraphTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    OPENVINO_ASSERT(targetInputStaticShapes.size() == kInputCount);
    const size_t L = targetInputStaticShapes[kQuery][1];
    const size_t B = m_batch;
    const size_t D = m_indexer_head_dim;
    m_call = m_next_call++;
    OPENVINO_ASSERT(m_call < m_calls.size() && static_cast<size_t>(m_calls[m_call]) == L);
    const auto& key_length = m_key_lengths[m_call];

    const auto make_data = [this](const ov::Shape& shape, uint32_t seed) {
        return ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                       shape,
                                                       ov::test::utils::InputGenerateData(-1, 2, 1000, seed));
    };
    const auto call_seed = static_cast<uint32_t>(10 + m_call * 4);
    std::vector<ov::Tensor> tensors(kInputCount);
    tensors[kQuery] = make_data(targetInputStaticShapes[kQuery], call_seed + 1);
    tensors[kRawKeys] = make_data(targetInputStaticShapes[kRawKeys], call_seed + 2);

    if (m_scenario == QSAIndexerScenario::kNaNScores) {
        // one NaN in the keys of the tokens of a few blocks
        auto k_values = read_tensor_to_f32(tensors[kRawKeys]);
        for (size_t b = 0; b < B; b++) {
            for (const size_t token : {6UL, 9UL, 34UL}) {
                if (token < L) {
                    k_values[(b * L + token) * D + 1] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        write_f32_to_tensor(tensors[kRawKeys], k_values);
    }

    // rope_cos/rope_sin can't be arbitrary random data: they must be genuine (cos, sin) pairs, pre-duplicated across
    // both halves of rotary_dim (the rotate-half convention), so they are synthesized directly.
    std::vector<float> rope_cos_data(B * L * m_rotary_dim);
    std::vector<float> rope_sin_data(B * L * m_rotary_dim);
    const size_t half = m_rotary_dim / 2;
    for (size_t b = 0; b < B; b++) {
        for (size_t l = 0; l < L; l++) {
            const auto abs_pos =
                static_cast<int64_t>(key_length[b]) - static_cast<int64_t>(L) + static_cast<int64_t>(l);
            for (size_t j = 0; j < m_rotary_dim; j++) {
                const float theta = static_cast<float>(abs_pos) * 0.1F + static_cast<float>(j % half) * 0.05F;
                rope_cos_data[(b * L + l) * m_rotary_dim + j] = std::cos(theta);
                rope_sin_data[(b * L + l) * m_rotary_dim + j] = std::sin(theta);
            }
        }
    }
    tensors[kRopeCos] = ov::Tensor(m_data_precision, targetInputStaticShapes[kRopeCos]);
    tensors[kRopeSin] = ov::Tensor(m_data_precision, targetInputStaticShapes[kRopeSin]);
    write_f32_to_tensor(tensors[kRopeCos], rope_cos_data);
    write_f32_to_tensor(tensors[kRopeSin], rope_sin_data);

    tensors[kKeyLength] = ov::Tensor(ov::element::i32, targetInputStaticShapes[kKeyLength]);
    std::copy(key_length.begin(), key_length.end(), tensors[kKeyLength].data<int32_t>());

    // values in [0.5, 1.5), the gain of the norms
    tensors[kQNormWeight] =
        ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                targetInputStaticShapes[kQNormWeight],
                                                ov::test::utils::InputGenerateData(0.5, 1, 1000, 6));
    tensors[kKNormWeight] =
        ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                targetInputStaticShapes[kKNormWeight],
                                                ov::test::utils::InputGenerateData(0.5, 1, 1000, 7));
    if (m_scenario == QSAIndexerScenario::kOverflowScores || m_scenario == QSAIndexerScenario::kTinyScores) {
        // Degenerate scores through the gain of the query RMSNorm: it scales the query, so the scores.
        const float gain = m_scenario == QSAIndexerScenario::kOverflowScores ? 3e38F : 1e-37F;
        write_f32_to_tensor(tensors[kQNormWeight], std::vector<float>(D, gain));
    }

    // The state of a call is the output of the previous one (the Assign ops of a model feed it back).
    const std::vector<size_t> state_inputs{kBlockKeysCache, kPastRawKeys, kPastRope};
    const std::vector<size_t> state_outputs{kBlockKeysOut, kPastRawKeysOut, kPastRopeOut};
    m_state_inputs.clear();
    m_state_snapshot.clear();
    for (size_t i = 0; i < state_inputs.size(); i++) {
        const auto& shape = targetInputStaticShapes[state_inputs[i]];
        ov::Tensor state;
        if (m_call == 0) {
            state = make_zero_tensor(m_data_precision, shape);
        } else {
            const auto previous = inferRequest.get_tensor(compiledModel.output(state_outputs[i]));
            OPENVINO_ASSERT(previous.get_shape() == shape, "state of the previous call: ", previous.get_shape());
            state = ov::Tensor(previous.get_element_type(), previous.get_shape());
            previous.copy_to(state);
        }
        if (m_scenario == QSAIndexerScenario::kStateBatchMismatch && state_inputs[i] == kPastRope) {
            auto shape = state.get_shape();
            shape[0]++;
            state = make_zero_tensor(m_data_precision, shape);
        }
        tensors[state_inputs[i]] = state;
        m_state_inputs.push_back(state);
        ov::Tensor snapshot(state.get_element_type(), state.get_shape());
        state.copy_to(snapshot);
        m_state_snapshot.push_back(snapshot);
    }

    const auto& params = function->get_parameters();
    for (size_t i = 0; i < kInputCount; i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> QSAIndexerSubgraphTest::calculate_refs() {
    return calculate_qsa_indexer_refs(host_inputs, function, m_block_topk, m_query_norm, m_query_rope);
}

void QSAIndexerSubgraphTest::compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold, m_mismatch_share);
    }
}

bool QSAIndexerSubgraphTest::is_degenerate_scores_scenario() const {
    return m_scenario != QSAIndexerScenario::kReference;
}

void QSAIndexerSubgraphTest::check_selection_invariants(const std::vector<ov::Tensor>& outs) const {
    const auto L = static_cast<size_t>(m_calls[m_call]);
    const auto r = static_cast<int64_t>(m_key_block_size);
    const auto* indices = outs[kSelectedIndices].data<const int32_t>();
    const auto* counts = outs[kSelectedCount].data<const int32_t>();
    for (size_t b = 0; b < m_batch; b++) {
        for (size_t l = 0; l < L; l++) {
            const int64_t abs_pos = m_key_lengths[m_call][b] - static_cast<int64_t>(L) + static_cast<int64_t>(l);
            const int64_t visible = (abs_pos + 1) / r;  // the complete blocks up to the token
            const size_t row = b * L + l;
            const auto count = static_cast<size_t>(counts[row]);
            const std::string where = "row " + std::to_string(b) + ", token " + std::to_string(l);
            ASSERT_EQ(count, std::min(m_block_topk, static_cast<size_t>(std::max<int64_t>(visible, 0)))) << where;
            const int32_t* chosen = indices + row * m_block_topk;
            for (size_t i = 0; i < count; i++) {
                ASSERT_GE(chosen[i], 0) << where;
                ASSERT_LT(chosen[i], visible) << where;
                if (i > 0) {
                    ASSERT_LT(chosen[i - 1], chosen[i]) << "blocks not distinct and ascending, " << where;
                }
            }
        }
    }
}

void QSAIndexerSubgraphTest::validate() {
    if (is_degenerate_scores_scenario()) {
        // no reference: the selection invariants only
        check_selection_invariants(get_plugin_outputs());
    } else {
        SubgraphBaseTest::validate();
    }
    // The op must not update its state inputs in place.
    for (size_t i = 0; i < m_state_inputs.size(); i++) {
        if (m_state_snapshot[i].get_byte_size() == 0) {
            continue;
        }
        EXPECT_EQ(
            0,
            std::memcmp(m_state_inputs[i].data(), m_state_snapshot[i].data(), m_state_snapshot[i].get_byte_size()))
            << "state input " << i << " was modified in place (call " << m_call << ")";
    }
}

void QSAIndexerSubgraphTest::check_fusion() const {
    const auto types = runtime_layer_types(compiledModel);
    const auto count = [&](const std::string& type) {
        return std::count(types.begin(), types.end(), type);
    };
    EXPECT_EQ(count("QSAIndexer"), 1) << "QSAPrepare + SparseAttentionIndexer are fused into one QSAIndexer";
    // The query norm and rope in front of the indexer are absorbed by it.
    EXPECT_EQ(count("RMS"), 0) << "the query RMSNorm is moved into the indexer";
    EXPECT_EQ(count("RoPE"), 0) << "the query RoPE is moved into the indexer";
}

TEST_P(QSAIndexerSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED();
    if (expects_state_batch_rejection()) {
        compile_model();
        generate_inputs(targetStaticShapes.at(0));
        EXPECT_ANY_THROW(infer());
        return;
    }
    run();
    check_fusion();
}

}  // namespace ov::test
