// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "common_test_utils/ov_tensor_utils.hpp"
#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/precision_support.h"

namespace {

using ov::test::query_row_ref;
using ov::test::read_tensor_to_f32;
using ov::test::write_f32_to_tensor;

constexpr float kEps = 1e-6F;
constexpr float kScale = 0.25F;

// Host-vector reference of one call, from the paged-qsa-indexer spec. The caches of the previous calls are read from
// the inputs (snapshots taken before the op writes them), the new tokens from the activations.
std::vector<ov::Tensor> calculate_paged_qsa_indexer_refs(const std::map<std::shared_ptr<ov::Node>, ov::Tensor>& inputs,
                                                         const std::shared_ptr<ov::Model>& function,
                                                         size_t h_sel,
                                                         size_t block_topk,
                                                         bool query_norm,
                                                         bool query_rope) {
    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    const auto& params = function->get_parameters();
    const auto shape_of = [&](size_t idx) -> const ov::Shape& {
        return inputs.at(params[idx]).get_shape();
    };
    const auto to_f32 = [&](size_t idx) {
        return read_tensor_to_f32(inputs.at(params[idx]));
    };
    const auto to_i32 = [&](size_t idx) {
        const auto& t = inputs.at(params[idx]);
        return std::vector<int32_t>(t.data<const int32_t>(), t.data<const int32_t>() + t.get_size());
    };
    const size_t d_idx = shape_of(InputIdx::Q_NORM_WEIGHT)[0];
    const size_t rotary_dim = shape_of(InputIdx::ROPE_COS)[1];
    const size_t tokens = shape_of(InputIdx::INDEX_QUERY)[0];
    const size_t h_iq = shape_of(InputIdx::INDEX_QUERY)[1] / d_idx;
    const size_t h_ik = shape_of(InputIdx::INDEX_KEY_CACHE)[1];
    const size_t page_size = shape_of(InputIdx::INDEX_KEY_CACHE)[2];
    const size_t slots = shape_of(InputIdx::INDEX_BLOCK_ROPE_CACHE)[1];
    const size_t r = page_size / slots;
    const size_t num_sequences = shape_of(InputIdx::PAST_LENS)[0];

    const auto q_idx = to_f32(InputIdx::INDEX_QUERY);
    const auto k_idx = to_f32(InputIdx::RAW_INDEX_KEYS);
    const auto rope_cos = to_f32(InputIdx::ROPE_COS);
    const auto rope_sin = to_f32(InputIdx::ROPE_SIN);
    const auto q_norm_weight = to_f32(InputIdx::Q_NORM_WEIGHT);
    const auto k_norm_weight = to_f32(InputIdx::K_NORM_WEIGHT);
    const auto past_lens = to_i32(InputIdx::PAST_LENS);
    const auto subsequence_begins = to_i32(InputIdx::SUBSEQUENCE_BEGINS);
    const auto block_indices = to_i32(InputIdx::BLOCK_INDICES);
    const auto block_indices_begins = to_i32(InputIdx::BLOCK_INDICES_BEGINS);
    const auto raw_k_cache = to_f32(InputIdx::INDEX_KEY_CACHE);
    const auto rope_cache = to_f32(InputIdx::INDEX_BLOCK_ROPE_CACHE);
    const auto summary_cache = to_f32(InputIdx::BLOCK_INDEX_KEYS_CACHE);

    std::vector<int32_t> sel_indices_data(tokens * h_sel * block_topk, -1);
    std::vector<int32_t> sel_count_data(tokens * h_sel, 0);

    const size_t group = h_iq / h_ik;
    std::vector<size_t> kv_of_head(h_iq), sel_of_head(h_iq);
    for (size_t h = 0; h < h_iq; h++) {
        kv_of_head[h] = h / group;
        sel_of_head[h] = (kv_of_head[h] * h_sel) / h_ik;
    }

    for (size_t s = 0; s < num_sequences; s++) {
        const auto begin = static_cast<size_t>(subsequence_begins[s]);
        const auto new_tokens = static_cast<size_t>(subsequence_begins[s + 1]) - begin;
        const auto past = static_cast<size_t>(past_lens[s]);
        const size_t total_blocks = (past + new_tokens) / r;
        // The physical page of a token position of the sequence.
        const auto page_of = [&](size_t position) {
            return static_cast<size_t>(
                block_indices[static_cast<size_t>(block_indices_begins[s]) + position / page_size]);
        };
        const auto raw_key = [&](size_t position, size_t h) {
            if (position >= past) {
                return &k_idx[((begin + position - past) * h_ik + h) * d_idx];
            }
            return &raw_k_cache[((page_of(position) * h_ik + h) * page_size + position % page_size) * d_idx];
        };

        // The block summaries: the cached ones of the past, the ones this call completes.
        std::vector<float> summaries(h_ik * total_blocks * d_idx, 0.0F);
        for (size_t block = 0; block < total_blocks; block++) {
            const size_t start = block * r;
            const size_t slot = (start % page_size) / r;
            // The RoPE row of the block's first token: this call's, or the interleaved one stored in the rope cache.
            std::vector<float> cached_cos(rotary_dim), cached_sin(rotary_dim);
            const float* cos_row = nullptr;
            const float* sin_row = nullptr;
            if (start >= past) {
                cos_row = &rope_cos[(begin + start - past) * rotary_dim];
                sin_row = &rope_sin[(begin + start - past) * rotary_dim];
            } else {
                const float* entry = &rope_cache[(page_of(start) * slots + slot) * 2 * rotary_dim];
                for (size_t j = 0; j < rotary_dim; j++) {
                    cached_cos[j] = entry[2 * j];
                    cached_sin[j] = entry[2 * j + 1];
                }
                cos_row = cached_cos.data();
                sin_row = cached_sin.data();
            }
            for (size_t h = 0; h < h_ik; h++) {
                float* summary = &summaries[(h * total_blocks + block) * d_idx];
                if (start + r <= past) {
                    std::copy_n(&summary_cache[((page_of(start) * h_ik + h) * slots + slot) * d_idx], d_idx, summary);
                    continue;
                }
                std::vector<float> pooled(d_idx, 0.0F);
                for (size_t position = start; position < start + r; position++) {
                    const float* raw = raw_key(position, h);
                    for (size_t d = 0; d < d_idx; d++) {
                        pooled[d] += raw[d];
                    }
                }
                for (float& value : pooled) {
                    value /= static_cast<float>(r);
                }
                query_row_ref(pooled.data(),
                              summary,
                              k_norm_weight.data(),
                              d_idx,
                              rotary_dim,
                              kEps,
                              cos_row,
                              sin_row,
                              true,
                              true);
            }
        }

        for (size_t l = 0; l < new_tokens; l++) {
            const size_t token = begin + l;
            const size_t n_complete = (past + l + 1) / r;
            std::vector<float> q_normed(h_iq * d_idx);
            for (size_t h = 0; h < h_iq; h++) {
                query_row_ref(&q_idx[(token * h_iq + h) * d_idx],
                              &q_normed[h * d_idx],
                              q_norm_weight.data(),
                              d_idx,
                              rotary_dim,
                              kEps,
                              &rope_cos[token * rotary_dim],
                              &rope_sin[token * rotary_dim],
                              query_norm,
                              query_rope);
            }
            for (size_t sh = 0; sh < h_sel; sh++) {
                size_t k = 0;
                std::vector<int32_t> chosen;
                if (n_complete > 0) {
                    std::vector<std::pair<float, int32_t>> scores(n_complete);
                    for (size_t block = 0; block < n_complete; block++) {
                        float score = 0.0F;
                        for (size_t h = 0; h < h_iq; h++) {
                            if (sel_of_head[h] != sh) {
                                continue;
                            }
                            const float* summary = &summaries[(kv_of_head[h] * total_blocks + block) * d_idx];
                            float dot = 0.0F;
                            for (size_t d = 0; d < d_idx; d++) {
                                dot += q_normed[h * d_idx + d] * summary[d];
                            }
                            score += std::max(0.0F, dot);
                        }
                        scores[block] = {score * kScale, static_cast<int32_t>(block)};
                    }
                    k = std::min(block_topk, n_complete);
                    std::partial_sort(scores.begin(),
                                      scores.begin() + static_cast<ptrdiff_t>(k),
                                      scores.end(),
                                      [](const auto& a, const auto& b) {
                                          return a.first != b.first ? a.first > b.first : a.second < b.second;
                                      });
                    chosen.resize(k);
                    for (size_t i = 0; i < k; i++) {
                        chosen[i] = scores[i].second;
                    }
                    std::sort(chosen.begin(), chosen.end());
                }
                std::copy(chosen.begin(), chosen.end(), &sel_indices_data[(token * h_sel + sh) * block_topk]);
                sel_count_data[token * h_sel + sh] = static_cast<int32_t>(k);
            }
        }
    }

    ov::Tensor sel_indices(ov::element::i32, ov::Shape{tokens, h_sel, block_topk});
    ov::Tensor sel_count(ov::element::i32, ov::Shape{tokens, h_sel});
    std::copy(sel_indices_data.begin(), sel_indices_data.end(), sel_indices.data<int32_t>());
    std::copy(sel_count_data.begin(), sel_count_data.end(), sel_count.data<int32_t>());
    return {sel_indices, sel_count};
}

}  // namespace

namespace ov {
namespace test {

// The new tokens per sequence of every call of an InputShape test parameter, e.g. "7-1-1" (a prefill, then decode
// steps).
static std::string calls_to_string(const ov::test::InputShape& calls) {
    std::string out;
    for (const auto& shape : calls.second) {
        out += (out.empty() ? "" : "-") + std::to_string(shape[0]);
    }
    return out;
}

// The past lengths of the sequences of a test parameter, e.g. "0-4".
static std::string vec_to_string(const std::vector<int32_t>& values) {
    std::string out;
    for (const auto value : values) {
        out += (out.empty() ? "" : "-") + std::to_string(value);
    }
    return out;
}

std::string PagedQSAIndexerLayerCPUTest::getTestCaseName(
    const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj) {
    const auto& [h_iq,
                 h_ik,
                 h_sel,
                 indexer_head_dim,
                 rotary_dim,
                 key_block_size,
                 block_topk,
                 page_size,
                 num_sequences,
                 past_lens,
                 new_tokens,
                 query_fusion_type,
                 scenario,
                 data_precision,
                 inference_precision,
                 device] = obj.param;
    const char* scenario_name = "Reference";
    switch (scenario) {
    case PagedQSAIndexerScenario::kReference:
        break;
    case PagedQSAIndexerScenario::kNaNScores:
        scenario_name = "NaNScores";
        break;
    case PagedQSAIndexerScenario::kOverflowScores:
        scenario_name = "OverflowScores";
        break;
    case PagedQSAIndexerScenario::kTinyScores:
        scenario_name = "TinyScores";
        break;
    }
    return "Hiq=" + std::to_string(h_iq) + "_Hik=" + std::to_string(h_ik) + "_Hsel=" + std::to_string(h_sel) +
           "_Didx=" + std::to_string(indexer_head_dim) + "_RotaryDim=" + std::to_string(rotary_dim) +
           "_R=" + std::to_string(key_block_size) + "_BlockTopk=" + std::to_string(block_topk) +
           "_PageSize=" + std::to_string(page_size) + "_NumSeq=" + std::to_string(num_sequences) +
           "_PastLens=" + vec_to_string(past_lens) + "_NewTokens=" + calls_to_string(new_tokens) +
           "_QueryFusionType=" + std::to_string(static_cast<int>(query_fusion_type)) + "_Scenario=" + scenario_name +
           "_DataPrecision=" + data_precision.get_type_name() +
           "_InferencePrecision=" + inference_precision.get_type_name() + "_Target=" + device;
}

void PagedQSAIndexerLayerCPUTest::SetUp() {
    const auto& [h_iq,
                 h_ik,
                 h_sel,
                 indexer_head_dim,
                 rotary_dim,
                 key_block_size,
                 block_topk,
                 page_size,
                 num_sequences,
                 past_lens,
                 new_tokens,
                 query_fusion_type,
                 scenario,
                 data_precision,
                 inference_precision,
                 device] = GetParam();
    // Without hardware support the plugin runs a bf16 / f16 model in f32 and converts its parameters.
    if (!ov::intel_cpu::hasHardwareSupport(data_precision)) {
        GTEST_SKIP() << "CPU precision policy does not preserve " << data_precision << " data on this system.";
    }
    m_query_norm = query_fusion_type == QSAIndexerQueryFusionType::kNormAndRope ||
                   query_fusion_type == QSAIndexerQueryFusionType::kNormOnly;
    m_query_rope = query_fusion_type == QSAIndexerQueryFusionType::kNormAndRope ||
                   query_fusion_type == QSAIndexerQueryFusionType::kRopeOnly;
    targetDevice = device;
    m_scenario = scenario;
    m_data_precision = data_precision;
    m_h_iq = h_iq;
    m_h_ik = h_ik;
    m_h_sel = h_sel;
    m_num_sequences = num_sequences;
    m_indexer_head_dim = indexer_head_dim;
    m_rotary_dim = rotary_dim;
    m_key_block_size = key_block_size;
    m_block_topk = block_topk;
    m_past_lens = past_lens;
    OPENVINO_ASSERT(m_past_lens.size() >= m_num_sequences, "a past length is needed for every sequence");
    OPENVINO_ASSERT(m_key_block_size > 0 && page_size % m_key_block_size == 0,
                    "the page size must be a multiple of the key block size");
    m_calls.clear();
    size_t total_new_tokens = 0;
    for (const auto& shape : new_tokens.second) {
        m_calls.push_back(shape[0]);
        total_new_tokens += shape[0];
    }
    m_page_size = page_size;
    m_slots_per_page = page_size / m_key_block_size;
    // the tokens of the longest sequence after all the calls in whole pages, and one spare page
    m_seq_blocks = (total_new_tokens + static_cast<size_t>(*std::max_element(m_past_lens.begin(), m_past_lens.end())) +
                    m_page_size - 1) /
                       m_page_size +
                   1;
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
        // The oracle keeps the block keys in f32, the kernel in bf16 (8 mantissa bits): a few near-tied scores pick
        // another block.
        m_mismatch_share = 0.05;
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
        m_mismatch_share = 0.02;
    }
    // A hint of the model data precision unless the test parameter asks for another one.
    m_convert_outputs = inference_precision != ov::element::dynamic && inference_precision != data_precision;
    m_inference_precision = inference_precision;
    if (m_convert_outputs) {
        if (!ov::intel_cpu::hasHardwareSupport(inference_precision)) {
            GTEST_SKIP() << "CPU precision policy does not preserve " << inference_precision << " on this system.";
        }
        configuration.insert(ov::hint::inference_precision(inference_precision));
    } else if (data_precision != ov::element::f32) {
        configuration.insert(ov::hint::inference_precision(data_precision));
    }

    const size_t n_phys = m_num_sequences * m_seq_blocks;

    // The tokens of a call are dynamic: every call of the test parameter is a target shape of them.
    const ov::Dimension tokens = ov::Dimension::dynamic();
    auto p_q_idx = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_h_iq * m_indexer_head_dim)});
    auto p_k_idx = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_h_ik * m_indexer_head_dim)});
    auto p_rope_cos = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_rotary_dim)});
    auto p_rope_sin = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_rotary_dim)});
    auto p_q_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{m_indexer_head_dim});
    auto p_k_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{m_indexer_head_dim});
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences});
    auto p_subsequence_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{n_phys});
    auto p_block_indices_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    // The caches are typed by ConvertPagedAttnInputs (the indexer data type) and their number of pages and page size
    // are dynamic, as in a real pipeline (the page size is known when the model runs).
    const auto cache_dim = static_cast<ov::Dimension::value_type>(-1);
    auto p_raw_k_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim,
                         static_cast<ov::Dimension::value_type>(m_h_ik),
                         cache_dim,
                         static_cast<ov::Dimension::value_type>(m_indexer_head_dim)});
    auto p_indexer_rope_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim, cache_dim, static_cast<ov::Dimension::value_type>(2 * m_rotary_dim)});
    auto p_summary_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim,
                         static_cast<ov::Dimension::value_type>(m_h_ik),
                         cache_dim,
                         static_cast<ov::Dimension::value_type>(m_indexer_head_dim)});

    const ov::ParameterVector params_in{p_q_idx,
                                        p_k_idx,
                                        p_rope_cos,
                                        p_rope_sin,
                                        p_q_norm_weight,
                                        p_k_norm_weight,
                                        p_past_lens,
                                        p_subsequence_begins,
                                        p_block_indices,
                                        p_block_indices_begins,
                                        p_raw_k_cache,
                                        p_indexer_rope_cache,
                                        p_summary_cache};
    // One target shape per call: the token inputs follow the tokens of the call, the others do not change.
    std::vector<InputShape> input_shapes;
    // The caches are dynamic in the model, their target shape does not change.
    const std::map<const ov::Node*, ov::Shape> cache_shapes{
        {p_raw_k_cache.get(), ov::Shape{n_phys, m_h_ik, m_page_size, m_indexer_head_dim}},
        {p_indexer_rope_cache.get(), ov::Shape{n_phys, m_slots_per_page, 2 * m_rotary_dim}},
        {p_summary_cache.get(), ov::Shape{n_phys, m_h_ik, m_slots_per_page, m_indexer_head_dim}}};
    for (const auto& p : params_in) {
        const auto cache_shape = cache_shapes.find(p.get());
        const bool token_input = cache_shape == cache_shapes.end() && p->get_partial_shape().is_dynamic();
        std::vector<ov::Shape> target_shapes;
        for (const auto new_tokens_per_seq : m_calls) {
            ov::Shape shape = token_input                         ? ov::Shape{m_num_sequences * new_tokens_per_seq,
                                                      static_cast<size_t>(p->get_partial_shape()[1].get_length())}
                              : cache_shape != cache_shapes.end() ? cache_shape->second
                                                                  : p->get_shape();
            target_shapes.push_back(shape);
        }
        input_shapes.emplace_back(p->get_partial_shape(), target_shapes);
    }
    init_input_shapes(input_shapes);

    ov::op::internal::PagedQSAIndexer::Config config;
    config.key_block_size = m_key_block_size;
    config.block_topk = m_block_topk;
    config.num_selection_heads = m_h_sel;
    config.indexer_rotary_dim = m_rotary_dim;
    config.indexer_head_dim = m_indexer_head_dim;
    config.epsilon = kEps;
    config.scale = kScale;
    config.query_norm = m_query_norm;
    config.query_rope = m_query_rope;

    ov::OutputVector args(params_in.begin(), params_in.end());
    auto paged_qsa_indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(args, config);

    ov::ResultVector results;
    for (const auto& output : paged_qsa_indexer->outputs()) {
        // A Convert behind the outputs keeps the op out of the tail of the graph, see inference_precision.
        results.push_back(std::make_shared<ov::op::v0::Result>(
            m_convert_outputs ? std::make_shared<ov::op::v0::Convert>(output, inference_precision)->output(0)
                              : output));
    }
    function = std::make_shared<ov::Model>(results, params_in);
}

void PagedQSAIndexerLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    // The calls continue the sequences: each one starts where the previous one ended.
    const size_t L = targetInputStaticShapes[InputIdx::INDEX_QUERY][0] / m_num_sequences;
    if (m_next_call == 0) {
        m_past.assign(m_past_lens.begin(), m_past_lens.begin() + static_cast<ptrdiff_t>(m_num_sequences));
    } else {
        for (auto& past : m_past) {
            past += m_cur_l;
        }
    }
    m_cur_l = L;
    m_call = m_next_call++;

    // One fill covers all sequences, so they get different values: a kernel bug that mixed up the sequence index (or
    // its physical cache blocks) would otherwise go undetected.
    const auto make_seq_data = [this, L](size_t h, uint32_t seed) {
        return ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                       ov::Shape{m_num_sequences * L, h * m_indexer_head_dim},
                                                       ov::test::utils::InputGenerateData(-1, 2, 10000, seed));
    };
    // The caches are bound to the precision the compiled model gave them: the data precision, whatever the inference
    // precision hint is.
    const auto make_zeros = [this](size_t input_idx, const ov::Shape& shape) {
        ov::Tensor t(compiledModel.input(input_idx).get_element_type(), shape);
        std::memset(t.data(), 0, t.get_byte_size());
        return t;
    };
    const auto make_i32 = [](const std::vector<int32_t>& values) {
        ov::Tensor t(ov::element::i32, ov::Shape{values.size()});
        std::copy(values.begin(), values.end(), t.data<int32_t>());
        return t;
    };
    const auto make_norm_weight = [this](uint32_t seed) {
        // values in [0.5, 1.5), as the dense test does
        return ov::test::utils::create_and_fill_tensor(m_data_precision,
                                                       ov::Shape{m_indexer_head_dim},
                                                       ov::test::utils::InputGenerateData(0.5, 1, 1000, seed));
    };

    const size_t n_phys = m_num_sequences * m_seq_blocks;

    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    const auto call_seed = static_cast<uint32_t>(m_call) * 100U;
    tensors[InputIdx::INDEX_QUERY] = make_seq_data(m_h_iq, 1 + call_seed);
    tensors[InputIdx::RAW_INDEX_KEYS] = make_seq_data(m_h_ik, 2 + call_seed);
    if (m_scenario == PagedQSAIndexerScenario::kNaNScores) {
        // one NaN in the keys of a few tokens of every sequence: the summaries of their blocks (blocks 1, 2 and 8 of
        // the first sequence) are NaN, and so are their scores
        auto k_values = read_tensor_to_f32(tensors[InputIdx::RAW_INDEX_KEYS]);
        for (size_t s = 0; s < m_num_sequences; s++) {
            for (const size_t token : {6UL, 9UL, 34UL}) {
                if (token < L) {
                    k_values[((s * L + token) * m_h_ik + (token % m_h_ik)) * m_indexer_head_dim + 1] =
                        std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        write_f32_to_tensor(tensors[InputIdx::RAW_INDEX_KEYS], k_values);
    }
    {
        // theta follows each token's absolute position; the two halves of a row are deliberately not duplicates, so a
        // kernel assuming cos[j] == cos[j + rotary_dim / 2] fails.
        ov::Tensor t_cos(m_data_precision, ov::Shape{m_num_sequences * L, m_rotary_dim});
        ov::Tensor t_sin(m_data_precision, ov::Shape{m_num_sequences * L, m_rotary_dim});
        std::vector<float> cos_data(m_num_sequences * L * m_rotary_dim);
        std::vector<float> sin_data(m_num_sequences * L * m_rotary_dim);
        for (size_t s = 0; s < m_num_sequences; s++) {
            for (size_t l = 0; l < L; l++) {
                const size_t abs_pos = sequence_past_len(s) + l;
                const size_t row = s * L + l;
                for (size_t j = 0; j < m_rotary_dim; j++) {
                    const float theta = static_cast<float>(abs_pos) * 0.1F + static_cast<float>(j) * 0.05F;
                    cos_data[row * m_rotary_dim + j] = std::cos(theta);
                    sin_data[row * m_rotary_dim + j] = std::sin(theta);
                }
            }
        }
        write_f32_to_tensor(t_cos, cos_data);
        write_f32_to_tensor(t_sin, sin_data);
        tensors[InputIdx::ROPE_COS] = t_cos;
        tensors[InputIdx::ROPE_SIN] = t_sin;
    }
    tensors[InputIdx::Q_NORM_WEIGHT] = make_norm_weight(6);
    if (m_scenario == PagedQSAIndexerScenario::kOverflowScores || m_scenario == PagedQSAIndexerScenario::kTinyScores) {
        // Degenerate scores through the gain of the query RMSNorm: it scales the query, so the scores.
        const float gain = m_scenario == PagedQSAIndexerScenario::kOverflowScores ? 3e38F : 1e-37F;
        write_f32_to_tensor(tensors[InputIdx::Q_NORM_WEIGHT], std::vector<float>(m_indexer_head_dim, gain));
    }
    tensors[InputIdx::K_NORM_WEIGHT] = make_norm_weight(7);
    {
        // Reversed block table: a kernel walking physical blocks in logical order instead of through block_indices gets
        // wrong results.
        std::vector<int32_t> block_indices(n_phys);
        for (size_t i = 0; i < n_phys; i++) {
            block_indices[i] = static_cast<int32_t>(n_phys - 1 - i);
        }
        tensors[InputIdx::BLOCK_INDICES] = make_i32(block_indices);
    }
    {
        std::vector<int32_t> block_indices_begins(m_num_sequences + 1);
        for (size_t s = 0; s <= m_num_sequences; s++) {
            block_indices_begins[s] = static_cast<int32_t>(s * m_seq_blocks);
        }
        tensors[InputIdx::BLOCK_INDICES_BEGINS] = make_i32(block_indices_begins);
    }
    {
        std::vector<int32_t> subsequence_begins(m_num_sequences + 1);
        for (size_t s = 0; s <= m_num_sequences; s++) {
            subsequence_begins[s] = static_cast<int32_t>(s * L);
        }
        tensors[InputIdx::SUBSEQUENCE_BEGINS] = make_i32(subsequence_begins);
    }
    {
        std::vector<int32_t> past_lens_values(m_num_sequences);
        for (size_t s = 0; s < m_num_sequences; s++) {
            past_lens_values[s] = static_cast<int32_t>(sequence_past_len(s));
        }
        tensors[InputIdx::PAST_LENS] = make_i32(past_lens_values);
    }
    // The caches persist across the calls of a test: only the first call starts from zeros.
    if (m_call == 0) {
        m_caches = {make_zeros(InputIdx::INDEX_KEY_CACHE, ov::Shape{n_phys, m_h_ik, m_page_size, m_indexer_head_dim}),
                    make_zeros(InputIdx::INDEX_BLOCK_ROPE_CACHE, ov::Shape{n_phys, m_slots_per_page, 2 * m_rotary_dim}),
                    make_zeros(InputIdx::BLOCK_INDEX_KEYS_CACHE,
                               ov::Shape{n_phys, m_h_ik, m_slots_per_page, m_indexer_head_dim})};
    }
    tensors[InputIdx::INDEX_KEY_CACHE] = m_caches[0];
    tensors[InputIdx::INDEX_BLOCK_ROPE_CACHE] = m_caches[1];
    tensors[InputIdx::BLOCK_INDEX_KEYS_CACHE] = m_caches[2];

    for (size_t i = 0; i < params.size(); i++) {
        inputs[params[i]] = tensors[i];
        // The op writes the caches in place: the reference reads their state before the inference.
        const bool is_cache = i == InputIdx::INDEX_KEY_CACHE || i == InputIdx::INDEX_BLOCK_ROPE_CACHE ||
                              i == InputIdx::BLOCK_INDEX_KEYS_CACHE;
        if (is_cache) {
            host_inputs[params[i]] = ov::Tensor(tensors[i].get_element_type(), tensors[i].get_shape());
            tensors[i].copy_to(host_inputs[params[i]]);
        } else {
            host_inputs[params[i]] = tensors[i];
        }
    }
}

std::vector<ov::Tensor> PagedQSAIndexerLayerCPUTest::calculate_refs() {
    auto expected =
        calculate_paged_qsa_indexer_refs(host_inputs, function, m_h_sel, m_block_topk, m_query_norm, m_query_rope);
    if (m_convert_outputs) {
        // The outputs of the model are converted, so are the expected ones (block ids and counts are exact in 16 bits).
        for (auto& t : expected) {
            ov::Tensor converted(m_inference_precision, t.get_shape());
            const auto* src = t.data<const int32_t>();
            write_f32_to_tensor(converted, std::vector<float>(src, src + t.get_size()));
            t = converted;
        }
    }
    return expected;
}

void PagedQSAIndexerLayerCPUTest::compare(const std::vector<ov::Tensor>& expected,
                                          const std::vector<ov::Tensor>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); i++) {
        ov::test::utils::compare(expected[i], actual[i], abs_threshold, rel_threshold, m_mismatch_share);
    }
}

size_t PagedQSAIndexerLayerCPUTest::sequence_past_len(size_t seq) const {
    return m_past[seq];
}

// The op appends to the caches in place: check that each call leaves its raw keys and block summaries in the bound
// tensors, not in a copy.
void PagedQSAIndexerLayerCPUTest::check_caches_updated() const {
    const auto read = [](const ov::Tensor& t, size_t idx) {
        const auto type = t.get_element_type();
        if (type == ov::element::bf16) {
            return static_cast<float>(t.data<const ov::bfloat16>()[idx]);
        }
        if (type == ov::element::f16) {
            return static_cast<float>(t.data<const ov::float16>()[idx]);
        }
        return t.data<const float>()[idx];
    };
    const auto row_is_nonzero = [&](const ov::Tensor& t, size_t first, size_t count) {
        for (size_t i = 0; i < count; i++) {
            if (read(t, first + i) != 0.0F) {
                return true;
            }
        }
        return false;
    };
    const size_t n_phys = m_num_sequences * m_seq_blocks;
    const auto physical_page = [&](size_t seq, size_t token) {
        // block_indices is the reversed identity, see generate_inputs()
        return n_phys - 1 - (seq * m_seq_blocks + token / m_page_size);
    };
    const auto& raw_k_cache = m_caches[0];
    const auto& rope_cache = m_caches[1];
    const auto& summary_cache = m_caches[2];
    for (size_t s = 0; s < m_num_sequences; s++) {
        const size_t past = sequence_past_len(s);
        for (size_t token = past; token < past + m_cur_l; token++) {
            const size_t page = physical_page(s, token);
            for (size_t h = 0; h < m_h_ik; h++) {
                const size_t row = ((page * m_h_ik + h) * m_page_size + token % m_page_size) * m_indexer_head_dim;
                EXPECT_TRUE(row_is_nonzero(raw_k_cache, row, m_indexer_head_dim))
                    << "raw key of token " << token << " of sequence " << s << " (call " << m_call << ")";
            }
        }
        for (size_t block = past / m_key_block_size; block < (past + m_cur_l) / m_key_block_size; block++) {
            const size_t page = physical_page(s, block * m_key_block_size);
            const size_t slot = block % m_slots_per_page;
            EXPECT_TRUE(
                row_is_nonzero(rope_cache, (page * m_slots_per_page + slot) * 2 * m_rotary_dim, 2 * m_rotary_dim))
                << "rope row of block " << block << " of sequence " << s << " (call " << m_call << ")";
            for (size_t h = 0; h < m_h_ik; h++) {
                const size_t row = ((page * m_h_ik + h) * m_slots_per_page + slot) * m_indexer_head_dim;
                EXPECT_TRUE(row_is_nonzero(summary_cache, row, m_indexer_head_dim))
                    << "summary of block " << block << " of sequence " << s << " (call " << m_call << ")";
            }
        }
    }
}

bool PagedQSAIndexerLayerCPUTest::is_degenerate_scores_scenario() const {
    return m_scenario != PagedQSAIndexerScenario::kReference;
}

void PagedQSAIndexerLayerCPUTest::check_selection_invariants(const std::vector<ov::Tensor>& outs) const {
    // the outputs of the op: the selected block ids [tokens, H_sel, K] and their counts [tokens, H_sel]
    constexpr size_t kSelIndices = 0;
    constexpr size_t kSelCount = 1;
    const auto* indices = outs[kSelIndices].data<const int32_t>();
    const auto* counts = outs[kSelCount].data<const int32_t>();
    for (size_t s = 0; s < m_num_sequences; s++) {
        for (size_t l = 0; l < m_cur_l; l++) {
            // the complete blocks up to the token
            const auto visible = static_cast<int64_t>((sequence_past_len(s) + l + 1) / m_key_block_size);
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                const size_t row = (s * m_cur_l + l) * m_h_sel + sh;
                const std::string where = "sequence " + std::to_string(s) + ", token " + std::to_string(l) +
                                          ", sel head " + std::to_string(sh);
                ASSERT_EQ(counts[row], std::min<int64_t>(static_cast<int64_t>(m_block_topk), visible)) << where;
                const int32_t* chosen = indices + row * m_block_topk;
                for (int32_t i = 0; i < counts[row]; i++) {
                    ASSERT_GE(chosen[i], 0) << where;
                    ASSERT_LT(chosen[i], visible) << where;
                    if (i > 0) {
                        ASSERT_LT(chosen[i - 1], chosen[i]) << "blocks not distinct and ascending, " << where;
                    }
                }
            }
        }
    }
}

void PagedQSAIndexerLayerCPUTest::validate() {
    if (is_degenerate_scores_scenario()) {
        // no reference: the selection invariants only
        check_selection_invariants(get_plugin_outputs());
    } else {
        SubgraphBaseTest::validate();
    }
    check_caches_updated();
    if (m_convert_outputs) {
        // The caches are written in place, so they keep the data precision of the op whatever the hint says.
        using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
        for (const auto idx :
             {InputIdx::INDEX_KEY_CACHE, InputIdx::INDEX_BLOCK_ROPE_CACHE, InputIdx::BLOCK_INDEX_KEYS_CACHE}) {
            EXPECT_EQ(compiledModel.input(idx).get_element_type(), m_data_precision) << "cache input " << idx;
        }
    }
}

TEST_P(PagedQSAIndexerLayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov