// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "paged_qsa_indexer.hpp"

#include <algorithm>
#include <array>
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
#include "openvino/op/paged_qsa_indexer.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/runtime/properties.hpp"
#include "utils/precision_support.h"

namespace {

using ov::test::query_row_ref;
using ov::test::rmsnorm_partial_rope;

// Small configuration exercised by this test: up to 2 independent sequences that get the new tokens of every call (see
// PagedQSAIndexerLayerTestParams), each with its own past_len (kPastLens[s]) and its own run of physical cache blocks
// (enough for its tokens and a spare one), addressed through a non-identity BLOCK_INDICES permutation (see
// generate_inputs) so a kernel bug that walked physical blocks in logical order instead of resolving through
// BLOCK_INDICES would be caught. An odd number of tokens is not a multiple of compress_ratio, so such an instantiation
// naturally exercises the causal-tail-marker path on its final tokens and the empty-selection path on its first token.
// The per-token math is that of the dense QSAIndexer test (classes/qsa_indexer.cpp), whose oracle this test reuses: the
// paged and dense kernels share the per-token / per-block math and differ only in cache addressing.
constexpr size_t kDidx = 16;
constexpr size_t kRotaryDim = 8;
constexpr size_t kR = 4;  // the key block size of the index blocks (a cache page of 32 tokens holds 8 of them)
constexpr size_t kBlockTopk = 2;
constexpr std::array<int32_t, 2> kPastLens = {0, 4};
constexpr float kEps = 1e-6F;
constexpr float kScale = 0.25F;

// Paged-cache layout: the page size (a test parameter) must be a multiple of compress_ratio. The physical blocks of a
// sequence (m_seq_blocks) hold its past and new tokens plus a spare page, so the last, incomplete block always has
// room.

// Independent (host-vector based) re-implementation of the paged-qsa-indexer.rst per-sequence
// spec, used purely as a test oracle -- NOT a copy of the CPU node's PlainTensor-based paged
// addressing. Per-sequence math is bit-for-bit identical to the dense QSAIndexer oracle
// (classes/qsa_indexer.cpp's run_qsa_indexer_reference): the paged kernel's logical_pos ==
// past_len + local token index, exactly dense's abs_pos, and sequences never interact.
struct PagedQSAIndexerOracleOutput {
    std::vector<int64_t> sel_indices;
    std::vector<int64_t> sel_count;
};

PagedQSAIndexerOracleOutput run_paged_qsa_indexer_reference(size_t new_tokens,
                                                            size_t h_iq,
                                                            size_t h_ik,
                                                            size_t h_sel,
                                                            const std::vector<float>& q_idx,
                                                            const std::vector<float>& k_idx,
                                                            const std::vector<float>& rope_cos,
                                                            const std::vector<float>& rope_sin,
                                                            const std::vector<float>& q_norm_weight,
                                                            const std::vector<float>& k_norm_weight,
                                                            int64_t past_len,
                                                            bool query_norm,
                                                            bool query_rope) {
    const size_t group = h_iq / h_ik;
    std::vector<size_t> kv_of_head(h_iq), sel_of_head(h_iq);
    for (size_t h = 0; h < h_iq; h++) {
        kv_of_head[h] = h / group;
        sel_of_head[h] = (kv_of_head[h] * h_sel) / h_ik;
    }

    const size_t s_cap = new_tokens + static_cast<size_t>(past_len);
    const size_t summary_len = s_cap / kR;
    std::vector<float> raw_k_cache(h_ik * s_cap * kDidx, 0.0F);
    std::vector<float> summary_cache(h_ik * summary_len * kDidx, 0.0F);

    PagedQSAIndexerOracleOutput out;
    out.sel_indices.assign(h_sel * new_tokens * kBlockTopk, -1);
    out.sel_count.assign(h_sel * new_tokens, 0);

    for (size_t l = 0; l < new_tokens; l++) {
        const int64_t abs_pos = past_len + static_cast<int64_t>(l);

        std::vector<float> q_normed(h_iq * kDidx);
        const float* cos_row = &rope_cos[l * kRotaryDim];
        const float* sin_row = &rope_sin[l * kRotaryDim];
        for (size_t h = 0; h < h_iq; h++) {
            const float* src = &q_idx[(l * h_iq + h) * kDidx];
            query_row_ref(src,
                          &q_normed[h * kDidx],
                          q_norm_weight.data(),
                          kDidx,
                          kRotaryDim,
                          kEps,
                          cos_row,
                          sin_row,
                          query_norm,
                          query_rope);
        }

        for (size_t h = 0; h < h_ik; h++) {
            const float* src = &k_idx[(l * h_ik + h) * kDidx];
            std::copy_n(src, kDidx, &raw_k_cache[(h * s_cap + static_cast<size_t>(abs_pos)) * kDidx]);
        }

        const int64_t n_complete = (abs_pos + 1) / static_cast<int64_t>(kR);
        if ((abs_pos + 1) % static_cast<int64_t>(kR) == 0) {
            const int64_t block_idx = n_complete - 1;
            const int64_t block_start = block_idx * static_cast<int64_t>(kR);
            const auto block_row = static_cast<size_t>(block_start - past_len);
            const float* block_cos = &rope_cos[block_row * kRotaryDim];
            const float* block_sin = &rope_sin[block_row * kRotaryDim];
            for (size_t h = 0; h < h_ik; h++) {
                std::vector<float> pooled(kDidx, 0.0F);
                for (int64_t p = block_start; p < block_start + static_cast<int64_t>(kR); p++) {
                    const float* raw = &raw_k_cache[(h * s_cap + static_cast<size_t>(p)) * kDidx];
                    for (size_t d = 0; d < kDidx; d++) {
                        pooled[d] += raw[d];
                    }
                }
                for (float& v : pooled) {
                    v /= static_cast<float>(kR);
                }
                std::vector<float> summary(kDidx);
                rmsnorm_partial_rope(pooled.data(),
                                     summary.data(),
                                     k_norm_weight.data(),
                                     kDidx,
                                     kRotaryDim,
                                     kEps,
                                     block_cos,
                                     block_sin);
                std::copy(summary.begin(),
                          summary.end(),
                          &summary_cache[(h * summary_len + static_cast<size_t>(block_idx)) * kDidx]);
            }
        }

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
                        const float* summary =
                            &summary_cache[(kv_of_head[h] * summary_len + static_cast<size_t>(bidx)) * kDidx];
                        float dot = 0.0F;
                        for (size_t d = 0; d < kDidx; d++) {
                            dot += q_normed[h * kDidx + d] * summary[d];
                        }
                        score += std::max(0.0F, dot);
                    }
                    scores[static_cast<size_t>(bidx)] = {score * kScale, static_cast<int32_t>(bidx)};
                }
                k = std::min(kBlockTopk, static_cast<size_t>(n_complete));
                std::partial_sort(scores.begin(),
                                  scores.begin() + static_cast<ptrdiff_t>(k),
                                  scores.end(),
                                  [](const auto& a, const auto& b) {
                                      if (a.first != b.first) {
                                          return a.first > b.first;
                                      }
                                      return a.second < b.second;
                                  });
                chosen.resize(k);
                for (size_t i = 0; i < k; i++) {
                    chosen[i] = scores[i].second;
                }
                std::sort(chosen.begin(), chosen.end());
            }

            auto* out_row = &out.sel_indices[(sh * new_tokens + l) * kBlockTopk];
            for (size_t i = 0; i < k; i++) {
                out_row[i] = chosen[i];
            }
            out.sel_count[sh * new_tokens + l] = static_cast<int64_t>(k);
        }
    }
    return out;
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

std::string PagedQSAIndexerLayerCPUTest::getTestCaseName(
    const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj) {
    const auto& [device,
                 data_precision,
                 h_iq,
                 h_ik,
                 h_sel,
                 num_sequences,
                 new_tokens,
                 page_size,
                 query_fusion_type,
                 scenario] = obj.param;
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
    return "Target=" + device + "_DataPrecision=" + data_precision.get_type_name() + "_Hiq=" + std::to_string(h_iq) +
           "_Hik=" + std::to_string(h_ik) + "_Hsel=" + std::to_string(h_sel) +
           "_NumSeq=" + std::to_string(num_sequences) + "_NewTokens=" + calls_to_string(new_tokens) +
           "_PageSize=" + std::to_string(page_size) +
           "_QueryFusionType=" + std::to_string(static_cast<int>(query_fusion_type)) + "_Scenario=" + scenario_name;
}

void PagedQSAIndexerLayerCPUTest::SetUp() {
    const auto& [device,
                 data_precision,
                 h_iq,
                 h_ik,
                 h_sel,
                 num_sequences,
                 new_tokens,
                 page_size,
                 query_fusion_type,
                 scenario] = GetParam();
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
    m_calls.clear();
    size_t total_new_tokens = 0;
    for (const auto& shape : new_tokens.second) {
        m_calls.push_back(shape[0]);
        total_new_tokens += shape[0];
    }
    m_page_size = page_size;
    m_slots_per_page = page_size / kR;
    // the tokens of the longest sequence after all the calls in whole pages, and one spare page
    m_seq_blocks = (total_new_tokens + static_cast<size_t>(kPastLens.back()) + m_page_size - 1) / m_page_size + 1;
    rel_threshold = 1e-3F;
    abs_threshold = 1e-4F;
    if (data_precision == ov::element::bf16) {
        rel_threshold = 5e-2F;
        abs_threshold = 5e-2F;
        // The oracle keeps the block keys in f32, the kernel in bf16 (8 mantissa bits): a few near-tied scores pick
        // another block.
        m_mismatch_share = 0.05;
        configuration.insert(ov::hint::inference_precision(ov::element::bf16));
    } else if (data_precision == ov::element::f16) {
        rel_threshold = 5e-3F;
        abs_threshold = 5e-3F;
        m_mismatch_share = 0.02;
        configuration.insert(ov::hint::inference_precision(ov::element::f16));
    }

    const size_t n_phys = m_num_sequences * m_seq_blocks;

    // The tokens of a call are dynamic: every call of the test parameter is a target shape of them.
    const ov::Dimension tokens = ov::Dimension::dynamic();
    auto p_q_idx = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_h_iq * kDidx)});
    auto p_k_idx = std::make_shared<ov::op::v0::Parameter>(
        data_precision,
        ov::PartialShape{tokens, static_cast<ov::Dimension::value_type>(m_h_ik * kDidx)});
    auto p_rope_cos = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{tokens, kRotaryDim});
    auto p_rope_sin = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::PartialShape{tokens, kRotaryDim});
    auto p_q_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{kDidx});
    auto p_k_norm_weight = std::make_shared<ov::op::v0::Parameter>(data_precision, ov::Shape{kDidx});
    auto p_past_lens = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences});
    auto p_subsequence_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    auto p_block_indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{n_phys});
    auto p_block_indices_begins =
        std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{m_num_sequences + 1});
    // The caches are typed by ConvertPagedAttnInputs (the inference precision) and their number of pages and page size
    // are dynamic, as in a real pipeline (the page size is known when the model runs).
    const auto cache_dim = static_cast<ov::Dimension::value_type>(-1);
    auto p_raw_k_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim, static_cast<ov::Dimension::value_type>(m_h_ik), cache_dim, kDidx});
    auto p_indexer_rope_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim, cache_dim, static_cast<ov::Dimension::value_type>(2 * kRotaryDim)});
    auto p_summary_cache = std::make_shared<ov::op::v0::Parameter>(
        ov::element::dynamic,
        ov::PartialShape{cache_dim, static_cast<ov::Dimension::value_type>(m_h_ik), cache_dim, kDidx});

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
        {p_raw_k_cache.get(), ov::Shape{n_phys, m_h_ik, m_page_size, kDidx}},
        {p_indexer_rope_cache.get(), ov::Shape{n_phys, m_slots_per_page, 2 * kRotaryDim}},
        {p_summary_cache.get(), ov::Shape{n_phys, m_h_ik, m_slots_per_page, kDidx}}};
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
    config.key_block_size = kR;
    config.block_topk = kBlockTopk;
    config.num_selection_heads = m_h_sel;
    config.indexer_rotary_dim = kRotaryDim;
    config.indexer_head_dim = kDidx;
    config.epsilon = kEps;
    config.scale = kScale;
    config.query_norm = m_query_norm;
    config.query_rope = m_query_rope;

    ov::OutputVector args(params_in.begin(), params_in.end());
    auto paged_qsa_indexer = std::make_shared<ov::op::internal::PagedQSAIndexer>(args, config);

    function = std::make_shared<ov::Model>(
        ov::ResultVector{std::make_shared<ov::op::v0::Result>(paged_qsa_indexer->output(0)),
                         std::make_shared<ov::op::v0::Result>(paged_qsa_indexer->output(1))},
        params_in);
}

void PagedQSAIndexerLayerCPUTest::generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) {
    inputs.clear();
    host_inputs.clear();

    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    // The calls continue the sequences: each one starts where the previous one ended.
    const size_t L = targetInputStaticShapes[InputIdx::INDEX_QUERY][0] / m_num_sequences;
    if (m_next_call == 0) {
        m_past.assign(kPastLens.begin(), kPastLens.begin() + static_cast<ptrdiff_t>(m_num_sequences));
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
                                                       ov::Shape{m_num_sequences * L, h * kDidx},
                                                       ov::test::utils::InputGenerateData(-1, 2, 10000, seed));
    };
    // The caches are bound to the precision the compiled model settled on for them (the inference precision), which
    // can differ from the data precision.
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
                                                       ov::Shape{kDidx},
                                                       ov::test::utils::InputGenerateData(0.5, 1, 1000, seed));
    };

    const size_t n_phys = m_num_sequences * m_seq_blocks;

    const auto& params = function->get_parameters();
    std::vector<ov::Tensor> tensors(params.size());
    const auto call_seed = static_cast<uint32_t>(m_call) * 100U;
    tensors[InputIdx::INDEX_QUERY] = make_seq_data(m_h_iq, 1 + call_seed);
    tensors[InputIdx::RAW_INDEX_KEYS] = make_seq_data(m_h_ik, 2 + call_seed);
    if (m_scenario == PagedQSAIndexerScenario::kNaNScores) {
        // one NaN in the keys of a few tokens of every sequence: the summaries of their blocks (blocks 1, 2 and 8) are
        // NaN, and so are their scores
        auto k_values = read_tensor_to_f32(tensors[InputIdx::RAW_INDEX_KEYS]);
        for (size_t s = 0; s < m_num_sequences; s++) {
            for (const size_t token : {6UL, 9UL, 34UL}) {
                if (token < L) {
                    k_values[((s * L + token) * m_h_ik + (token % m_h_ik)) * kDidx + 1] =
                        std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        write_f32_to_tensor(tensors[InputIdx::RAW_INDEX_KEYS], k_values);
    }
    {
        // theta is a function of each token's own absolute position (kPastLens[s] + local index),
        // matching the per-token rope_cos/rope_sin ABI -- no shared position-indexed table.
        // Rows are indexer_rotary_dim wide, one coefficient per rotary channel. The two halves are
        // deliberately NOT duplicates, so a kernel assuming cos[j] == cos[j + rotary_dim/2] fails.
        ov::Tensor t_cos(m_data_precision, ov::Shape{m_num_sequences * L, kRotaryDim});
        ov::Tensor t_sin(m_data_precision, ov::Shape{m_num_sequences * L, kRotaryDim});
        std::vector<float> cos_data(m_num_sequences * L * kRotaryDim);
        std::vector<float> sin_data(m_num_sequences * L * kRotaryDim);
        for (size_t s = 0; s < m_num_sequences; s++) {
            for (size_t l = 0; l < L; l++) {
                const size_t abs_pos = sequence_past_len(s) + l;
                const size_t row = s * L + l;
                for (size_t j = 0; j < kRotaryDim; j++) {
                    const float theta = static_cast<float>(abs_pos) * 0.1F + static_cast<float>(j) * 0.05F;
                    cos_data[row * kRotaryDim + j] = std::cos(theta);
                    sin_data[row * kRotaryDim + j] = std::sin(theta);
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
        write_f32_to_tensor(tensors[InputIdx::Q_NORM_WEIGHT], std::vector<float>(kDidx, gain));
    }
    tensors[InputIdx::K_NORM_WEIGHT] = make_norm_weight(7);
    {
        // Non-identity permutation (simple reversal) across all physical blocks shared by every
        // sequence, so a kernel bug that walked physical blocks in logical/sequential order
        // instead of resolving through block_indices would produce numerically wrong results.
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
        m_caches = {make_zeros(InputIdx::INDEX_KEY_CACHE, ov::Shape{n_phys, m_h_ik, m_page_size, kDidx}),
                    make_zeros(InputIdx::INDEX_BLOCK_ROPE_CACHE, ov::Shape{n_phys, m_slots_per_page, 2 * kRotaryDim}),
                    make_zeros(InputIdx::BLOCK_INDEX_KEYS_CACHE, ov::Shape{n_phys, m_h_ik, m_slots_per_page, kDidx})};
    }
    tensors[InputIdx::INDEX_KEY_CACHE] = m_caches[0];
    tensors[InputIdx::INDEX_BLOCK_ROPE_CACHE] = m_caches[1];
    tensors[InputIdx::BLOCK_INDEX_KEYS_CACHE] = m_caches[2];

    for (size_t i = 0; i < params.size(); i++) {
        host_inputs[params[i]] = tensors[i];
        inputs[params[i]] = tensors[i];
    }
}

std::vector<ov::Tensor> PagedQSAIndexerLayerCPUTest::calculate_refs() {
    const auto& params = function->get_parameters();
    const auto to_f32 = [&](const std::shared_ptr<ov::Node>& p) {
        return read_tensor_to_f32(host_inputs.at(p));
    };

    using InputIdx = ov::op::internal::PagedQSAIndexer::InputIdx;
    auto q_idx = to_f32(params[InputIdx::INDEX_QUERY]);
    auto k_idx = to_f32(params[InputIdx::RAW_INDEX_KEYS]);
    auto rope_cos = to_f32(params[InputIdx::ROPE_COS]);
    auto rope_sin = to_f32(params[InputIdx::ROPE_SIN]);
    auto q_norm_weight = to_f32(params[InputIdx::Q_NORM_WEIGHT]);
    auto k_norm_weight = to_f32(params[InputIdx::K_NORM_WEIGHT]);
    const size_t L = m_cur_l;
    const size_t tokens_total = m_num_sequences * L;
    ov::Tensor sel_indices(ov::element::i32, ov::Shape{tokens_total, m_h_sel, kBlockTopk});
    ov::Tensor sel_count(ov::element::i32, ov::Shape{tokens_total, m_h_sel});
    auto* sel_indices_data = sel_indices.data<int32_t>();
    auto* sel_count_data = sel_count.data<int32_t>();

    // The result of a call depends on the tokens of all the previous calls (the caches), so the oracle replays every
    // token of the sequence from the first call and its rows of the current call are taken.
    if (m_call == 0) {
        m_history.assign(m_num_sequences, {});
    }
    const size_t q_idx_stride = L * m_h_iq * kDidx;
    const size_t k_idx_stride = L * m_h_ik * kDidx;
    const size_t rope_stride = L * kRotaryDim;
    const auto append = [](std::vector<float>& history, const std::vector<float>& data, size_t stride, size_t seq) {
        history.insert(history.end(),
                       data.begin() + static_cast<ptrdiff_t>(seq * stride),
                       data.begin() + static_cast<ptrdiff_t>((seq + 1) * stride));
    };
    for (size_t s = 0; s < m_num_sequences; s++) {
        auto& history = m_history[s];
        append(history.q_idx, q_idx, q_idx_stride, s);
        append(history.k_idx, k_idx, k_idx_stride, s);
        append(history.rope_cos, rope_cos, rope_stride, s);
        append(history.rope_sin, rope_sin, rope_stride, s);
        const size_t replayed_tokens = history.rope_cos.size() / kRotaryDim;

        auto oracle = run_paged_qsa_indexer_reference(replayed_tokens,
                                                      m_h_iq,
                                                      m_h_ik,
                                                      m_h_sel,
                                                      history.q_idx,
                                                      history.k_idx,
                                                      history.rope_cos,
                                                      history.rope_sin,
                                                      q_norm_weight,
                                                      k_norm_weight,
                                                      kPastLens[s],
                                                      m_query_norm,
                                                      m_query_rope);

        // Oracle output is [h_sel, replayed_tokens, kBlockTopk]/[h_sel, replayed_tokens] (per-sequence, sh-major); the
        // op's real outputs are token-major ([tokens_total, h_sel, kBlockTopk]/[tokens_total, h_sel]), so transpose
        // while scattering the rows of the current call into the global, token-offset tensor.
        const size_t first_row = replayed_tokens - L;
        for (size_t l = 0; l < L; l++) {
            const size_t token = s * L + l;
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                for (size_t i = 0; i < kBlockTopk; i++) {
                    sel_indices_data[(token * m_h_sel + sh) * kBlockTopk + i] = static_cast<int32_t>(
                        oracle.sel_indices[(sh * replayed_tokens + first_row + l) * kBlockTopk + i]);
                }
                sel_count_data[token * m_h_sel + sh] =
                    static_cast<int32_t>(oracle.sel_count[sh * replayed_tokens + first_row + l]);
            }
        }
    }
    return {sel_indices, sel_count};
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

// The op appends to its caches in place: every call must leave its new raw keys and the summaries of the blocks it
// completes in the tensors the user bound, not in a temporary copy (which would make the caches stale from the second
// call on). Checks the rows of the tokens / blocks of the call in the three caches.
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
                const size_t row = ((page * m_h_ik + h) * m_page_size + token % m_page_size) * kDidx;
                EXPECT_TRUE(row_is_nonzero(raw_k_cache, row, kDidx))
                    << "raw key of token " << token << " of sequence " << s << " (call " << m_call << ")";
            }
        }
        for (size_t block = past / kR; block < (past + m_cur_l) / kR; block++) {
            const size_t page = physical_page(s, block * kR);
            const size_t slot = block % m_slots_per_page;
            EXPECT_TRUE(row_is_nonzero(rope_cache, (page * m_slots_per_page + slot) * 2 * kRotaryDim, 2 * kRotaryDim))
                << "rope row of block " << block << " of sequence " << s << " (call " << m_call << ")";
            for (size_t h = 0; h < m_h_ik; h++) {
                const size_t row = ((page * m_h_ik + h) * m_slots_per_page + slot) * kDidx;
                EXPECT_TRUE(row_is_nonzero(summary_cache, row, kDidx))
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
            const auto visible = static_cast<int64_t>((sequence_past_len(s) + l + 1) / kR);
            for (size_t sh = 0; sh < m_h_sel; sh++) {
                const size_t row = (s * m_cur_l + l) * m_h_sel + sh;
                const std::string where = "sequence " + std::to_string(s) + ", token " + std::to_string(l) +
                                          ", sel head " + std::to_string(sh);
                ASSERT_EQ(counts[row], std::min<int64_t>(static_cast<int64_t>(kBlockTopk), visible)) << where;
                const int32_t* chosen = indices + row * kBlockTopk;
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
}

TEST_P(PagedQSAIndexerLayerCPUTest, CompareWithRefs) {
    run();
}

}  // namespace test
}  // namespace ov