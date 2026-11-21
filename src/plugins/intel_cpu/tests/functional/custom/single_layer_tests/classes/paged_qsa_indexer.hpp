// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "openvino/core/type/element_type.hpp"
#include "openvino/runtime/tensor.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/qsa_indexer_reference.hpp"

namespace ov {
namespace test {

// kReference compares with the host oracle; the others have degenerate scores, no reference, and check only the
// selection invariants.
enum class PagedQSAIndexerScenario {
    kReference,
    // a NaN in the keys of a few tokens
    kNaNScores,
    // query norm weights that overflow the scores to +-Inf and NaN
    kOverflowScores,
    // query norm weights that make the scores of a row differ by less than 256 / FLT_MAX
    kTinyScores
};

// new_tokens: one target shape {n} per call, n new tokens per sequence; the calls continue the same caches.
// inference_precision: dynamic follows data_precision; another one adds a Convert behind the outputs (mid-graph node).
using PagedQSAIndexerLayerTestParams = std::tuple<size_t,                // H_iq (index-query head count)
                                                  size_t,                // H_ik (index-key head count)
                                                  size_t,                // h_sel (selection-head count)
                                                  size_t,                // indexer_head_dim
                                                  size_t,                // rotary_dim
                                                  size_t,                // key_block_size (compress ratio)
                                                  size_t,                // block_topk
                                                  size_t,                // page_size
                                                  size_t,                // num_sequences
                                                  std::vector<int32_t>,  // past_lens
                                                  InputShape,            // new_tokens
                                                  QSAIndexerQueryFusionType,
                                                  PagedQSAIndexerScenario,
                                                  ov::element::Type,  // data_precision
                                                  ov::element::Type,  // inference_precision
                                                  std::string>;       // device

class PagedQSAIndexerLayerCPUTest : public testing::WithParamInterface<PagedQSAIndexerLayerTestParams>,
                                    virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<PagedQSAIndexerLayerTestParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void validate() override;

private:
    // The past length of a sequence in the current call: the calls of a test continue the sequences.
    size_t sequence_past_len(size_t seq) const;
    void check_caches_updated() const;
    [[nodiscard]] bool is_degenerate_scores_scenario() const;
    // The selection of a call without a reference: the count of visible blocks, distinct ascending visible ids.
    void check_selection_invariants(const std::vector<ov::Tensor>& outs) const;

    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    std::vector<ov::Tensor> m_caches;  // raw key, block rope and block summary caches, kept across the calls
    std::vector<size_t> m_calls;       // new tokens of every sequence in each call
    std::vector<size_t> m_past;        // past length of every sequence in the current call
    size_t m_next_call = 0;
    size_t m_call = 0;   // index of the current call
    size_t m_cur_l = 0;  // new tokens of every sequence in the current call
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_indexer_head_dim = 16;
    size_t m_rotary_dim = 8;
    size_t m_key_block_size = 4;
    size_t m_block_topk = 2;
    std::vector<int32_t> m_past_lens;
    ov::element::Type m_inference_precision = ov::element::dynamic;
    bool m_convert_outputs = false;  // see inference_precision
    size_t m_h_iq = 2;
    size_t m_h_ik = 2;
    size_t m_h_sel = 1;
    bool m_query_norm = true;
    bool m_query_rope = true;
    PagedQSAIndexerScenario m_scenario = PagedQSAIndexerScenario::kReference;
    size_t m_num_sequences = 1;
    // Share of output elements that may differ: near-tied selections flip with the rounding of a 16-bit cache, which
    // the f32 oracle does not do.
    double m_mismatch_share = 0.0;
    size_t m_page_size = 32;
    size_t m_slots_per_page = 8;  // blocks of a page: page_size / compress_ratio
    size_t m_seq_blocks = 4;      // physical cache blocks of every sequence, enough for its tokens and a spare one
};

}  // namespace test
}  // namespace ov