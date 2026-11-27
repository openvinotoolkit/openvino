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
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "utils/qsa_indexer_reference.hpp"

namespace ov::test {

// kReference compares with the host oracle; the others have degenerate scores and check only the selection invariants.
enum class QSAIndexerScenario {
    kReference,
    // a NaN in the keys of a few tokens
    kNaNScores,
    // query norm weights that overflow the scores to +-Inf and NaN
    kOverflowScores,
    // query norm weights that make the scores of a row differ by less than 256 / FLT_MAX
    kTinyScores,
    // the incomplete-rope state has one row more than the batch: the node must throw instead of writing out of bounds
    kStateBatchMismatch
};

// Public QSAPrepare + SparseAttentionIndexer with the query RMSNorm / RoPE in front, fused by the plugin into
// QSAIndexer. new_tokens: one target shape {n} per call; key_lengths: valid key tokens per row after each call.
using QSAIndexerSubgraphParams = std::tuple<size_t,                             // H_iq (index-query head count)
                                            size_t,                             // batch
                                            size_t,                             // indexer_head_dim
                                            size_t,                             // rotary_dim
                                            size_t,                             // key_block_size (compress ratio)
                                            size_t,                             // block_topk
                                            InputShape,                         // new_tokens
                                            std::vector<std::vector<int32_t>>,  // key_lengths
                                            QSAIndexerQueryFusionType,          // query_fusion_type
                                            QSAIndexerScenario,                 // scenario
                                            ov::element::Type,                  // data_precision
                                            std::string>;                       // device

class QSAIndexerSubgraphTest : public testing::WithParamInterface<QSAIndexerSubgraphParams>,
                               virtual public SubgraphBaseTest {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<QSAIndexerSubgraphParams>& obj);

protected:
    void SetUp() override;
    void generate_inputs(const std::vector<ov::Shape>& targetInputStaticShapes) override;
    std::vector<ov::Tensor> calculate_refs() override;
    void compare(const std::vector<ov::Tensor>& expected, const std::vector<ov::Tensor>& actual) override;
    void validate() override;
    // The plugin has to run the model as the fused QSAIndexer, without the query RMS / RoPE in front of it.
    void check_fusion() const;
    [[nodiscard]] bool expects_state_batch_rejection() const {
        return m_scenario == QSAIndexerScenario::kStateBatchMismatch;
    }

private:
    [[nodiscard]] bool is_degenerate_scores_scenario() const;
    void check_selection_invariants(const std::vector<ov::Tensor>& outs) const;

    std::map<std::shared_ptr<ov::Node>, ov::Tensor> host_inputs;
    std::vector<ov::Tensor> m_state_inputs;    // block keys, incomplete raw keys, incomplete rope of the current call
    std::vector<ov::Tensor> m_state_snapshot;  // their bytes before the inference: the op must not write them
    std::vector<int32_t> m_calls;              // new tokens of every row in each call
    std::vector<std::vector<int32_t>> m_key_lengths;
    size_t m_next_call = 0;
    size_t m_call = 0;  // index of the current call
    ov::element::Type m_data_precision = ov::element::f32;
    size_t m_h_iq = 2;
    size_t m_batch = 1;
    size_t m_indexer_head_dim = 16;
    size_t m_rotary_dim = 8;
    size_t m_key_block_size = 4;
    size_t m_block_topk = 2;
    bool m_query_norm = true;
    bool m_query_rope = true;
    QSAIndexerScenario m_scenario = QSAIndexerScenario::kReference;
    // Share of output elements that may differ: 16-bit types can flip the score order of near-equal blocks.
    double m_mismatch_share = 0.0;
};

}  // namespace ov::test
