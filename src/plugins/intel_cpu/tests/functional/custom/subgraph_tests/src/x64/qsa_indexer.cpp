// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/qsa_indexer.hpp"

#include <vector>

#include "utils/cpu_test_utils.hpp"

namespace ov::test {

namespace {

const std::vector<ov::element::Type> all_precisions{ov::element::f32, ov::element::f16, ov::element::bf16};
const std::vector<QSAIndexerQueryFusionType> all_fusion_types{QSAIndexerQueryFusionType::kNormAndRope,
                                                              QSAIndexerQueryFusionType::kNormOnly,
                                                              QSAIndexerQueryFusionType::kRopeOnly,
                                                              QSAIndexerQueryFusionType::kNone};

// The calls of a test as an InputShape: the new tokens of every row in each call. The calls continue the same rows.
InputShape calls(const std::vector<size_t>& new_tokens_per_call) {
    std::vector<ov::Shape> target_shapes;
    for (const auto n : new_tokens_per_call) {
        target_shapes.push_back(ov::Shape{n});
    }
    return InputShape{ov::PartialShape{-1}, target_shapes};
}

// A call schedule: the key block size, the batch, the new tokens of every call and the key lengths (valid key tokens of
// every row) after it.
struct Schedule {
    size_t key_block_size;
    size_t batch;
    std::vector<size_t> new_tokens;
    std::vector<std::vector<int32_t>> key_lengths;
};

QSAIndexerSubgraphParams make_case(size_t h_iq,
                                   const Schedule& schedule,
                                   QSAIndexerQueryFusionType fusion_type,
                                   QSAIndexerScenario scenario,
                                   ov::element::Type data_precision) {
    return {h_iq,
            schedule.batch,
            /*indexer_head_dim=*/16,
            /*rotary_dim=*/8,
            schedule.key_block_size,
            /*block_topk=*/2,
            calls(schedule.new_tokens),
            schedule.key_lengths,
            fusion_type,
            scenario,
            data_precision,
            ov::test::utils::DEVICE_CPU};
}

std::vector<QSAIndexerSubgraphParams> combine(const std::vector<size_t>& h_iqs,
                                              const std::vector<Schedule>& schedules,
                                              const std::vector<QSAIndexerQueryFusionType>& fusion_types,
                                              const std::vector<QSAIndexerScenario>& scenarios,
                                              const std::vector<ov::element::Type>& precisions) {
    std::vector<QSAIndexerSubgraphParams> cases;
    for (const auto h_iq : h_iqs) {
        for (const auto& schedule : schedules) {
            for (const auto fusion_type : fusion_types) {
                for (const auto scenario : scenarios) {
                    for (const auto precision : precisions) {
                        cases.push_back(make_case(h_iq, schedule, fusion_type, scenario, precision));
                    }
                }
            }
        }
    }
    return cases;
}

// Dense (no padding) single-call prefill of 7 tokens, one and two rows.
const std::vector<Schedule> dense_prefill{{2, 1, {7}, {{7}}}, {2, 2, {7}, {{7, 7}}}};

// The worked example of the QSAPrepare spec: a left-padded prefill followed by a decode call that completes an
// incomplete tail block.
const Schedule worked_example{4, 2, {10, 1}, {{10, 7}, {11, 8}}};
// A row without any real token (key_length == 0) next to a regular one, then that row starting from scratch.
const Schedule fully_padded_row{4, 2, {5, 3}, {{5, 0}, {8, 3}}};
// Several calls with L > 1 and different past lengths per row (non-empty incomplete-block tails, blocks completed
// mid-call and exactly at the end of a call).
const Schedule ragged_calls{4, 2, {7, 3, 5}, {{7, 5}, {10, 8}, {15, 13}}};
// One 80-token prefill with key_block_size 2, so the vectorized selection runs.
const Schedule long_prefill{2, 2, {80}, {{80, 80}}};

}  // namespace

INSTANTIATE_TEST_SUITE_P(
    smoke_QSAIndexer_Subgraph_DensePrefill,
    QSAIndexerSubgraphTest,
    testing::ValuesIn(
        combine({2, 4}, dense_prefill, all_fusion_types, {QSAIndexerScenario::kReference}, all_precisions)),
    QSAIndexerSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_Subgraph_Calls,
                         QSAIndexerSubgraphTest,
                         testing::ValuesIn(combine({2, 4},
                                                   {worked_example, fully_padded_row, ragged_calls},
                                                   all_fusion_types,
                                                   {QSAIndexerScenario::kReference},
                                                   all_precisions)),
                         QSAIndexerSubgraphTest::getTestCaseName);

// Degenerate scores: NaN in the keys, scores overflowing to +-Inf / NaN, and scores that differ by less than
// 256 / FLT_MAX. The gains of the last two are out of the f16 range. Only the selection invariants are checked.
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_Subgraph_DegenerateScores_NaN,
                         QSAIndexerSubgraphTest,
                         testing::ValuesIn(combine({2, 4},
                                                   {long_prefill},
                                                   {QSAIndexerQueryFusionType::kNormAndRope},
                                                   {QSAIndexerScenario::kNaNScores},
                                                   all_precisions)),
                         QSAIndexerSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_Subgraph_DegenerateScores_OverflowAndTiny,
                         QSAIndexerSubgraphTest,
                         testing::ValuesIn(combine({2, 4},
                                                   {long_prefill},
                                                   {QSAIndexerQueryFusionType::kNormAndRope},
                                                   {QSAIndexerScenario::kOverflowScores,
                                                    QSAIndexerScenario::kTinyScores},
                                                   {ov::element::f32, ov::element::bf16})),
                         QSAIndexerSubgraphTest::getTestCaseName);

// A state with another batch than the query: the node rejects it (the kernels index every tensor by the query batch).
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_Subgraph_StateBatchMismatch,
                         QSAIndexerSubgraphTest,
                         testing::ValuesIn(combine({2},
                                                   {worked_example},
                                                   {QSAIndexerQueryFusionType::kNormAndRope},
                                                   {QSAIndexerScenario::kStateBatchMismatch},
                                                   {ov::element::f32})),
                         QSAIndexerSubgraphTest::getTestCaseName);

}  // namespace ov::test
