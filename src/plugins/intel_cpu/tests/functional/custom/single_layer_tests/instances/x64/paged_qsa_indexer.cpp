// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/paged_qsa_indexer.hpp"

#include <vector>

#include "utils/cpu_test_utils.hpp"

namespace ov::test {

namespace {

// The calls of a test as an InputShape: the new tokens per sequence of each call. The first one is the prefill, the
// following ones are decode steps (1) or chunks; the calls continue the same sequences.
InputShape calls(const std::vector<size_t>& new_tokens_per_call) {
    std::vector<ov::Shape> target_shapes;
    for (const auto n : new_tokens_per_call) {
        target_shapes.push_back(ov::Shape{n});
    }
    return InputShape{ov::PartialShape{-1}, target_shapes};
}

const auto reference = PagedQSAIndexerScenario::kReference;
const std::vector<QSAIndexerQueryFusionType> all_fusion_types{QSAIndexerQueryFusionType::kNormAndRope,
                                                              QSAIndexerQueryFusionType::kNormOnly,
                                                              QSAIndexerQueryFusionType::kRopeOnly,
                                                              QSAIndexerQueryFusionType::kNone};

// Tokens of a page of the CPU caches: always 32 (the key cache block size of PagedAttention).
constexpr size_t kPageSize = 32;

// A small index head: 16 channels, 8 of them rotated, a block of 4 tokens (a cache page of 32 tokens holds 8 of them),
// 2 selected blocks. The sequences start with a past of 0 and 4 tokens, the second one on a block boundary.
PagedQSAIndexerLayerTestParams make_case(size_t h_iq,
                                         size_t h_ik,
                                         size_t h_sel,
                                         size_t num_sequences,
                                         const InputShape& new_tokens,
                                         QSAIndexerQueryFusionType fusion_type,
                                         PagedQSAIndexerScenario scenario,
                                         ov::element::Type data_precision,
                                         ov::element::Type inference_precision = ov::element::dynamic) {
    return {h_iq,
            h_ik,
            h_sel,
            /*indexer_head_dim=*/16,
            /*rotary_dim=*/8,
            /*key_block_size=*/4,
            /*block_topk=*/2,
            kPageSize,
            num_sequences,
            /*past_lens=*/{0, 4},
            new_tokens,
            fusion_type,
            scenario,
            data_precision,
            inference_precision,
            ov::test::utils::DEVICE_CPU};
}

// Every combination of the lists, in the order of the parameters of the cases.
std::vector<PagedQSAIndexerLayerTestParams> combine(size_t h_iq,
                                                    size_t h_ik,
                                                    size_t h_sel,
                                                    const std::vector<InputShape>& new_tokens,
                                                    const std::vector<ov::element::Type>& precisions,
                                                    const std::vector<QSAIndexerQueryFusionType>& fusion_types,
                                                    const std::vector<PagedQSAIndexerScenario>& scenarios,
                                                    const std::vector<size_t>& h_iqs = {}) {
    std::vector<PagedQSAIndexerLayerTestParams> cases;
    for (const auto iq : h_iqs.empty() ? std::vector<size_t>{h_iq} : h_iqs) {
        for (const auto precision : precisions) {
            for (const size_t num_sequences : {1, 2}) {
                for (const auto& calls_of_test : new_tokens) {
                    for (const auto fusion_type : fusion_types) {
                        for (const auto scenario : scenarios) {
                            cases.push_back(make_case(iq,
                                                      h_ik,
                                                      h_sel,
                                                      num_sequences,
                                                      calls_of_test,
                                                      fusion_type,
                                                      scenario,
                                                      precision));
                        }
                    }
                }
            }
        }
    }
    return cases;
}

}  // namespace

// H_iq == H_ik, h_sel == 1: every index-query head contributes to the single selection head.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_Hsel1,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(combine(2,
                                                   2,
                                                   1,
                                                   {calls({7})},
                                                   {ov::element::f32, ov::element::f16, ov::element::bf16},
                                                   all_fusion_types,
                                                   {reference})),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// H_iq = 2 * H_ik, h_sel == H_ik: exercises the grouped per-selection-head scoring path.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_HselEqualsHik,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(combine(4,
                                                   2,
                                                   2,
                                                   {calls({7})},
                                                   {ov::element::f32, ov::element::f16, ov::element::bf16},
                                                   all_fusion_types,
                                                   {reference})),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// A decode step (1 token) and a 20-token chunk (more blocks than block_topk), with one and two sequences.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_Lengths,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(combine(4,
                                                   2,
                                                   2,
                                                   {calls({1}), calls({20})},
                                                   {ov::element::f32, ov::element::bf16},
                                                   all_fusion_types,
                                                   {reference})),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// Long sequences over many pages: the 32-block scoring chunks span four 8-slot pages; 16-bit caches flip a few
// near-tied selections, so compare() allows a small mismatch share.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_LongSequences,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(combine(4,
                                                   2,
                                                   2,
                                                   {calls({1}), calls({300})},
                                                   {ov::element::f32, ov::element::bf16},
                                                   all_fusion_types,
                                                   {reference})),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// Several calls over the same caches: a prefill, decode steps, then a chunk; every call checks that the bound
// caches are updated, not a copy.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_MultiCall,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(combine(4,
                                                   2,
                                                   2,
                                                   {calls({7, 1, 1, 1, 5, 1})},
                                                   {ov::element::f32, ov::element::f16, ov::element::bf16},
                                                   all_fusion_types,
                                                   {reference})),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// Degenerate scores (NaN keys, overflow to Inf / NaN, tiny differences) over 40 blocks: no reference, the selection
// invariants are checked.
namespace {

std::vector<PagedQSAIndexerLayerTestParams> degenerate_cases() {
    std::vector<PagedQSAIndexerLayerTestParams> cases;
    for (const size_t h_sel : {1, 2}) {
        for (auto params : combine(0,
                                   2,
                                   h_sel,
                                   {calls({160})},
                                   {ov::element::f32, ov::element::f16, ov::element::bf16},
                                   {QSAIndexerQueryFusionType::kNormAndRope},
                                   {PagedQSAIndexerScenario::kNaNScores},
                                   {2, 4})) {
            cases.push_back(params);
        }
        for (auto params : combine(0,
                                   2,
                                   h_sel,
                                   {calls({160})},
                                   {ov::element::f32, ov::element::bf16},
                                   {QSAIndexerQueryFusionType::kNormAndRope},
                                   {PagedQSAIndexerScenario::kOverflowScores, PagedQSAIndexerScenario::kTinyScores},
                                   {2, 4})) {
            cases.push_back(params);
        }
    }
    return cases;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_DegenerateScores,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(degenerate_cases()),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// The hint differs from the data precision (f32 data, bf16 hint), and the op is in the middle of the graph: the op
// needs one element type on all its float inputs, so EnforceInferencePrecision has to keep the node in f32.
namespace {

std::vector<PagedQSAIndexerLayerTestParams> inference_precision_cases() {
    std::vector<PagedQSAIndexerLayerTestParams> cases;
    for (const size_t num_sequences : {1, 2}) {
        for (const auto& calls_of_test : {calls({7}), calls({7, 1, 1})}) {
            cases.push_back(make_case(4,
                                      2,
                                      2,
                                      num_sequences,
                                      calls_of_test,
                                      QSAIndexerQueryFusionType::kNormAndRope,
                                      reference,
                                      ov::element::f32,
                                      ov::element::bf16));
        }
    }
    return cases;
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_InferencePrecision,
                         PagedQSAIndexerLayerCPUTest,
                         testing::ValuesIn(inference_precision_cases()),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

}  // namespace ov::test
