// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/paged_qsa_indexer.hpp"
#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

// H_iq == H_ik, h_sel == 1: every index-query head contributes to the single selection head.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_Hsel1,
                         PagedQSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(7))),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// H_iq = 2 * H_ik, h_sel == H_ik: exercises the grouped per-selection-head scoring path.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_HselEqualsHik,
                         PagedQSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(7))),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// Other query lengths: a decode step (1 token: a block completes or not, nothing or the past blocks to score) and a
// chunk of 20 tokens (more blocks than block_topk, several pages, a sequence with enough tokens to spread over
// threads), one and two sequences.
INSTANTIATE_TEST_SUITE_P(smoke_PagedQSAIndexer_CPU_Lengths,
                         PagedQSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(20))),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov