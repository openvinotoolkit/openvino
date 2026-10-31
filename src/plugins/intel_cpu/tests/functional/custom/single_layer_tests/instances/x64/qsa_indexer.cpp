// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/qsa_indexer.hpp"

#include "utils/cpu_test_utils.hpp"

namespace ov {
namespace test {

// Dense (no padding) single-call prefill: H_iq == H_ik, h_sel == 1.
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_DensePrefill_Hsel1,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kDensePrefill),
                                          testing::Values(false)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

// Dense (no padding) single-call prefill: H_iq = 2 * H_ik, h_sel == H_ik.
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_DensePrefill_HselEqualsHik,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kDensePrefill),
                                          testing::Values(false)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

// new_spec/qsa-prepare-17.rst's worked example verbatim: left-padded prefill followed by a decode
// call that completes an incomplete tail block. Scenario forces batch == 2; the batch axis here is
// a don't-care placeholder.
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_WorkedExample,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kWorkedExample),
                                          testing::Values(false)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

// A row without any real token (key_length == 0) next to a regular one, then that row starting from scratch.
// H_iq in {2, 4} x h_sel in {1, 2} with H_ik == 2.
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_FullyPaddedRow,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2), static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kFullyPaddedRow),
                                          testing::Values(false)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

// Several calls with L > 1 and different past lengths per row (non-empty incomplete-block tails, blocks completed
// mid-call and exactly at the end of a call).
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_RaggedCalls,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2), static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kRaggedCalls),
                                          testing::Values(false)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

// The chosen blocks of a row by descending score (equal scores: ascending block id), the order of
// SparseAttentionIndexer-17, instead of ascending block ids: dense prefill with one and with several selection heads,
// the worked example and calls with ragged lengths (which also cover the tokens that take all their blocks).
INSTANTIATE_TEST_SUITE_P(smoke_QSAIndexer_CPU_OrderByScore,
                         QSAIndexerLayerCPUTest,
                         testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::Values(ov::element::f32, ov::element::f16, ov::element::bf16),
                                          testing::Values(static_cast<size_t>(2), static_cast<size_t>(4)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                                          testing::Values(static_cast<size_t>(2)),
                                          testing::Values(QSAIndexerScenario::kDensePrefill,
                                                          QSAIndexerScenario::kWorkedExample,
                                                          QSAIndexerScenario::kRaggedCalls),
                                          testing::Values(true)),
                         QSAIndexerLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov
