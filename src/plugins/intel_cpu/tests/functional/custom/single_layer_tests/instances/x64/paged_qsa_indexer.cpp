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
                                          testing::Values(static_cast<size_t>(7)),
                                          testing::Values(static_cast<size_t>(4))),
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
                                          testing::Values(static_cast<size_t>(7)),
                                          testing::Values(static_cast<size_t>(4))),
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
                                          testing::Values(static_cast<size_t>(1), static_cast<size_t>(20)),
                                          testing::Values(static_cast<size_t>(4))),
                         PagedQSAIndexerLayerCPUTest::getTestCaseName);

// Pages of the size of a real cache, long sequences: the scoring takes 32 blocks at a time, which lie in one page
// (128 tokens = 64 blocks per page, used in place), run over two (96 tokens = 48 blocks per page, gathered), or are
// the blocks of many tiny pages; several tiles of tokens, a decode step over a long cache, a group of heads.
// The oracle keeps the block keys in f32 while the kernel stores them in the cache precision, so with 150 blocks to
// choose the best 2 from, a 16-bit cache flips a few near-tied selections (the reference scalar kernel does the same):
// the test allows a small fraction of the indices to differ for the 16-bit precisions (see compare()).
INSTANTIATE_TEST_SUITE_P(
    smoke_PagedQSAIndexer_CPU_LongSequences,
    PagedQSAIndexerLayerCPUTest,
    testing::Combine(testing::Values(ov::test::utils::DEVICE_CPU),
                     testing::Values(ov::element::f32, ov::element::bf16),
                     testing::Values(static_cast<size_t>(4)),
                     testing::Values(static_cast<size_t>(2)),
                     testing::Values(static_cast<size_t>(2)),
                     testing::Values(static_cast<size_t>(1), static_cast<size_t>(2)),
                     testing::Values(static_cast<size_t>(1), static_cast<size_t>(300)),
                     testing::Values(static_cast<size_t>(8), static_cast<size_t>(96), static_cast<size_t>(128))),
    PagedQSAIndexerLayerCPUTest::getTestCaseName);

}  // namespace test
}  // namespace ov