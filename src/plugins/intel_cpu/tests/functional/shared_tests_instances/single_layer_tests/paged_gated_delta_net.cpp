// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/paged_gated_delta_net.hpp"

namespace ov::test {

std::vector<PagedGatedDeltaNetLayerParams> paged_gdn_test_cases = {
    // PagedGatedDeltaNetLayerParams
    // {int32_t qk_heads,
    //  int32_t v_heads,
    //  int32_t qk_head_size,
    //  int32_t v_head_size,
    //  std::vector<int32_t> seq_lengths,
    //  std::vector<int32_t> cache_intervals,
    //  ov::element::Type element_type,
    //  std::string target_device}
    {2, 4, 8, 8, {3, 3}, {2, 3}, ov::element::f32, "CPU"},
    {2, 4, 8, 8, {3, 2}, {0, 0}, ov::element::f32, "CPU"},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}, ov::element::f32, "CPU"},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}, ov::element::f32, "CPU"},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}, ov::element::f32, "CPU"},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}, ov::element::f32, "CPU"},
    {2, 4, 8, 8, {3, 3}, {2, 3}, ov::element::f16, "CPU"},
    {2, 4, 8, 8, {3, 2}, {0, 0}, ov::element::f16, "CPU"},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}, ov::element::f16, "CPU"},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}, ov::element::f16, "CPU"},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}, ov::element::f16, "CPU"},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}, ov::element::f16, "CPU"},
    {2, 4, 8, 8, {3, 3}, {2, 3}, ov::element::bf16, "CPU"},
    {2, 4, 8, 8, {3, 2}, {0, 0}, ov::element::bf16, "CPU"},
    {1, 4, 16, 8, {2, 5, 1}, {1, 4, 2}, ov::element::bf16, "CPU"},
    {2, 6, 32, 64, {4, 2, 3}, {3, 2, 5}, ov::element::bf16, "CPU"},
    {4, 8, 32, 64, {15, 32, 33}, {16, 16, 16}, ov::element::bf16, "CPU"},
    {8, 8, 128, 128, {15, 32, 33}, {16, 16, 16}, ov::element::bf16, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedGatedDeltaNet,
                         PagedGatedDeltaNetLayerTest,
                         ::testing::ValuesIn(paged_gdn_test_cases),
                         PagedGatedDeltaNetLayerTest::getTestCaseName);

}  // namespace ov::test
