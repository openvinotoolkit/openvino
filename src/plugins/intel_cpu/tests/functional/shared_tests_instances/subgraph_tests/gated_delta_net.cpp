// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/gated_delta_net.hpp"

namespace ov::test {

std::vector<gated_delta_net_params> test_cases = {
    {1, 1, 2, 2, 16, 16, ov::element::f32, "CPU"},
    {1, 39, 2, 2, 16, 32, ov::element::f32, "CPU"},
    {2, 16, 2, 2, 16, 16, ov::element::f32, "CPU"},
    {2, 39, 2, 2, 16, 16, ov::element::f32, "CPU"},
    // grouped-query cases: qk_heads != v_heads
    {1, 16, 2, 4, 16, 16, ov::element::f32, "CPU"},
    {2, 8, 4, 8, 16, 16, ov::element::f32, "CPU"},
    {1, 16, 2, 2, 128, 128, ov::element::f32, "CPU"},
    {1, 16, 2, 2, 64, 128, ov::element::f32, "CPU"},
    {1, 31, 2, 2, 128, 128, ov::element::f32, "CPU"},
    // large batch
    {32, 2, 32, 32, 16, 16, ov::element::f32, "CPU"},
    {32, 2, 64, 64, 16, 16, ov::element::f32, "CPU"},
    //  odd head size
    {1, 2, 2, 2, 7, 7, ov::element::f32, "CPU"},
    {1, 2, 2, 2, 15, 15, ov::element::f32, "CPU"},
    {1, 2, 2, 2, 31, 31, ov::element::f32, "CPU"},
    {1, 2, 2, 2, 1, 1, ov::element::f32, "CPU"},
};
INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet,
                         GatedDeltaNet,
                         ::testing::ValuesIn(test_cases),
                         GatedDeltaNet::getTestCaseName);
}  // namespace ov::test