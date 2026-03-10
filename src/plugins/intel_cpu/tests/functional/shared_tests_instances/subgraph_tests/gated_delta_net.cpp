// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/gated_delta_net.hpp"

namespace ov {
namespace test {

std::vector<gated_delta_net_params> test_cases = {
    {1, 1, 2, 2, 16, ov::element::f32, "CPU"},
    {1, 39, 2, 2, 16, ov::element::f32, "CPU"},
    {2, 16, 2, 2, 16, ov::element::f32, "CPU"},
    {2, 39, 2, 2, 16, ov::element::f32, "CPU"},
    {1, 16, 2, 2, 128, ov::element::f32, "CPU"},
    {1, 31, 2, 2, 128, ov::element::f32, "CPU"},
};
INSTANTIATE_TEST_SUITE_P(smoke_GatedDeltaNet,
                         GatedDeltaNet,
                         ::testing::ValuesIn(test_cases),
                         GatedDeltaNet::getTestCaseName);
}  // namespace test
}  // namespace ov