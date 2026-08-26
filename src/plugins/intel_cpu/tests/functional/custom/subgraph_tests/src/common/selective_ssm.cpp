// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/selective_ssm.hpp"

namespace ov::test {

std::vector<selective_ssm_params> selective_ssm_test_cases = {
    {1, 1, 1, 1, 1, 1, ov::element::f32, "CPU"},
    {1, 0, 2, 1, 3, 4, ov::element::f32, "CPU"},
    {1, 3, 4, 4, 5, 3, ov::element::f32, "CPU"},
    {2, 5, 6, 3, 7, 5, ov::element::f32, "CPU"},
    {1, 4, 4, 2, 8, 16, ov::element::f32, "CPU"},
    {2, 3, 4, 1, 8, 8, ov::element::f32, "CPU"},
    {1, 4, 4, 2, 8, 16, ov::element::f16, "CPU"},
    {1, 1, 4, 2, 8, 16, ov::element::f16, "CPU"},
    {1, 4, 4, 2, 8, 16, ov::element::bf16, "CPU"},
    {1, 1, 4, 2, 8, 16, ov::element::bf16, "CPU"},
    {1, 1, 64, 1, 64, 128, ov::element::f32, "CPU"},
    {1, 5, 96, 8, 80, 80, ov::element::f32, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_SelectiveSSM,
                         SelectiveSSM,
                         ::testing::ValuesIn(selective_ssm_test_cases),
                         SelectiveSSM::getTestCaseName);

}  // namespace ov::test
