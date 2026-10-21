// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/subgraph_tests/include/selective_ssm.hpp"

#include <algorithm>
#include <array>
#include <vector>

namespace ov::test {

namespace {
std::vector<selective_ssm_params> jit_numerical_cases() {
    std::vector<selective_ssm_params> cases;
    const auto append = [&](const selective_ssm_params& params) {
        if (std::find(cases.begin(), cases.end(), params) == cases.end()) {
            cases.push_back(params);
        }
    };
    // Exercise grouping, empty inputs and prefill/decode through CPU inference instead of direct kernel calls.
    const std::vector<std::array<int32_t, 6>> shapes{
        {0, 3, 2, 1, 3, 5},    {2, 0, 4, 2, 5, 3},    {1, 1, 1, 1, 1, 1},    {1, 1, 2, 1, 5, 17},
        {2, 2, 4, 4, 3, 5},    {1, 5, 6, 3, 7, 9},    {2, 3, 8, 4, 5, 7},    {1, 4, 3, 1, 9, 2},
        {1, 2, 2, 1, 5, 16},   {1, 2, 2, 1, 5, 17},   {1, 2, 2, 1, 5, 128},  {1, 2, 2, 1, 5, 129},
        {1, 7, 2, 1, 17, 127}, {1, 8, 2, 1, 17, 128}, {2, 9, 4, 2, 33, 129}, {1, 17, 2, 1, 32, 128},
        {1, 8, 2, 1, 33, 1},   {1, 9, 2, 1, 65, 3},   {2, 63, 4, 2, 5, 17},  {1, 64, 4, 2, 9, 128},
        {2, 65, 4, 2, 5, 129},
    };
    for (const auto& precision : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        for (const auto& shape : shapes) {
            append({shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], precision, "CPU"});
        }
    }
    // Dynamic-shape iterations also run three-token prefill for each single-token decode case.
    for (const int32_t state : {1, 2, 3, 4, 5, 7, 8, 127, 128, 129, 135, 255, 256, 257, 263, 4095, 4096}) {
        for (const int32_t rows : {1, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65}) {
            append({1, 1, 2, 1, rows, state, ov::element::f32, "CPU"});
        }
    }
    return cases;
}
}  // namespace

std::vector<selective_ssm_params> selective_ssm_test_cases = {
    {1, 1, 1, 1, 1, 1, ov::element::f32, "CPU"},
    {1, 0, 2, 1, 3, 4, ov::element::f32, "CPU"},
    {1, 3, 4, 4, 5, 3, ov::element::f32, "CPU"},
    {2, 5, 6, 3, 7, 5, ov::element::f32, "CPU"},
    {1, 4, 4, 2, 8, 16, ov::element::f32, "CPU"},
    {2, 3, 4, 1, 8, 8, ov::element::f32, "CPU"},
    // Exercise JIT token-batch boundaries with row/vector tails and grouped projections through CPU inference.
    {2, 63, 4, 2, 5, 17, ov::element::f32, "CPU"},
    {1, 64, 4, 2, 9, 128, ov::element::f32, "CPU"},
    {2, 65, 4, 2, 5, 129, ov::element::f32, "CPU"},
    {1, 129, 4, 2, 5, 17, ov::element::f32, "CPU"},
    {1, 7, 2, 1, 17, 127, ov::element::f32, "CPU"},
    {1, 8, 2, 1, 17, 128, ov::element::f32, "CPU"},
    {2, 9, 4, 2, 33, 129, ov::element::f32, "CPU"},
    {1, 17, 2, 1, 32, 128, ov::element::f32, "CPU"},
    {1, 8, 2, 1, 33, 1, ov::element::f32, "CPU"},
    {1, 9, 2, 1, 65, 3, ov::element::f32, "CPU"},
    {1, 8, 2, 1, 9, 4095, ov::element::f32, "CPU"},
    {1, 8, 2, 1, 9, 4096, ov::element::f32, "CPU"},
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

INSTANTIATE_TEST_SUITE_P(smoke_SelectiveSSMJitNumerical,
                         SelectiveSSM,
                         ::testing::ValuesIn(jit_numerical_cases()),
                         SelectiveSSM::getTestCaseName);

}  // namespace ov::test
