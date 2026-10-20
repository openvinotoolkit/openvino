// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/paged_selective_ssm.hpp"

#include <vector>

namespace ov::test {

namespace {
std::vector<PagedSelectiveSSMLayerParams> jit_numerical_cases() {
    std::vector<PagedSelectiveSSMLayerParams> cases{
        {2, 1, 3, 5, {0, 0}, {0, 11}, {0, -7}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {1, 1, 1, 1, {1}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 3, 5, {1}, {7}, {0}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {4, 2, 3, 5, {1}, {4}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {4, 4, 5, 3, {1}, {3}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {6, 3, 7, 9, {2, 0, 5}, {0, 11, 4}, {3, 0, 2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {4, 1, 2, 7, {4, 3}, {1, 0}, {3, 2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 2, 5, 4, {3, 2}, {5, 9}, {0, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 5, 16, {2}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 5, 17, {2}, {1}, {3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 5, 128, {2}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 5, 129, {2}, {3}, {4}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 17, 127, {7, 8, 9}, {0, 3, 7}, {3, 4, 0}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {4, 2, 33, 129, {8, 1, 17}, {1, 0, 4}, {1, 3, 5}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 32, 128, {17, 8}, {3, 9}, {0, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 33, 1, {8, 9}, {0, 1}, {2, 3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {2, 1, 65, 3, {8, 9, 17}, {1, 0, 3}, {4, 1, 5}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
        {4,
         2,
         5,
         129,
         {63, 64, 65, 0},
         {1, 0, 64, 0},
         {3, 64, 65, 0},
         ov::element::f32,
         ov::element::f32,
         ov::element::i32,
         "CPU"},
        {4, 2, 9, 17, {65, 64}, {7, 3}, {0, -1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    };
    const auto shapes = cases;
    cases.clear();
    for (const auto& precision : {ov::element::f32, ov::element::f16, ov::element::bf16}) {
        for (auto params : shapes) {
            std::get<7>(params) = precision;
            std::get<8>(params) = precision;
            cases.push_back(params);
        }
    }
    // With snapshots disabled, prefill exercises the no-store state mode and checks the complete unchanged cache.
    for (const int32_t state : {1, 2, 3, 4, 5, 7, 8, 127, 128, 129, 135, 255, 256, 257, 263, 4095, 4096}) {
        for (const int32_t rows : {1, 4, 5, 8, 9, 16, 17, 32, 33, 64, 65}) {
            cases.push_back(
                {2, 1, rows, state, {3}, {7}, {0}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"});
        }
    }
    return cases;
}
}  // namespace

std::vector<PagedSelectiveSSMLayerParams> paged_selective_ssm_test_cases = {
    {4, 2, 5, 3, {3}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3}, {4}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {4}, {1}, {3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {4}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {3}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {0, 2, 1}, {7, 1, 0}, {2, 2, 4}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {0, 0}, {0, 7}, {2, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 1, 3, 4, {3, 2}, {5, 9}, {0, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {3, 3, 1, 1, {4}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    // Snapshot boundaries split JIT token batches; include boundaries on either side of the 64-token limit.
    {4,
     2,
     5,
     129,
     {63, 64, 65, 0},
     {1, 0, 64, 0},
     {3, 64, 65, 0},
     ov::element::f32,
     ov::element::f32,
     ov::element::i32,
     "CPU"},
    {4, 2, 9, 17, {129, 65}, {7, 3}, {64, 0}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    // Numerical coverage through the executor: row/vector tails, large states and single-snapshot aliasing.
    {2, 1, 17, 127, {7, 8, 9}, {0, 3, 7}, {3, 4, 0}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 33, 129, {8, 1, 17}, {1, 0, 4}, {1, 3, 5}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {2, 1, 9, 4095, {8}, {3}, {4}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {2, 1, 9, 4096, {8}, {3}, {4}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {2, 1, 5, 129, {5}, {0}, {16}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {2, 1, 5, 129, {5}, {3}, {16}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {64, 1, 64, 128, {1}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {96, 8, 80, 80, {5}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::f16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::bf16, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f32, ov::element::f16, ov::element::i32, "CPU"},
    {64, 1, 64, 128, {1}, {0}, {1}, ov::element::f32, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::f32, ov::element::i32, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedSelectiveSSM,
                         PagedSelectiveSSMLayerTest,
                         ::testing::ValuesIn(paged_selective_ssm_test_cases),
                         PagedSelectiveSSMLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PagedSelectiveSSMJitNumerical,
                         PagedSelectiveSSMLayerTest,
                         ::testing::ValuesIn(jit_numerical_cases()),
                         PagedSelectiveSSMLayerTest::getTestCaseName);

}  // namespace ov::test
