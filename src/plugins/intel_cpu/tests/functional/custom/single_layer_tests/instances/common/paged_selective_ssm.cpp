// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/paged_selective_ssm.hpp"

namespace ov::test {

std::vector<PagedSelectiveSSMLayerParams> paged_selective_ssm_test_cases = {
    {4, 2, 5, 3, {3}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3}, {4}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {4}, {1}, {3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {4}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {3}, {2}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {0, 2, 1}, {7, 1, 0}, {2, 2, 4}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {0, 0}, {0, 7}, {2, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {4, 1, 3, 4, {3, 2}, {5, 9}, {0, -3}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {3, 3, 1, 1, {4}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i64, "CPU"},
    {64, 1, 64, 128, {1}, {0}, {1}, ov::element::f32, ov::element::f32, ov::element::i32, "CPU"},
    {96, 8, 80, 80, {5}, {0}, {2}, ov::element::f32, ov::element::f32, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::f16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::f16, ov::element::f16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::f16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::bf16, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::bf16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {1}, {0}, {2}, ov::element::bf16, ov::element::bf16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::bf16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f32, ov::element::f16, ov::element::i32, "CPU"},
    {64, 1, 64, 128, {1}, {0}, {1}, ov::element::f32, ov::element::bf16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::f32, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::f16, ov::element::bf16, ov::element::i64, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::f16, ov::element::i32, "CPU"},
    {4, 2, 5, 3, {3, 1}, {1, 4}, {3, 2}, ov::element::bf16, ov::element::f32, ov::element::i64, "CPU"},
};

INSTANTIATE_TEST_SUITE_P(smoke_PagedSelectiveSSM,
                         PagedSelectiveSSMLayerTest,
                         ::testing::ValuesIn(paged_selective_ssm_test_cases),
                         PagedSelectiveSSMLayerTest::getTestCaseName);

}  // namespace ov::test
