// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/result.hpp"

using ov::test::ResultLayerTest;

namespace {
std::vector<std::vector<size_t>> input_shapes = {
    {7},
    {1000},
    {3, 5},
    {65, 33},
    {33, 65},
    {1, 1000},
    {223, 217, 21},
    {3, 4, 5, 1},
    {3, 4, 1, 5, 1}
};

std::vector<ov::element::Type> model_types = {
    ov::element::u8,
    ov::element::f32,
    ov::element::i32,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ResultLayerTest, ResultLayerTest,
    ::testing::Combine(::testing::ValuesIn(input_shapes),
                       ::testing::ValuesIn(model_types),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
                       ResultLayerTest::getTestCaseName);
} // namespace
