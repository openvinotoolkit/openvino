// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/variadic_split.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using ov::test::VariadicSplitLayerTest;

namespace {
const std::vector<ov::element::Type> model_types = {ov::element::f32, ov::element::f16};

// Sum of elements numSplits = inputShapes[Axis]
const std::vector<std::vector<size_t>> num_splits = {{1, 16, 5, 8},
                                                     {2, 19, 5, 4},
                                                     {7, 13, 2, 8},
                                                     {5, 8, 12, 5},
                                                     {4, 11, 6, 9}};

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{30, 30, 30, 30}},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_NumSplitsCheck,
    VariadicSplitLayerTest,
    ::testing::Combine(::testing::ValuesIn(num_splits),
                       ::testing::Values(0, 1, 2, 3),
                       ::testing::ValuesIn(model_types),
                       ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    VariadicSplitLayerTest::getTestCaseName);
}  // namespace
