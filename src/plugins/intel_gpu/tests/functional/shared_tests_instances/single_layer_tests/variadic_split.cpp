// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/variadic_split.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::VariadicSplitLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16
};

// Sum of elements numSplits = inputShapes[Axis]
const std::vector<std::vector<size_t>> numSplits = {
        {1, 16, 5, 8},
        {2, 19, 5, 4},
        {7, 13, 2, 8},
        {5, 8, 12, 5},
        {4, 11, 6, 9}
};

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck, VariadicSplitLayerTest,
        ::testing::Combine(
            ::testing::ValuesIn(numSplits),
            ::testing::Values(0, 1, 2, 3),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{30, 30, 30, 30}}))),
            ::testing::Values(ov::test::utils::DEVICE_GPU)),
        VariadicSplitLayerTest::getTestCaseName);
}  // namespace
