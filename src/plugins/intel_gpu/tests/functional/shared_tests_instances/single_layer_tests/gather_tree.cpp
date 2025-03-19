// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/gather_tree.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GatherTreeLayerTest;
using ov::test::utils::InputLayerType;

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32
};

const std::vector<ov::Shape> inputShapes = { {5, 1, 10}, {1, 1, 10}, {20, 1, 10}, {20, 20, 10} };

const std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(Basic_smoke, GatherTreeLayerTest,
        ::testing::Combine(
        ::testing::ValuesIn(inputShapes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::test::utils::DEVICE_GPU)),
        GatherTreeLayerTest::getTestCaseName);

}  // namespace
