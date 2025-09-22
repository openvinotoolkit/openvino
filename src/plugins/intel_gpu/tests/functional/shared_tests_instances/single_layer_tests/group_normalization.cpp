// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/group_normalization.hpp"

using namespace ov::test;

namespace {

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f16,
    ov::element::f32,
};

const std::vector<ov::Shape> inputShapes = {
    {3, 8, 32, 64},
    {3, 8, 28, 32, 12},
};

// dynmaic shapes
const std::vector<ov::test::InputShape> DynamicInputShapes = {
    {{-1, -1, -1}, {{1, 8, 22}, {2, 4, 7}, {1, 8, 22}}},
    {{-1, -1, -1, -1}, {{1, 16, 8, 8}, {2, 8, 4, 4}, {1, 16, 8, 8}}},
    {{{1, 4}, {4, 16}, -1, -1}, {{1, 4, 6, 6}, {4, 16, 10, 10}, {1, 4, 6, 6}}},
    {{-1, -1, -1, -1, -1}, {{1, 16, 7, 7, 1}, {2, 8, 4, 4, 1}, {1, 16, 7, 7, 1}}},
    {{1, 64, {25, 50}, {31, 62}}, {{1, 64, 25, 31}}},
    {{{1, 2}, 64, 25, 31}, {{1, 64, 25, 31}}},
};

const std::vector<int64_t> numGroups = {
    2, 4,
};

const std::vector<double> epsilon = {
    0.0025
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupNormalizationStatic,
    GroupNormalizationTest,
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
                     testing::ValuesIn(numGroups),
                     testing::ValuesIn(epsilon),
                     testing::Values(ov::test::utils::DEVICE_GPU),
                     testing::Values(ov::AnyMap())),
    GroupNormalizationTest::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(
    smoke_GroupNormalizationDynamic,
    GroupNormalizationTest,
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(DynamicInputShapes),
                     testing::ValuesIn(numGroups),
                     testing::ValuesIn(epsilon),
                     testing::Values(ov::test::utils::DEVICE_GPU),
                     testing::Values(ov::AnyMap())),
    GroupNormalizationTest::getTestCaseName);
} // anonymous namespace
