// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "single_op_tests/group_normalization.hpp"

namespace {
using ov::test::GroupNormalizationTest;

const std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
};

// static shapes
const std::vector<ov::Shape> staticInputShapes = {
    {3, 8, 3},
    {3, 8, 8},
    {3, 8, 16},
    {3, 8, 21},
    {1, 4, 8, 8},
    {1, 8, 1, 22},
    {3, 16, 1, 33},
    {1, 4, 1, 1, 34},
    {1, 8, 1, 8, 2, 2},
    {1, 8, 1, 8, 2, 2, 2},
};

// dynmaic shapes
const std::vector<ov::test::InputShape> DynamicInputShapes = {
    {{-1, -1, -1}, {{1, 8, 22}, {2, 4, 7}, {1, 8, 22}}},
    {{-1, -1, -1, -1}, {{1, 16, 8, 8}, {2, 8, 4, 4}, {1, 16, 8, 8}}},
    {{{1, 4}, {4, 16}, -1, -1}, {{1, 4, 6, 6}, {4, 16, 10, 10}, {1, 4, 6, 6}}},
    {{-1, -1, -1, -1, -1}, {{1, 16, 7, 7, 1}, {2, 8, 4, 4, 1}, {1, 16, 7, 7, 1}}},
};

const std::vector<int64_t> numGroups = {
    2, 4,
};

const std::vector<double> epsilon = {
    0.0001
};

std::vector<ov::AnyMap> additionalConfig = {
    {{ov::hint::inference_precision(ov::element::f32)}},
    {{ov::hint::inference_precision(ov::element::bf16)}}
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupNormalizationStatic,
    GroupNormalizationTest,
    testing::Combine(testing::ValuesIn(netPrecisions),
                     ::testing::Values(ov::element::dynamic),
                     ::testing::Values(ov::element::dynamic),
                     testing::ValuesIn(ov::test::static_shapes_to_test_representation(staticInputShapes)),
                     testing::ValuesIn(numGroups),
                     testing::ValuesIn(epsilon),
                     testing::Values(ov::test::utils::DEVICE_CPU),
                     testing::ValuesIn(additionalConfig)),
    GroupNormalizationTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupNormalizationDyn,
                         GroupNormalizationTest,
                         testing::Combine(testing::ValuesIn(netPrecisions),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          testing::ValuesIn(DynamicInputShapes),
                                          testing::ValuesIn(numGroups),
                                          testing::ValuesIn(epsilon),
                                          testing::Values(ov::test::utils::DEVICE_CPU),
                                          testing::ValuesIn(additionalConfig)),
                         GroupNormalizationTest::getTestCaseName);

} // anonymous namespace
