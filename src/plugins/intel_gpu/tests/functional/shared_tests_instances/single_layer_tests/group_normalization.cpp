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

const std::vector<int64_t> numGroups = {
    2, 4,
};

const std::vector<double> epsilon = {
    0.0025
};

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupNormalization,
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

} // anonymous namespace
