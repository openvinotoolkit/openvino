// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::MaxMinLayerTest;

const std::vector<std::vector<ov::Shape>> inShapes = {
        {{2}, {1}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{256, 56}, {256, 56}},
        {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
};

const std::vector<ov::test::utils::MinMaxOpType> opType = {
        ov::test::utils::MinMaxOpType::MINIMUM,
        ov::test::utils::MinMaxOpType::MAXIMUM,
};

const std::vector<ov::test::utils::InputLayerType> second_inputType = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::ValuesIn(opType),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(second_inputType),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        MaxMinLayerTest::getTestCaseName);

}  // namespace
