// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/minimum_maximum.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::MaxMinLayerTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::MinMaxOpType;

const std::vector<std::vector<ov::Shape>> input_shapes_static = {
        {{2}, {1}},
        {{1, 1, 1, 3}, {1}},
        {{1, 2, 4}, {1}},
        {{1, 4, 4}, {1}},
        {{1, 4, 4, 1}, {1}},
        {{256, 56}, {256, 56}},
        {{8, 1, 6, 1}, {7, 1, 5}},
};

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16,
};

const std::vector<MinMaxOpType> op_types = {
        MinMaxOpType::MINIMUM,
        MinMaxOpType::MAXIMUM,
};

const std::vector<InputLayerType> second_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

INSTANTIATE_TEST_SUITE_P(smoke_maximum, MaxMinLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::ValuesIn(op_types),
                                ::testing::ValuesIn(model_types),
                                ::testing::ValuesIn(second_input_types),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        MaxMinLayerTest::getTestCaseName);

}  // namespace
