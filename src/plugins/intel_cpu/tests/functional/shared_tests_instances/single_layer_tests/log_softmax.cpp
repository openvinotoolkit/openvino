// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/log_softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LogSoftmaxLayerTest;

const std::vector<ov::element::Type> model_types = {
    ov::element::f32,
};

const std::vector<std::vector<ov::Shape>> input_shapes_2d = {
    {{1, 100}},
    {{100, 1}},
    {{10, 10}},
};

const std::vector<int64_t> axis_2d = {
    -2, -1, 0, 1
};

const auto params_2d = testing::Combine(
    testing::ValuesIn(model_types),
    testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_2d)),
    testing::ValuesIn(axis_2d),
    testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_LogSoftmax2D,
        LogSoftmaxLayerTest,
        params_2d,
        LogSoftmaxLayerTest::getTestCaseName
);

const std::vector<std::vector<ov::Shape>> input_shapes_4d = {
    {{1, 100, 1, 1}},
    {{1, 3, 4, 3}},
    {{2, 3, 4, 5}},
};

const std::vector<int64_t> axis_4d = {
    -4, -3, -2, -1, 0, 1, 2, 3
};

const auto params_4d = testing::Combine(
    testing::ValuesIn(model_types),
    testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_4d)),
    testing::ValuesIn(axis_4d),
    testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(
        smoke_LogSoftmax4D,
        LogSoftmaxLayerTest,
        params_4d,
        LogSoftmaxLayerTest::getTestCaseName
);

}  // namespace
