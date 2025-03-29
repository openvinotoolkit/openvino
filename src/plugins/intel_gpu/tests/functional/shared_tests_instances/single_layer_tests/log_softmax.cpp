// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/log_softmax.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LogSoftmaxLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
};

const std::vector<std::vector<ov::Shape>> inputShapes2D = {
    {{1, 100}},
    {{100, 1}},
    {{10, 10}},
};

const std::vector<int64_t> axis2D = {
    -1, 1
};

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax2D,
                         LogSoftmaxLayerTest,
                         testing::Combine(testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes2D)),
                                          testing::ValuesIn(axis2D),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         LogSoftmaxLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inputShapes4D = {
    {{1, 100, 1, 1}},
    {{1, 3, 4, 3}},
    {{2, 3, 4, 5}},
};

const std::vector<int64_t> axis4D = {
    -3, -2, -1, 1, 2, 3
};

INSTANTIATE_TEST_SUITE_P(smoke_LogSoftmax4D,
                         LogSoftmaxLayerTest,
                         testing::Combine(testing::ValuesIn(netPrecisions),
                                          testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes4D)),
                                          testing::ValuesIn(axis4D),
                                          testing::Values(ov::test::utils::DEVICE_GPU)),
                         LogSoftmaxLayerTest::getTestCaseName);

}  // namespace
