// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/cum_sum.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> shapes = {
    {16},
    {9, 15},
    {16, 10, 12},
    {5, 14, 5, 7},
    {7, 8, 6, 7, 13},
    {2, 3, 4, 2, 3, 5},
    {4, 3, 6, 2, 3, 4, 5, 2, 3, 4},
};

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32
};

const std::vector<int64_t> axes = { 0, 1, 2, 3, 4, 5, 6};
const std::vector<int64_t> negativeAxes = { -1, -2, -3, -4, -5, -6 };

const std::vector<bool> exclusive = {true, false};
const std::vector<bool> reverse =   {true, false};

const auto testCasesNegativeAxis = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{4, 16, 3, 6, 5, 2}),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::ValuesIn(negativeAxes),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_0 = ::testing::Combine(
    ::testing::ValuesIn(shapes),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[0]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_1 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 1, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[1]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_2 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 2, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[2]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_3 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 3, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[3]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_4 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 4, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[4]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_5 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 5, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[5]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCasesAxis_6 = ::testing::Combine(
    ::testing::ValuesIn(std::vector<std::vector<size_t>>(shapes.begin() + 6, shapes.end())),
    ::testing::ValuesIn(inputPrecision),
    ::testing::Values(axes[6]),
    ::testing::ValuesIn(exclusive),
    ::testing::ValuesIn(reverse),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_negative_axis, CumSumLayerTest, testCasesNegativeAxis, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_0, CumSumLayerTest, testCasesAxis_0, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_1, CumSumLayerTest, testCasesAxis_1, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_2, CumSumLayerTest, testCasesAxis_2, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_3, CumSumLayerTest, testCasesAxis_3, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_4, CumSumLayerTest, testCasesAxis_4, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_5, CumSumLayerTest, testCasesAxis_5, CumSumLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsCumSum_axis_6, CumSumLayerTest, testCasesAxis_6, CumSumLayerTest::getTestCaseName);
