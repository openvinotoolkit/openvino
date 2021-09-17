// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roll.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::I8,
    InferenceEngine::Precision::U8,
    InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::BF16
};

const auto testCase2DZeroShifts = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{17, 19}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{0, 0}), // Shift
    ::testing::Values(std::vector<int64_t>{0, 1}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCase1D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{16}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{5}), // Shift
    ::testing::Values(std::vector<int64_t>{0}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCase2D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{600, 450}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{300, 250}), // Shift
    ::testing::Values(std::vector<int64_t>{0, 1}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCase3D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{2, 320, 320}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{160, 160}), // Shift
    ::testing::Values(std::vector<int64_t>{1, 2}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCaseNegativeUnorderedAxes4D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 11, 6, 4}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{7, 3}), // Shift
    ::testing::Values(std::vector<int64_t>{-3, -2}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCaseRepeatingAxes5D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{2, 16, 32, 32}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{16, 15, 10, 2, 1, 7, 2, 8, 1, 1}), // Shift
    ::testing::Values(std::vector<int64_t>{-1, -2, -3, 1, 0, 3, 3, 2, -2, -3}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCaseNegativeShifts6D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{4, 16, 3, 6, 5, 2}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{-2, -15, -2, -1, -4, -1}), // Shift
    ::testing::Values(std::vector<int64_t>{0, 1, 2, 3, 4, 5}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto testCaseUnordNegAxesAndShifts10D = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{2, 2, 4, 2, 3, 6, 3, 2, 3, 2}), // Input shape
    ::testing::ValuesIn(inputPrecision), // Precision
    ::testing::Values(std::vector<int64_t>{-2, -1, 1, 1, 1, -2}), // Shift
    ::testing::Values(std::vector<int64_t>{-6, -4, -3, 1, -10, -2}), // Axes
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_2d_zero_shifts, RollLayerTest, testCase2DZeroShifts, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_1d, RollLayerTest, testCase1D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_2d, RollLayerTest, testCase2D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_3d, RollLayerTest, testCase3D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_negative_unordered_axes_4d, RollLayerTest, testCaseNegativeUnorderedAxes4D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_negative_unordered_axes_5d, RollLayerTest, testCaseRepeatingAxes5D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_negative_shifts_6d, RollLayerTest, testCaseNegativeShifts6D, RollLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_MKLDNN_TestsRoll_unord_neg_shifts_and_axes_10d, RollLayerTest, testCaseUnordNegAxesAndShifts10D, RollLayerTest::getTestCaseName);

}  // namespace
