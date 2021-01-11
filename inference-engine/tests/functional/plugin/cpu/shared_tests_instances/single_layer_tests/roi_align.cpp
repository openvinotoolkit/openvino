// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const auto ROIAlignCases_average = ::testing::Combine(
        ::testing::ValuesIn(
    std::vector<std::vector<size_t>> {
            { 3, 8, 16, 16 },
            { 2, 1, 16, 16 }}),
        ::testing::Values(std::vector<size_t>{ 2, 4 }),
        ::testing::Values(2),
        ::testing::Values(2),
        ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
        ::testing::Values(2),
        ::testing::Values("avg"),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsROIAlign_average, ROIAlignLayerTest, ROIAlignCases_average, ROIAlignLayerTest::getTestCaseName);

const auto ROIAlignCases_max = ::testing::Combine(
        ::testing::ValuesIn(
        std::vector<std::vector<size_t>> {
            { 2, 8, 20, 20 },
            { 2, 1, 20, 20 }}),
        ::testing::Values(std::vector<size_t>{ 2, 4 }),
        ::testing::Values(2),
        ::testing::Values(2),
        ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
        ::testing::Values(2),
        ::testing::Values("max"),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsROIAlign_max, ROIAlignLayerTest, ROIAlignCases_max, ROIAlignLayerTest::getTestCaseName);
