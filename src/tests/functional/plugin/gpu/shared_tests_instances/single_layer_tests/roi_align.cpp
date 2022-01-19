// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <vector>

#include "single_layer_tests/roi_align.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;


const std::vector<InferenceEngine::Precision> netPRCs = {
    InferenceEngine::Precision::FP32,
    // There is no possibility to test ROIAlign in fp16 precision,
    // because on edge cases where in fp32 version ROI value is
    // a little bit smaller than the nearest integer value,
    // it would be bigger than the nearest integer in fp16 precision.
    // Such behavior leads to completely different results of ROIAlign
    // in fp32 and fp16 precisions.
    // In real AI applications this problem is solved by precision-aware training.

    // InferenceEngine::Precision::FP16,
};

const auto ROIAlignCases_average =
    ::testing::Combine(
                       ::testing::ValuesIn(
                                           std::vector<std::vector<size_t>> {
                                               { 3, 8, 16, 16 },
                                               { 2, 1, 16, 16 },
                                               { 2, 1, 8, 16 }}),
                       ::testing::Values(std::vector<size_t>{ 2, 4 }),
                       ::testing::Values(2),
                       ::testing::Values(2),
                       ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
                       ::testing::Values(2),
                       ::testing::Values("avg"),
                       ::testing::ValuesIn(netPRCs),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_average, ROIAlignLayerTest, ROIAlignCases_average, ROIAlignLayerTest::getTestCaseName);

const auto ROIAlignCases_max =
    ::testing::Combine(
                       ::testing::ValuesIn(
                                           std::vector<std::vector<size_t>> {
                                               { 2, 8, 20, 20 },
                                               { 2, 1, 20, 20 },
                                               { 2, 1, 10, 20 }
                                           }),
                       ::testing::Values(std::vector<size_t>{ 2, 4 }),
                       ::testing::Values(2),
                       ::testing::Values(2),
                       ::testing::ValuesIn(std::vector<float> { 1, 0.625 }),
                       ::testing::Values(2),
                       ::testing::Values("max"),
                       ::testing::ValuesIn(netPRCs),
                       ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIAlign_max, ROIAlignLayerTest, ROIAlignCases_max, ROIAlignLayerTest::getTestCaseName);
