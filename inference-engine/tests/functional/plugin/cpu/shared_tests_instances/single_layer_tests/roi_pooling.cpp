// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {
    {1, 3, 8, 8},
    {3, 4, 50, 50}
};

const std::vector<std::vector<size_t>> pooledShapes_max = {
    {1, 1},
    {2, 2},
    {3, 3},
    {6, 6}
};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {
    {1, 1},
    {2, 2},
    {3, 3},
    {6, 6}
};

const std::vector<std::vector<size_t>> coordShapes = {
    {1, 5},
    {3, 5},
    {5, 5}
};

const std::vector<InferenceEngine::Precision> netPRCs = {
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(coordShapes),
    ::testing::ValuesIn(pooledShapes_max),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
    ::testing::ValuesIn(netPRCs),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto test_ROIPooling_bilinear = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(coordShapes),
    ::testing::ValuesIn(pooledShapes_bilinear),
    ::testing::Values(spatial_scales[1]),
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
    ::testing::ValuesIn(netPRCs),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest, test_ROIPooling_max, ROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest, test_ROIPooling_bilinear, ROIPoolingLayerTest::getTestCaseName);
