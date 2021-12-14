// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<size_t>> inShapes = {
    {1, 4, 50, 50}
};

const std::vector<std::vector<size_t>> pooledShapes_max = {
    {1, 1},
    {3, 3},
};

const std::vector<std::vector<size_t>> pooledShapes_bilinear = {
    {2, 2},
    {6, 6}
};

const std::vector<std::vector<size_t>> coordShapes = {
    {1, 5},
    {3, 5},
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(coordShapes),
    ::testing::ValuesIn(pooledShapes_max),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

const auto test_ROIPooling_bilinear = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(coordShapes),
    ::testing::ValuesIn(pooledShapes_bilinear),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_BILINEAR),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest, test_ROIPooling_max, ROIPoolingLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsROIPooling_bilinear, ROIPoolingLayerTest, test_ROIPooling_bilinear, ROIPoolingLayerTest::getTestCaseName);

} // namespace
