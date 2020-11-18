// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/roi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<uint64_t>> inShapes = {
    {1, 3, 8, 8},
    {3, 4, 50, 50}
};

const std::vector<std::vector<size_t>> pooledShapes = {
    {1, 1},
    {2, 2},
    {3, 3},
    {6, 6}
};

const std::vector<std::vector<uint64_t>> coordShapes = {
    {1, 5},
    {3, 5},
    {5, 5}
};

const std::vector<float> spatial_scales = {0.625f, 1.f};

const auto test_ROIPooling_max = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(coordShapes),
    ::testing::ValuesIn(pooledShapes),
    ::testing::ValuesIn(spatial_scales),
    ::testing::Values(ngraph::helpers::ROIPoolingTypes::ROI_MAX),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_TestsROIPooling_max, ROIPoolingLayerTest, test_ROIPooling_max, ROIPoolingLayerTest::getTestCaseName);
