// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/psroi_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

std::vector<float> spatialScales = {1, 0.625};

const auto PSROICases_average = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 8, 16, 16}),
    ::testing::Values(std::vector<size_t>{10, 5}),
    ::testing::Values(2),
    ::testing::Values(2),
    ::testing::ValuesIn(spatialScales),
    ::testing::Values(1),
    ::testing::Values(1),
    ::testing::Values("average"),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsPSROIPooling_average, PSROIPoolingLayerTest, PSROICases_average, PSROIPoolingLayerTest::getTestCaseName);


const auto PSROICases_bilinear = ::testing::Combine(
    ::testing::Values(std::vector<size_t>{3, 32, 20, 20}),
    ::testing::Values(std::vector<size_t>{10, 5}),
    ::testing::Values(4),
    ::testing::Values(3),
    ::testing::ValuesIn(spatialScales),
    ::testing::Values(4),
    ::testing::Values(2),
    ::testing::Values("bilinear"),
    ::testing::Values(InferenceEngine::Precision::FP32),
    ::testing::Values(ov::test::utils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsPSROIPooling_bilinear, PSROIPoolingLayerTest, PSROICases_bilinear, PSROIPoolingLayerTest::getTestCaseName);
