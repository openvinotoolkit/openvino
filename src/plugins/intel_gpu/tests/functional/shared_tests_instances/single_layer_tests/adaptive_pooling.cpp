// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <vector>

#include "single_layer_tests/adaptive_pooling.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ngraph::helpers;
using namespace LayerTestsDefinitions;
using namespace ngraph::element;

namespace {
const std::vector<std::string> poolingModes = {"max", "avg"};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> inputShapes1D = {
       {1, 3, 5},
       {1, 1, 17},
};
const std::vector<std::vector<int>> outputShapes1D = {
        {2},
        {5},
};

INSTANTIATE_TEST_SUITE_P(smoke_AdaptivePooling1D, AdaPoolLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes1D),
                                 ::testing::ValuesIn(outputShapes1D),
                                 ::testing::ValuesIn(poolingModes),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         AdaPoolLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes2D = {
        {1, 3, 4, 6},
        {1, 1, 17, 5},
};
const std::vector<std::vector<int>> outputShapes2D = {
        {2, 4},
        {4, 5},
};

INSTANTIATE_TEST_SUITE_P(smoke_AdaptivePooling2D, AdaPoolLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes2D),
                                 ::testing::ValuesIn(outputShapes2D),
                                 ::testing::ValuesIn(poolingModes),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         AdaPoolLayerTest::getTestCaseName);

const std::vector<std::vector<size_t>> inputShapes3D = {
        {1, 1, 3, 3, 3},
        {1, 3, 5, 7, 11},
};
const std::vector<std::vector<int>> outputShapes3D = {
        {2, 2, 2},
        {4, 5, 3},
};

INSTANTIATE_TEST_SUITE_P(smoke_AdaptivePooling3D, AdaPoolLayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(inputShapes3D),
                                 ::testing::ValuesIn(outputShapes3D),
                                 ::testing::ValuesIn(poolingModes),
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                         AdaPoolLayerTest::getTestCaseName);

}  // namespace
