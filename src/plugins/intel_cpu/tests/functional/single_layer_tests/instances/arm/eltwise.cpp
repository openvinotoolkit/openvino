// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/mvn.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace CPULayerTestsDefinitions {
namespace Eltwise {

std::vector<InferenceEngine::Precision> dataPrecisions = {
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::FP32
};

std::vector<InferenceEngine::Precision> idxPrecisions = {
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I64
};

const std::vector<float> epsilonF = {
        0.0001f
};

const std::vector<std::string> epsMode = {
        "inside_sqrt",
        "outside_sqrt"
};

INSTANTIATE_TEST_SUITE_P(smoke_power_mvn_decomposition_3D, Mvn6LayerTest,
                         ::testing::Combine(
                         ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 32, 17}, {1, 37, 9}}),
                         ::testing::ValuesIn(dataPrecisions),
                         ::testing::ValuesIn(idxPrecisions),
                         ::testing::ValuesIn(std::vector<std::vector<int>>{{0, 1, 2}, {0}, {1}}),
                         ::testing::ValuesIn({true}),
                         ::testing::ValuesIn(epsilonF),
                         ::testing::ValuesIn(epsMode),
                         ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_power_mvn_decomposition_4D, Mvn6LayerTest,
                         ::testing::Combine(
                                 ::testing::ValuesIn(std::vector<std::vector<size_t>>{{1, 16, 5, 8}, {2, 19, 5, 10}}),
                                 ::testing::ValuesIn(dataPrecisions),
                                 ::testing::ValuesIn(idxPrecisions),
                                 ::testing::ValuesIn(std::vector<std::vector<int>>{
                                         {0, 1, 2, 3}, {0, 1, 2},
                                         {0, 3}, {0}, {1}, {2}, {3}}),
                                 ::testing::ValuesIn({true}),
                                 ::testing::ValuesIn(epsilonF),
                                 ::testing::ValuesIn(epsMode),
                                 ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                         Mvn6LayerTest::getTestCaseName);

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions
