// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/mvn.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<std::vector<int>> indices_4D = {
        {2, 3},     // equivalent MVN-1 across_channel=0
        {1, 2, 3}   // equivalent MVN-1 across_channel=1
};

const std::vector<std::vector<int>> indices_3D = {
        {2},
        {0, 2},
        {1, 2},     // equivalent MVN-1 across_channel=0
        {0, 1, 2}   // equivalent MVN-1 across_channel=1
};

const std::vector<InferenceEngine::SizeVector> input_shape_4D = {
        {3, 3, 51, 89},
        {1, 3, 256, 384},
        {1, 10, 5, 17},
        {1, 3, 8, 9}
};

const std::vector<InferenceEngine::SizeVector> input_shape_3D = {
        {1, 32, 17},
        {1, 37, 9}
};

const std::vector<float> eps = {
        1.0e-10, 1.0e-8, 1.0e-7, 1.0e-5, 1.0e-3
};

INSTANTIATE_TEST_SUITE_P(smoke_MVN_4D, Mvn6LayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shape_4D),
                                ::testing::Values(InferenceEngine::Precision::FP16),
                                ::testing::Values(InferenceEngine::Precision::I32),
                                ::testing::ValuesIn(indices_4D),
                                ::testing::Values(false, true),
                                ::testing::ValuesIn(eps),
                                ::testing::Values("outside_sqrt"),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        Mvn6LayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_MVN_3D, Mvn6LayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(input_shape_3D),
                                ::testing::Values(InferenceEngine::Precision::FP16),
                                ::testing::Values(InferenceEngine::Precision::I32),
                                ::testing::ValuesIn(indices_3D),
                                ::testing::Values(false, true),
                                ::testing::ValuesIn(eps),
                                ::testing::Values("outside_sqrt"),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        Mvn6LayerTest::getTestCaseName);

}  // namespace
