// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/squeeze.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
std::map<InputShape, std::vector<SqueezeAxes>> axesVectors = {
        {{1, 1, 1, 1}, {{-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
        {{1, 2, 3, 4}, {{0}}},
        {{2, 1, 3, 4}, {{1}}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(Basic, SqueezeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(SqueezeLayerTest::combineShapes(axesVectors)),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                                ::testing::ValuesIn(std::vector<bool>{false, true})),
                        SqueezeLayerTest::getTestCaseName);
}  // namespace
