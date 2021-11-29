// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/squeeze_unsqueeze.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(SqueezeUnsqueezeLayerTest, Serialize) {
    Serialize();
}

std::map<std::vector<size_t>, std::vector<std::vector<int>>> axesVectors = {
        {{1, 1, 1, 1}, {{}, {-1}, {0}, {1}, {2}, {3}, {0, 1}, {0, 2}, {0, 3}, {1, 2}, {2, 3}, {0, 1, 2}, {0, 2, 3}, {1, 2, 3}, {0, 1, 2, 3}}},
        {{1, 2, 3, 4}, {{}, {0}}},
        {{2, 1, 3, 4}, {{}, {1}}},
        {{1}, {{}, {-1}, {0}}},
        {{1, 2}, {{}, {0}}},
        {{2, 1}, {{}, {1}, {-1}}},
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32,
    InferenceEngine::Precision::U32};

const std::vector<ngraph::helpers::SqueezeOpType> opTypes = {
    ngraph::helpers::SqueezeOpType::SQUEEZE};

INSTANTIATE_TEST_CASE_P(smoke_Squeeze_Basic, SqueezeUnsqueezeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(axesVectors)),
                            ::testing::ValuesIn(opTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        SqueezeUnsqueezeLayerTest::getTestCaseName);
} // namespace
