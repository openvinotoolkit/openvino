// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/gather_tree.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(GatherTreeLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I32
};

const std::vector<std::vector<size_t>> inputShapes = { {5, 1, 10}, {1, 1, 10}, {20, 1, 10}, {20, 20, 10} };

const std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER
};

INSTANTIATE_TEST_SUITE_P(smoke_GatherTree_Serialization, GatherTreeLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(inputShapes),
                            ::testing::ValuesIn(secondaryInputTypes),
                            ::testing::ValuesIn(netPrecisions),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(InferenceEngine::Layout::ANY),
                            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        GatherTreeLayerTest::getTestCaseName);
}   // namespace