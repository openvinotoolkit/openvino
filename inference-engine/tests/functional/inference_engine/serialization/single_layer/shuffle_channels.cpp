// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/shuffle_channels.hpp"

using namespace LayerTestsDefinitions;

namespace {
    TEST_P(ShuffleChannelsLayerTest, Serialize) {
        Serialize();
    }

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32
};

const std::vector<int> axes = {0, 1, 2, 3};
const std::vector<int> negativeAxes = {-4, -3, -2, -1};
const std::vector<int> groups = {1, 2, 3};

const auto shuffleChannelsParams4D = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(groups)
);

const auto shuffleChannelsParamsNegativeAxis4D = ::testing::Combine(
        ::testing::ValuesIn(negativeAxes),
        ::testing::ValuesIn(groups)
);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannelsSerialization, ShuffleChannelsLayerTest,
        ::testing::Combine(
                shuffleChannelsParams4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({6, 6, 6, 6})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);
}  // namespace
