// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::U16,
        InferenceEngine::Precision::FP32
};

const std::vector<int> axes = {0, 1, 2, 3};
const std::vector<int> negativeAxes = {-4, -3, -2, -1};
const std::vector<int> groups = {2, 3, 10};

const auto shuffleChannelsParams4D = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(groups)
);

const auto shuffleChannelsParamsNegativeAxis4D = ::testing::Combine(
        ::testing::ValuesIn(negativeAxes),
        ::testing::ValuesIn(groups)
);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                shuffleChannelsParams4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({30, 30, 30, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_ShuffleChannelsNegativeAxis4D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                shuffleChannelsParamsNegativeAxis4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({30, 30, 30, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

}  // namespace
