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

const std::vector<int> axes = {-4, -3, -2, -1, 0, 1, 2, 3};
const std::vector<int> groups = {1, 2, 3, 6};

const auto shuffleChannelsParams4D = ::testing::Combine(
        ::testing::ValuesIn(axes),
        ::testing::ValuesIn(groups)
);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels4D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                shuffleChannelsParams4D,
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({12, 18, 30, 36})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

// ND support tests
INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels6D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(2, 3)),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({24, 6, 12, 18, 30, 36})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels5D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(2, 3)),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({6, 12, 18, 30, 36})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels3D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(1, 3)),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({18, 30, 36})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels2D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(1, 3)),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({18, 30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels1D, ShuffleChannelsLayerTest,
        ::testing::Combine(
                ::testing::Values(std::tuple<int, int>(0, 3)),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t >({30})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        ShuffleChannelsLayerTest::getTestCaseName);

}  // namespace
