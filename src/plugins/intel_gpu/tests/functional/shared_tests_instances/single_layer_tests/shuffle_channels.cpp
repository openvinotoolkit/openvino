// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::U8,
};

const std::vector<std::vector<size_t>> inputShapes = {
    {3, 4, 9, 5}, {2, 16, 24, 15}, {1, 32, 12, 25}
};

const std::vector<std::tuple<int, int>> shuffleParameters = {
    std::make_tuple(1, 2), std::make_tuple(-3, 2),
    std::make_tuple(2, 3), std::make_tuple(-2, 3),
    std::make_tuple(3, 5), std::make_tuple(-1, 5)
};

INSTANTIATE_TEST_SUITE_P(smoke_GPU_ShuffleChannels,
                         ShuffleChannelsLayerTest,
                         ::testing::Combine(::testing::ValuesIn(shuffleParameters),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapes),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ShuffleChannelsLayerTest::getTestCaseName);

// ND support tests
INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels3D,
                         ShuffleChannelsLayerTest,
                         ::testing::Combine(::testing::Values(std::tuple<int, int>(1, 3)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({18, 30, 36})),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels2D,
                         ShuffleChannelsLayerTest,
                         ::testing::Combine(::testing::Values(std::tuple<int, int>(1, 3)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({18, 30})),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ShuffleChannelsLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ShuffleChannels1D,
                         ShuffleChannelsLayerTest,
                         ::testing::Combine(::testing::Values(std::tuple<int, int>(0, 3)),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(std::vector<size_t>({30})),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         ShuffleChannelsLayerTest::getTestCaseName);
