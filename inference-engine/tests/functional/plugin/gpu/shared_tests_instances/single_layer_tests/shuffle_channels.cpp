// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/shuffle_channels.hpp"

using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> inputPrecision = {
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

const auto testCases = ::testing::Combine(::testing::ValuesIn(shuffleParameters),
                                          ::testing::ValuesIn(inputPrecision),
                                          ::testing::ValuesIn(inputShapes),
                                          ::testing::Values(CommonTestUtils::DEVICE_GPU));


INSTANTIATE_TEST_CASE_P(smoke_GPU_ShuffleChannels, ShuffleChannelsLayerTest, testCases, ShuffleChannelsLayerTest::getTestCaseName);
