// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/tile.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

const std::vector<std::vector<int64_t>> repeats3D = {
        {1, 2, 3},
        {2, 1, 1},
        {2, 3, 1},
        {2, 2, 2},
        {1, 1, 1}
};

INSTANTIATE_TEST_SUITE_P(smoke_Tile, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats3D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({2, 3, 4})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

const std::vector<std::vector<int64_t>> repeats6D = {
        {1, 1, 1, 2, 1, 2},
        {1, 1, 1, 1, 1, 1}
};

INSTANTIATE_TEST_SUITE_P(smoke_Tile6d, TileLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(repeats6D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(std::vector<size_t>({1, 4, 3, 1, 3, 1})),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

}  // namespace
