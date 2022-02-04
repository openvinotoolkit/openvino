// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/tile.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::BF16,
        InferenceEngine::Precision::FP32
};

const std::vector<InferenceEngine::Precision> netTPrecisions = {
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::FP16,
};

const std::vector<std::vector<size_t>> inputShapes = {
        {2, 3, 4},
        {1, 1, 1},
};

const std::vector<std::vector<int64_t>> repeats3D = {
        {1, 2, 3},
        {1, 1, 2, 3},
        {1, 2, 1, 3},
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
                ::testing::ValuesIn(inputShapes),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
        TileLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_PrecTransformation, TileLayerTest,
        ::testing::Combine(
                ::testing::Values(repeats3D[0]),
                ::testing::ValuesIn(netTPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(inputShapes[0]),
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
