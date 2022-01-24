// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/batch_to_space.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(BatchToSpaceLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

const std::vector<std::vector<int64_t>> block_shapes_4D = {
    {1, 1, 2, 2},
    {1, 1, 4, 2}
};

const std::vector<std::vector<int64_t>> crops_4D = {
    {0, 0, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
};

const auto batch_to_space_4D_params = ::testing::Combine(
        ::testing::ValuesIn(block_shapes_4D),
        ::testing::ValuesIn(crops_4D),
        ::testing::ValuesIn(crops_4D),
        ::testing::Values(std::vector<size_t>{16, 1, 2, 2}),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

const std::vector<std::vector<int64_t>> block_shapes_5D = {
    {1, 1, 2, 1, 3},
    {1, 1, 4, 2, 2}
};

const std::vector<std::vector<int64_t>> crops_5D = {
    {0, 0, 0, 0, 0},
    {0, 0, 1, 0, 1},
    {0, 0, 0, 1, 1}
};

const auto batch_to_space_5D_params = ::testing::Combine(
        ::testing::ValuesIn(block_shapes_5D),
        ::testing::ValuesIn(crops_5D),
        ::testing::ValuesIn(crops_5D),
        ::testing::Values(std::vector<size_t>{48, 1, 3, 4, 2}),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(
    smoke_BatchToSpace_Serialization_4D, BatchToSpaceLayerTest,
    batch_to_space_4D_params,
    BatchToSpaceLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BatchToSpace_Serialization_5D, BatchToSpaceLayerTest,
    batch_to_space_5D_params,
    BatchToSpaceLayerTest::getTestCaseName);
} // namespace
