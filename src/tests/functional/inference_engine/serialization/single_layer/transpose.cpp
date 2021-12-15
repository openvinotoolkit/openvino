// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/transpose.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(TransposeLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32
};

std::vector<std::vector<size_t>> inputShape2D = {{2, 10}, {10, 2}, {10, 10}};
std::vector<std::vector<size_t>> order2D      = {{}, {0, 1}, {1, 0}};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose2D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order2D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShape2D),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

std::vector<std::vector<size_t>> inputShape4D = {{2, 2, 2, 2}};
std::vector<std::vector<size_t>> order4D      = {
        {}, {0, 1, 2, 3}
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose4D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order4D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShape4D),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);

std::vector<std::vector<size_t>> inputShape5D = {{2, 3, 4, 5, 6}};
std::vector<std::vector<size_t>> order5D      = {
        {}, {0, 1, 2, 3, 4}
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose5D, TransposeLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(order5D),
                ::testing::ValuesIn(netPrecisions),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::Values(InferenceEngine::Layout::ANY),
                ::testing::ValuesIn(inputShape5D),
                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                TransposeLayerTest::getTestCaseName);
}  // namespace
