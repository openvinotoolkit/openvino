// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/transpose.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I64,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
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

std::vector<std::vector<size_t>> inputShape4D = {{2, 2, 2, 2}, {1, 10, 2, 3}, {2, 3, 4, 5}};
std::vector<std::vector<size_t>> order4D      = {
        {}, {0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1},
        {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0},
        {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0},
        {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}
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

std::vector<std::vector<size_t>> inputShape5D = {{2, 2, 2, 2, 2}, {1, 10, 2, 3, 4}, {2, 3, 4, 5, 6}};
std::vector<std::vector<size_t>> order5D      = {
        {}, {0, 1, 2, 3, 4}, {1, 0, 2, 3, 4}, {4, 3, 2, 1, 0}, {0, 2, 3, 4, 1},
        {1, 4, 2, 3, 0}, {2, 4, 1, 0, 3}, {3, 0, 2, 1, 4}, {4, 1, 0, 3, 2}, {0, 4, 1, 2, 3},
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

std::vector<std::vector<size_t>> inputShape6D = {{2, 2, 2, 2, 2, 2}, {1, 10, 2, 3, 4, 5}, {2, 3, 4, 5, 6, 7}};
std::vector<std::vector<size_t>> order6D      = {
        {}, {0, 1, 2, 3, 4, 5}, {1, 0, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {0, 2, 3, 4, 5, 1},
        {1, 5, 4, 2, 3, 0}, {2, 5, 4, 1, 0, 3}, {3, 0, 2, 1, 4, 5}, {5, 1, 0, 4, 3, 2}, {0, 5, 1, 2, 3, 4},
};

INSTANTIATE_TEST_SUITE_P(smoke_Transpose6D, TransposeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(order6D),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShape6D),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        TransposeLayerTest::getTestCaseName);

}  // namespace
