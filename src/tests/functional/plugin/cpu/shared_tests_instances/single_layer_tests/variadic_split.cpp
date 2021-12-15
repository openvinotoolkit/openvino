// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/variadic_split.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

    const std::vector<InferenceEngine::Precision> netPrecisions = {
            InferenceEngine::Precision::FP32,
            InferenceEngine::Precision::FP16
    };

    // Sum of elements numSplits = inputShapes[Axis]
    const std::vector<std::vector<size_t>> numSplits = {
            {1, 16, 5, 8},
            {2, 19, 5, 4},
            {7, 13, 2, 8},
            {5, 8, 12, 5},
            {4, 11, 6, 9}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck, VariadicSplitLayerTest,
            ::testing::Combine(
            ::testing::ValuesIn(numSplits),
            ::testing::Values(0, 1, 2, 3),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            VariadicSplitLayerTest::getTestCaseName);
}  // namespace
