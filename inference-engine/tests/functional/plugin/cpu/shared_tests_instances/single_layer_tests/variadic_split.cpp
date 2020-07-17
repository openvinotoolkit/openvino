// Copyright (C) 2019 Intel Corporation
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

    const std::vector<std::vector<size_t>> numSplits = {
            {1, 32, 17},
            {1, 37, 9},
            {1, 16, 5, 8},
            {2, 19, 5, 10},
            {7, 32, 2, 8},
            {5, 8, 3, 5},
            {4, 41, 6, 9},
            {1, 32, 8, 1, 6},
            {1, 9, 1, 15, 9},
            {6, 64, 6, 1, 18},
            {2, 31, 2, 9, 1},
            {10, 16, 5, 10, 6}
    };
    INSTANTIATE_TEST_CASE_P(NumSplitsCheck, VariadicSplitLayerTest,
            ::testing::Combine(
            ::testing::ValuesIn(numSplits),
            ::testing::Values(0, 1, 2, 3),
            ::testing::ValuesIn(netPrecisions),
            ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)),
            VariadicSplitLayerTest::getTestCaseName);
}  // namespace
