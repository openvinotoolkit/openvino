// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/split.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(5),
                                // TODO: 0-axis excluded
                                //  Check (status == ie::StatusCode::OK) failed: Failed to reshape Network:
                                //  Failed to infer shapes for Split layer (Split_2) with error:
                                //  The sum of the dimensions on the axis(0) is not equal out_sizes: [30]
                                ::testing::Values(1, 2, 3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        SplitLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_splitWithUnusedOutputsTest, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(5),
                                // TODO: 0-axis excluded
                                //  Check (status == ie::StatusCode::OK) failed: Failed to reshape Network:
                                //  Failed to infer shapes for Split layer (Split_2) with error:
                                //  The sum of the dimensions on the axis(0) is not equal out_sizes: [30]
                                ::testing::Values(1, 2, 3),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                                ::testing::Values(std::vector<size_t>({0, 2}),
                                                  std::vector<size_t>({0, 4}),
                                                  std::vector<size_t>({2, 3})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        SplitLayerTest::getTestCaseName);
}  // namespace
