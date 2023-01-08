// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/split.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

INSTANTIATE_TEST_SUITE_P(smoke_NumSplitsCheck, SplitLayerTest,
                        ::testing::Combine(
                                ::testing::Values(1, 2, 3, 5, 6, 10, 30),
                                ::testing::Values(0, 1, 2, 3),
                                ::testing::Values(InferenceEngine::Precision::FP32),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t>({30, 30, 30, 30})),
                                ::testing::Values(std::vector<size_t>({})),
                                ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                        SplitLayerTest::getTestCaseName);

}  // namespace
