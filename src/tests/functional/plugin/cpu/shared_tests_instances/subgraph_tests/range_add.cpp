// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/range_add.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<float> positiveStart = { 1.0f, 1.2f };
const std::vector<float> positiveStop = { 5.0f, 5.2f };
const std::vector<float> positiveStep = { 1.0f, 0.1f };

const std::vector<float> negativeStart = { 1.0f, 1.2f };
const std::vector<float> negativeStop = { -5.0f, -5.2f };
const std::vector<float> negativeStep = { -1.0f, -0.1f };

const std::vector<float> trunc_start = { 1.2f, 1.9f };
const std::vector<float> trunc_stop = { 11.4f, 11.8f };
const std::vector<float> trunc_step = { 1.3f, 2.8f };

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16        // "[NOT_IMPLEMENTED] Input image format FP16 is not supported yet...
};

// ------------------------------ V0 ------------------------------

INSTANTIATE_TEST_SUITE_P(smoke_BasicPositive, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(positiveStart),
                                ::testing::ValuesIn(positiveStop),
                                ::testing::ValuesIn(positiveStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicNegative, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(negativeStart),
                                ::testing::ValuesIn(negativeStop),
                                ::testing::ValuesIn(negativeStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);

// ------------------------------ V4 ------------------------------
INSTANTIATE_TEST_SUITE_P(smoke_BasicPositive, RangeNumpyAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(positiveStart),
                                ::testing::ValuesIn(positiveStop),
                                ::testing::ValuesIn(positiveStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeNumpyAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicNegative, RangeNumpyAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(negativeStart),
                                ::testing::ValuesIn(negativeStop),
                                ::testing::ValuesIn(negativeStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeNumpyAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_BasicTruncateInputs, RangeNumpyAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(trunc_start),
                                ::testing::ValuesIn(trunc_stop),
                                ::testing::ValuesIn(trunc_step),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::I32),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeNumpyAddSubgraphTest::getTestCaseName);
}  // namespace
