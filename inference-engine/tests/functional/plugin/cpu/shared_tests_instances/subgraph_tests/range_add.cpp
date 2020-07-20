// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/range_add.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<float> positiveStart = { 1.0f, 1.2f };
const std::vector<float> positiveStop = { 5.0f, 5.2f };
const std::vector<float> positiveStep = { 1.0f, 0.1f };

const std::vector<float> negativeStart = { 1.0f, 1.2f };
const std::vector<float> negativeStop = { -5.0f, -5.2f };
const std::vector<float> negativeStep = { -1.0f, -0.1f };

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

INSTANTIATE_TEST_CASE_P(BasicPositive, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(positiveStart),
                                ::testing::ValuesIn(positiveStop),
                                ::testing::ValuesIn(positiveStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BasicNegative, RangeAddSubgraphTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(negativeStart),
                                ::testing::ValuesIn(negativeStop),
                                ::testing::ValuesIn(negativeStep),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        RangeAddSubgraphTest::getTestCaseName);
}  // namespace
