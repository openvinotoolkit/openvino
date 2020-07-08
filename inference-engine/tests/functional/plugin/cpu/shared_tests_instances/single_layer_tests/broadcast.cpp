// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/broadcast.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::I8,
        InferenceEngine::Precision::U8,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::FP32
};

INSTANTIATE_TEST_CASE_P(BroadcastNUMPY, BroadcastLayerTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::Values(ngraph::op::BroadcastType::NUMPY),
                                        ::testing::Values(std::vector<size_t>{1, 16, 50, 50}),
                                        ::testing::Values(std::vector<size_t>{})),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({16, 1, 1})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BroadcastEXPLICIT, BroadcastLayerTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::Values(ngraph::op::BroadcastType::EXPLICIT),
                                        ::testing::Values(std::vector<size_t>{1, 16, 50, 50}),
                                        ::testing::Values(std::vector<size_t>{1})),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({16})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        BroadcastLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(BroadcastBIDIRECTIONAL, BroadcastLayerTest,
                        ::testing::Combine(
                                ::testing::Combine(
                                        ::testing::Values(ngraph::op::BroadcastType::BIDIRECTIONAL),
                                        ::testing::Values(std::vector<size_t>{1, 1, 3, 3}),
                                        ::testing::Values(std::vector<size_t>{})),
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(std::vector<size_t >({4, 1, 1})),
                                ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                        BroadcastLayerTest::getTestCaseName);

}  // namespace
