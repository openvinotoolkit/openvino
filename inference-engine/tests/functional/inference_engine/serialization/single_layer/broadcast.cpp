// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/broadcast.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(BroadcastLayerTest, Serialize) { Serialize(); }
const std::vector<InferenceEngine::Precision> inputPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32,
    InferenceEngine::Precision::BOOL};

// NUMPY MODE

std::vector<std::vector<size_t>> inShapesNumpy = {{3, 1}, {1, 4, 1}};

std::vector<std::vector<size_t>> targetShapesNumpy = {{2, 3, 6}, {1, 4, 4}};

const auto numpyBroadcastParams1 = ::testing::Combine(
    ::testing::Values(targetShapesNumpy[0]),
    ::testing::Values(ngraph::AxisSet{}), // not used in numpy mode
    ::testing::Values(ngraph::op::BroadcastType::NUMPY),
    ::testing::Values(inShapesNumpy[0]), ::testing::ValuesIn(inputPrecisions),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_Broadcast1Serialization, BroadcastLayerTest,
                        numpyBroadcastParams1,
                        BroadcastLayerTest::getTestCaseName);
} // namespace
