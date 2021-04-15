// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/exp.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(ExpLayerTest, Serialize) {
    Serialize();
}

const std::vector<std::vector<size_t>> inShapes = {
    {50}, {10, 10}, {1, 20, 20}, {2, 3, 50, 50},
    {331, 13, 5}, {21, 11}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32};

INSTANTIATE_TEST_CASE_P(
    smoke_Exp_Serialization, ExpLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ExpLayerTest::getTestCaseName);
} // namespace