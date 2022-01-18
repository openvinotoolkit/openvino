// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/clamp.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(ClampLayerTest, Serialize) {
    Serialize();
}

const std::vector<std::vector<size_t>> inShapes = {
    {50}, {10, 10}, {1, 20, 20}, {2, 3, 50, 50}};

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::I32};

const std::vector<std::pair<float, float>> intervals = {
    {-20.1, -10.5}, {-10.0, 10.0}, {10.3, 20.4}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Clamp_Serialization, ClampLayerTest,
    ::testing::Combine(
        ::testing::ValuesIn(inShapes),
        ::testing::ValuesIn(intervals),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ClampLayerTest::getTestCaseName);

}  // namespace
