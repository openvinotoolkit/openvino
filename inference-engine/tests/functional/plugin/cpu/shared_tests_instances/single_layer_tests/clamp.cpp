// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/clamp.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {
    {50},
    {10, 10},
    {1, 20, 20}
};

const std::vector<std::pair<float, float>> intervals = {
    {-20.1, -10.5},
    {-10.0, 10.0},
    {10.3, 20.4}
};

const std::vector<std::pair<float, float>> intervals_unsigned = {
    {0.1, 10.1},
    {10.0, 100.0},
    {10.6, 20.6}
};

const std::vector<InferenceEngine::Precision> netPrc = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I64,
    InferenceEngine::Precision::I32
};

const auto test_Clamp_signed = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(intervals),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto test_Clamp_unsigned = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(intervals_unsigned),
    ::testing::Values(InferenceEngine::Precision::U64),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_signed, ClampLayerTest, test_Clamp_signed, ClampLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_TestsClamp_unsigned, ClampLayerTest, test_Clamp_unsigned, ClampLayerTest::getTestCaseName);
