// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/exp.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

const std::vector<std::vector<size_t>> inShapes = {
    {50},
    {10, 10},
    {1, 20, 20},
    {14, 13, 212, 81}
};


const std::vector<InferenceEngine::Precision> netPrc = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};


const auto expParams = ::testing::Combine(
    ::testing::ValuesIn(inShapes),
    ::testing::ValuesIn(netPrc),
    ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Exp, ExpLayerTest, expParams, ExpLayerTest::getTestCaseName);