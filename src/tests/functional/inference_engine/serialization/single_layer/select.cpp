// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/select.hpp"

#include <vector>
using namespace LayerTestsDefinitions;

const std::vector<InferenceEngine::Precision> inputPrecision = {
    InferenceEngine::Precision::I8, InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::FP32};

const std::vector<std::vector<std::vector<size_t>>> noneShapes = {
    {{2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}, {2, 3, 4, 5, 6}}};

const auto noneCases = ::testing::Combine(
    ::testing::ValuesIn(noneShapes), ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastType::NONE),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

const std::vector<std::vector<std::vector<size_t>>> numpyShapes = {
    {{5, 1, 2, 1}, {8, 1, 9, 1, 1}, {5, 1, 2, 1}}};

const auto numpyCases = ::testing::Combine(
    ::testing::ValuesIn(numpyShapes), ::testing::ValuesIn(inputPrecision),
    ::testing::Values(ngraph::op::AutoBroadcastType::NUMPY),
    ::testing::Values(CommonTestUtils::DEVICE_CPU));

TEST_P(SelectLayerTest, Serialize) {
    Serialize();
}

INSTANTIATE_TEST_SUITE_P(smoke_Serialization_SelectLayerTest_none,
                         SelectLayerTest, noneCases,
                         SelectLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Serialization_SelectLayerTest_numpy,
                         SelectLayerTest, numpyCases,
                         SelectLayerTest::getTestCaseName);
