// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
// Common params
const std::vector<InferenceEngine::Precision> inputPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        // TODO: Issue:27391
        // InferenceEngine::Precision::FP32,
        // TODO: Issue:28036
        // InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid, {}},
        {Tanh,    {}},
        {Relu,    {}},
        {Exp,     {}},
        {Log,     {}},
        {Sign,    {}},
        {Abs,     {}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
        {{1, 10 * 1024}, {{}}},
        {{8, 128}, {{}}},
        {{1, 4, 2, 256}, {{}}},
        {{4, 4, 4, 4}, {{}}},
        {{1, 16, 1, 128}, {{}}},
        {{1, 8, 15, 128}, {{}}},
        {{1, 4, 4, 128}, {{}}},
        {{8}, {{}}},
        {{5}, {{}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)
);


INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
