// Copyright (C) 2020 Intel Corporation
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
        InferenceEngine::Precision::FP32
        // TODO: Fix Issue-27390
        // InferenceEngine::Precision::I16,
        // InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<ActivationTypes> activationTypes = {
        Sigmoid,
        Tanh,
        Relu,
        Exp,
        Log,
        Sign,
        Abs,
        Clamp,
        Negative,
        Acos,
        Asin,
        Atan,
        Cos,
        Cosh,
        Floor,
        Sin,
        Sinh,
        Sqrt,
        Tan,
        Elu,
        Erf,
        HardSigmoid,
        Selu,
        Ceiling,
};

const std::vector<ActivationTypes> activationParamTypes = {
        PReLu,
        LeakyRelu,
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(activationTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(CommonTestUtils::combineShapes<size_t>(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::ValuesIn(activationParamTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::ValuesIn(CommonTestUtils::combineShapes<size_t>(preluBasic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);


INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_CASE_P(Activation_Basic_Prelu, ActivationLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationParamLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);

}  // namespace
