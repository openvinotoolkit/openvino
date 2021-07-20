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
        InferenceEngine::Precision::FP32
        // TODO: Fix Issue-27390
        // InferenceEngine::Precision::I16,
        // InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<InferenceEngine::Precision> intPrecisions = {
        InferenceEngine::Precision::I32,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,               {}},
        {Tanh,                  {}},
        {Relu,                  {}},
        {Exp,                   {}},
        {Log,                   {}},
        {Sign,                  {}},
        {Abs,                   {}},
        {Clamp,                 {{-2.0f, 2.0f}}},
        {Negative,              {}},
        {Acos,                  {}},
        {Asin,                  {}},
        {Asinh,                 {}},
        {Atan,                  {}},
        {Cos,                   {}},
        {Cosh,                  {}},
        {Floor,                 {}},
        {Sin,                   {}},
        {Sinh,                  {}},
        {Sqrt,                  {}},
        {Tan,                   {}},
        {Elu,                   {{0.1f}}},
        {Erf,                   {}},
        {HardSigmoid,           {{0.2f, 0.5f}}},
        {Selu,                  {{1.6732f, 1.0507f}}},
        {Ceiling,               {}},
        {Mish,                  {}},
        {HSwish,                {}},
        {SoftPlus,              {}},
        {HSigmoid,              {}},
        {RoundHalfToEven,       {}},
        {RoundHalfAwayFromZero, {}},
        {Erf,                   {}},
        {GeluErf,               {}},
        {GeluTanh,              {}},
        {Swish,                 {{0.4f}}}
};

// List of operations that should be tested also with integer precision
const std::map<ActivationTypes, std::vector<std::vector<float>>> intActivationTypes = {
        {Asinh,                 {}},
        {Atan,                  {}},
        {Negative,              {}},
        {Ceiling,               {}},
        {Cos,                   {}},
        {Sign,                  {}},
        {Sinh,                  {}},
        {Sqrt,                  {}},
        {Tanh,                  {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
        {PReLu, {{}}}, // Slope will be filled with increasing values from -10 to match slope input shape
        {LeakyRelu, {{0.01f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 50}, {{1}, {50}}},
        {{1, 128}, {{1}, {128}}},

        // Broadcast check
        {{3, 2}, {{1}, {2}, {3, 2}}},
        {{3, 2, 5}, {{1}, {2}, {5}, {2, 5}, {3, 1, 5}, {1, 2, 1}, {1, 1, 5}, {3, 1, 1}, {3, 2, 5}}},
        {{2, 1, 2}, {{2}, {2, 1, 1}}},
        {{3, 2, 5, 7}, {{1}, {7}, {2}, {5, 7}, {2, 5, 7}, {2, 1, 1}, {1, 2, 1, 1}, {3, 2, 1, 1}, {3, 2, 5, 7}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto basicPreluCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(preluBasic)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

const auto basicIntegerOperations = ::testing::Combine(
            ::testing::ValuesIn(CommonTestUtils::combineParams(intActivationTypes)),
            ::testing::ValuesIn(intPrecisions),
            ::testing::ValuesIn(intPrecisions),
            ::testing::ValuesIn(intPrecisions),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::Values(InferenceEngine::Layout::ANY),
            ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
            ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationDynamicLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Integer_Activation_Basic, ActivationLayerTest, basicIntegerOperations, ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Const, ActivationLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu_Param, ActivationParamLayerTest, basicPreluCases, ActivationLayerTest::getTestCaseName);
}  // namespace
