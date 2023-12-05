// Copyright (C) 2018-2023 Intel Corporation
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
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I16,
        InferenceEngine::Precision::U8
};

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,               {}},
        {Tanh,                  {}},
        {Relu,                  {}},
        {Exp,                   {}},
        {Log,                   {}},
        {Sign,                  {}},
        {Abs,                   {}},
        {Gelu,                  {}},
        {Clamp,                 {{-2.0f, 2.0f}}},
        {Negative,              {}},
        {Acos,                  {}},
        {Acosh,                 {}},
        {Asin,                  {}},
        {Asinh,                  {}},
        {Atan,                  {}},
        {Atanh,                  {}},
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
        {Swish,                 {{0.5f}}},
        {RoundHalfToEven,       {}},
        {RoundHalfAwayFromZero, {}},
        {GeluErf,               {}},
        {GeluTanh,              {}},
        {SoftSign,              {}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> big_rank_activation_types = {
        {Relu,                  {}},
        {Exp,                   {}},
        {Log,                   {}},
        {Abs,                   {}},
        {Clamp,                 {{-2.0f, 2.0f}}},
        {Ceiling,               {}},
        {Swish,                 {{0.5f}}},
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationParamTypes = {
    {PReLu, {{-0.01f}}},
    {LeakyRelu, {{0.01f}}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> big_ranks = {
        {{1, 2, 3, 4, 5, 3}, {{}}},
        {{1, 2, 3, 4, 1, 3, 2}, {{}}},
        {{1, 2, 3, 4, 3, 2, 1, 2}, {{}}},
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> preluBasic = {
        {{1, 10, 20}, {{10}, {20}, {10, 20}}},
        {{1, 128}, {{1}, {128}}},
};

const auto basicCases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(basic)),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto basicPreluCases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(activationParamTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(preluBasic)),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

const auto big_rank_cases = []() {
    return ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(big_rank_activation_types)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(ov::test::utils::combineParams(big_ranks)),
        ::testing::Values(ov::test::utils::DEVICE_GPU));
};

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Activation_BigRanks, ActivationLayerTest, big_rank_cases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic_Prelu, ActivationLayerTest, basicPreluCases(), ActivationLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Basic, ActivationParamLayerTest, basicPreluCases(), ActivationLayerTest::getTestCaseName);

}  // namespace
