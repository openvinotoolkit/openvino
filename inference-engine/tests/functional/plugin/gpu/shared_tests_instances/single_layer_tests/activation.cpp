// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {


const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid,     {}},
        {Tanh,        {}},
        {Relu,        {}},
        {Exp,         {}},
        {Log,         {}},
        {Sign,        {}},
        {Abs,         {}},
        {Gelu,        {}},
        {Clamp,       {{-2.0f, 2.0f}}},
        {Negative,    {}},
        {Acos,        {}},
        {Asin,        {}},
        {Atan,        {}},
        {Cos,         {}},
        {Cosh,        {}},
        {Floor,       {}},
        {Sin,         {}},
        {Sinh,        {}},
        {Sqrt,        {}},
        {Tan,         {}},
        {Elu,         {{0.1f}}},
        {Erf,         {}},
        {HardSigmoid, {{0.2f, 0.5f}}},
        {Selu,        {{1.6732f, 1.0507f}}},
        {Ceiling,     {}},
        {Mish,        {}},
        {HSwish,      {}},
        {SoftPlus,    {}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 50}, {{}}},
        {{1, 128}, {{}}},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(smoke_Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
