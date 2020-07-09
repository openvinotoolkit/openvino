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

const std::vector<ActivationTypes> activationTypes = {
        Sigmoid,
        Tanh,
        Relu,
        Exp,
        Log,
        Sign,
        Abs,
        Gelu,
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
        Selu
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(activationTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(std::vector<size_t>({1, 50}), std::vector<size_t>({1, 128})),
        ::testing::Values(CommonTestUtils::DEVICE_GPU)
);

INSTANTIATE_TEST_CASE_P(Activation_Basic, ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);

}  // namespace
