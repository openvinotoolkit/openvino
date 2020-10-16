// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/activation_range.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
// Common params
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {Sigmoid, {}},
        {Tanh,    {}},
        {Relu,    {}},
        {Sign,    {}},
        {Abs,     {}}
};

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> basic = {
        {{1, 2 * 1024}, {{}}}
};

std::vector<std::pair<int32_t, uint32_t>> inputValuesRange = {
    {-1, 2},
    {-8, 16},
    {-16, 32},
    {-32, 64},
    {-64, 128},
};

const auto basicCases = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(activationTypes)),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::ValuesIn(CommonTestUtils::combineParams(basic)),
        ::testing::ValuesIn(inputValuesRange),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)
);


INSTANTIATE_TEST_CASE_P(Activation_Range, ActivationLayerRangeTest, basicCases, ActivationLayerRangeTest::getTestCaseName);

}  // namespace
