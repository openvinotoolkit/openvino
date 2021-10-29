// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "shared_test_classes/single_layer/activation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace ngraph::helpers;
namespace {
    TEST_P(ActivationLayerTest, Serialize) {
        Serialize();
    }

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::map<ActivationTypes, std::vector<std::vector<float>>> activationTypes = {
        {ActivationTypes::PReLu,     {{-0.01f}}},
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
        ::testing::Values(CommonTestUtils::DEVICE_CPU)
);

INSTANTIATE_TEST_SUITE_P(smoke_ActivationPreluSerialization,
                        ActivationLayerTest, basicCases, ActivationLayerTest::getTestCaseName);
}  // namespace
