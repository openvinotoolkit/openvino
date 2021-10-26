// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"

#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

typedef std::map<std::string, std::string> Config;

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{5}, {{1}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::vector<InferenceEngine::Precision> fpTypes = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

std::vector<InferenceEngine::Precision> intTypes = {
        InferenceEngine::Precision::I32,
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypesFpToFp = {
        ngraph::helpers::ComparisonTypes::EQUAL,
        ngraph::helpers::ComparisonTypes::NOT_EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER,
        ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
        ngraph::helpers::ComparisonTypes::LESS,
        ngraph::helpers::ComparisonTypes::LESS_EQUAL,
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypesFpToInt = {
        ngraph::helpers::ComparisonTypes::EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER,
        ngraph::helpers::ComparisonTypes::LESS,
};

INSTANTIATE_TEST_SUITE_P(smoke_ComparisonFp,
                        ComparisonLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
                            ::testing::ValuesIn(fpTypes),
                            ::testing::ValuesIn(comparisonOpTypesFpToFp),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ComparisonInt,
                        ComparisonLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
                                ::testing::ValuesIn(fpTypes),
                                ::testing::ValuesIn(comparisonOpTypesFpToInt),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::I32),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        ComparisonLayerTest::getTestCaseName);

}  // namespace
