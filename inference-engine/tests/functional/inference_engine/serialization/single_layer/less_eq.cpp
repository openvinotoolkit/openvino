// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "shared_test_classes/single_layer/comparison.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {
    TEST_P(ComparisonLayerTest, Serialize) {
        Serialize();
    }

const std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
    {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

const std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::U32,
        InferenceEngine::Precision::BOOL,
};

const std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

const std::map<std::string, std::string> additional_config = {};

const auto opType = ngraph::helpers::ComparisonTypes::LESS_EQUAL;

const auto ComparisonTestParams = ::testing::Combine(
        ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
        ::testing::ValuesIn(inputsPrecisions),
        ::testing::Values(opType),
        ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_LessEqual, ComparisonLayerTest, ComparisonTestParams, ComparisonLayerTest::getTestCaseName);
}  // namespace
