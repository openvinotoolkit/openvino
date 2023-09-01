// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/comparison.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::vector<InferenceEngine::Precision> inputsPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
        InferenceEngine::Precision::I32,
        InferenceEngine::Precision::BOOL,
};

std::vector<ov::helpers::ComparisonTypes> comparisonOpTypes = {
        ov::helpers::ComparisonTypes::EQUAL,
        ov::helpers::ComparisonTypes::NOT_EQUAL,
        ov::helpers::ComparisonTypes::GREATER,
        ov::helpers::ComparisonTypes::GREATER_EQUAL,
        ov::helpers::ComparisonTypes::LESS,
        ov::helpers::ComparisonTypes::LESS_EQUAL,
};

std::vector<ov::helpers::InputLayerType> secondInputTypes = {
        ov::helpers::InputLayerType::CONSTANT,
        ov::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

const auto ComparisonTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::utils::combineParams(inputShapes)),
        ::testing::ValuesIn(inputsPrecisions),
        ::testing::ValuesIn(comparisonOpTypes),
        ::testing::ValuesIn(secondInputTypes),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ComparisonLayerTest, ComparisonTestParams, ComparisonLayerTest::getTestCaseName);


std::vector<InputShapesTuple> inputShapesIsOps = {
        {{1}, {1}},
        {{1, 2}, {1}},
        {{3, 1}, {1}},
        {{2, 2}, {1}},
        {{1, 5, 1}, {1}},
        {{2, 1, 1, 3, 1}, {1}},
        {{7, 1, 1, 1, 1}, {1}},
        {{2, 2, 2}, {1}},
        {{3, 1, 3, 3}, {1}},
        {{17}, {1}},
        {{2, 18}, {1}},
        {{1, 3, 20}, {1}},
        {{2, 200}, {1}},
        {{2, 17, 3, 4}, {1}}
};

std::vector<ov::helpers::ComparisonTypes> comparisonOpTypesIs = {
        ov::helpers::ComparisonTypes::IS_FINITE,
        ov::helpers::ComparisonTypes::IS_NAN
};

const auto ComparisonTestParamsIs = ::testing::Combine(
        ::testing::ValuesIn(inputShapesIsOps),
        ::testing::Values(InferenceEngine::Precision::FP32),
        ::testing::ValuesIn(comparisonOpTypesIs),
        ::testing::Values(ov::helpers::InputLayerType::CONSTANT),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_IsOp, ComparisonLayerTest, ComparisonTestParamsIs, ComparisonLayerTest::getTestCaseName);

}  // namespace
