// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/comparison.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;
using namespace InferenceEngine;

namespace {

std::map<std::vector<size_t>, std::vector<std::vector<size_t>>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
        ngraph::helpers::ComparisonTypes::EQUAL,
        ngraph::helpers::ComparisonTypes::NOT_EQUAL,
        ngraph::helpers::ComparisonTypes::GREATER,
        ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
        ngraph::helpers::ComparisonTypes::LESS,
        ngraph::helpers::ComparisonTypes::LESS_EQUAL,
};

std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ComparisonLayerTest, 
        ::testing::Combine(
                ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
                ::testing::ValuesIn(std::vector<Precision>{Precision::FP32, Precision::I32, Precision::I64}),
                ::testing::ValuesIn(comparisonOpTypes),
                ::testing::ValuesIn(secondInputTypes),
                ::testing::Values(Precision::UNSPECIFIED),
                ::testing::Values(Precision::UNSPECIFIED),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ComparisonLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_CompareWithRefs, ComparisonLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(CommonTestUtils::combineParams(inputShapes)),
                ::testing::ValuesIn(std::vector<Precision>{Precision::FP16, Precision::BOOL}),
                ::testing::ValuesIn(comparisonOpTypes),
                ::testing::ValuesIn(secondInputTypes),
                ::testing::Values(Precision::UNSPECIFIED),
                ::testing::Values(Precision::UNSPECIFIED),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config)),
        ComparisonLayerTest::getTestCaseName);


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

std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypesIs = {
        ngraph::helpers::ComparisonTypes::IS_FINITE,
        ngraph::helpers::ComparisonTypes::IS_NAN
};

const auto ComparisonTestParamsIs = ::testing::Combine(
        ::testing::ValuesIn(inputShapesIsOps),
        ::testing::Values(Precision::FP32),
        ::testing::ValuesIn(comparisonOpTypesIs),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(Precision::UNSPECIFIED),
        ::testing::Values(Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_IsOp, ComparisonLayerTest, ComparisonTestParamsIs, ComparisonLayerTest::getTestCaseName);

}  // namespace
