// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/comparison.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace LayerTestsDefinitions::ComparisonParams;

namespace {

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
};

const std::vector<ngraph::helpers::ComparisonTypes> comparisonOpTypes = {
    ngraph::helpers::ComparisonTypes::EQUAL,
    ngraph::helpers::ComparisonTypes::NOT_EQUAL,
    ngraph::helpers::ComparisonTypes::GREATER,
    ngraph::helpers::ComparisonTypes::GREATER_EQUAL,
    ngraph::helpers::ComparisonTypes::LESS,
    ngraph::helpers::ComparisonTypes::LESS_EQUAL,
};

const std::vector<ngraph::helpers::InputLayerType> secondInputTypes = {
    ngraph::helpers::InputLayerType::CONSTANT,
    ngraph::helpers::InputLayerType::PARAMETER,
};

const std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs,
                         ComparisonLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::utils::combineParams(inputShapes)),
                                            ::testing::ValuesIn(inputsPrecisions),
                                            ::testing::ValuesIn(comparisonOpTypes),
                                            ::testing::ValuesIn(secondInputTypes),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         ComparisonLayerTest::getTestCaseName);

const std::vector<InputShapesTuple> inputShapesIsOps = {
    {{5}, {1}},
    {{2, 2}, {1}},
    {{2, 2, 2}, {1}},
    {{2, 17, 3, 4}, {1}},
    {{2, 17, 3, 4, 8}, {1}},
    {{2, 17, 3, 4, 8, 2}, {1}},
};

const std::vector<InferenceEngine::Precision> inputsPrecisionsIsOps = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16,
};

INSTANTIATE_TEST_SUITE_P(smoke_IsOp,
                         ComparisonLayerTest,
                         ::testing::Combine(::testing::ValuesIn(inputShapesIsOps),
                                            ::testing::ValuesIn(inputsPrecisionsIsOps),
                                            ::testing::Values(ngraph::helpers::ComparisonTypes::IS_FINITE,
                                                              ngraph::helpers::ComparisonTypes::IS_NAN),
                                            ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::BOOL),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         ComparisonLayerTest::getTestCaseName);

}  // namespace
