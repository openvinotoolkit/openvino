// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/comparison.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ComparisonLayerTest;

std::map<ov::Shape, std::vector<ov::Shape>> input_shapes_combinations = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

auto input_shapes_pair_vector = ov::test::utils::combineParams(input_shapes_combinations);

auto converter = [] (const std::vector<std::pair<ov::Shape, ov::Shape>>& shapes) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& shape : shapes) {
        result.push_back({shape.first, shape.second});
    }
    return result;
};

auto input_shapes_static = converter(input_shapes_pair_vector);

std::vector<ov::element::Type> model_type = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        ov::element::boolean,
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

const auto ComparisonTestParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
        ::testing::ValuesIn(comparisonOpTypes),
        ::testing::ValuesIn(secondInputTypes),
        ::testing::ValuesIn(model_type),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs, ComparisonLayerTest, ComparisonTestParams, ComparisonLayerTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> input_shapes_is_ops_static = {
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
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_is_ops_static)),
        ::testing::ValuesIn(comparisonOpTypesIs),
        ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_IsOp, ComparisonLayerTest, ComparisonTestParamsIs, ComparisonLayerTest::getTestCaseName);

}  // namespace
