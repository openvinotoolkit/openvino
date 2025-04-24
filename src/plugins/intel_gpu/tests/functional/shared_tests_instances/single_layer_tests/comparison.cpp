// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

const std::vector<ov::test::utils::ComparisonTypes> comparisonOpTypes = {
    ov::test::utils::ComparisonTypes::EQUAL,
    ov::test::utils::ComparisonTypes::NOT_EQUAL,
    ov::test::utils::ComparisonTypes::GREATER,
    ov::test::utils::ComparisonTypes::GREATER_EQUAL,
    ov::test::utils::ComparisonTypes::LESS,
    ov::test::utils::ComparisonTypes::LESS_EQUAL,
};

const std::vector<ov::test::utils::InputLayerType> secondInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER,
};

const std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs,
                         ComparisonLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                            ::testing::ValuesIn(comparisonOpTypes),
                                            ::testing::ValuesIn(secondInputTypes),
                                            ::testing::ValuesIn(model_type),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         ComparisonLayerTest::getTestCaseName);

const std::vector<std::vector<ov::Shape>> inputShapesIsOps = {
    {{5}, {1}},
    {{2, 2}, {1}},
    {{2, 2, 2}, {1}},
    {{2, 17, 3, 4}, {1}},
    {{2, 17, 3, 4, 8}, {1}},
    {{2, 17, 3, 4, 8, 2}, {1}},
};

std::vector<ov::test::utils::ComparisonTypes> comparisonOpTypesIs = {
        ov::test::utils::ComparisonTypes::IS_FINITE,
        ov::test::utils::ComparisonTypes::IS_NAN
};

INSTANTIATE_TEST_SUITE_P(smoke_IsOp,
                         ComparisonLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapesIsOps)),
                                            ::testing::ValuesIn(comparisonOpTypesIs),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::Values(ov::element::f32),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         ComparisonLayerTest::getTestCaseName);

}  // namespace
