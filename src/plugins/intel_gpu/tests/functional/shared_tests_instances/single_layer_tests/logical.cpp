// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/logical.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::LogicalLayerTest;

std::vector<std::vector<ov::Shape>> combine_shapes(const std::map<ov::Shape, std::vector<ov::Shape>>& input_shapes_static) {
    std::vector<std::vector<ov::Shape>> result;
    for (const auto& input_shape : input_shapes_static) {
        for (auto& item : input_shape.second) {
            result.push_back({input_shape.first, item});
        }

        if (input_shape.second.empty()) {
            result.push_back({input_shape.first, {}});
        }
    }
    return result;
}

std::map<ov::Shape, std::vector<ov::Shape>> inputShapes = {
        {{1}, {{1}, {17}, {1, 1}, {2, 18}, {1, 1, 2}, {2, 2, 3}, {1, 1, 2, 3}}},
        {{5}, {{1}, {1, 1}, {2, 5}, {1, 1, 1}, {2, 2, 5}}},
        {{2, 200}, {{1}, {200}, {1, 200}, {2, 200}, {2, 2, 200}}},
        {{1, 3, 20}, {{20}, {2, 1, 1}}},
        {{2, 17, 3, 4}, {{4}, {1, 3, 4}, {2, 1, 3, 4}}},
        {{2, 1, 1, 3, 1}, {{1}, {1, 3, 4}, {2, 1, 3, 4}, {1, 1, 1, 1, 1}}},
};

std::map<ov::Shape, std::vector<ov::Shape>> inputShapesNot = {
        {{1}, {}},
        {{5}, {}},
        {{2, 200}, {}},
        {{1, 3, 20}, {}},
        {{2, 17, 3, 4}, {}},
        {{2, 1, 1, 3, 1}, {}},
};

std::vector<ov::test::utils::LogicalTypes> logicalOpTypes = {
        ov::test::utils::LogicalTypes::LOGICAL_AND,
        ov::test::utils::LogicalTypes::LOGICAL_OR,
        ov::test::utils::LogicalTypes::LOGICAL_XOR,
};

std::vector<ov::test::utils::InputLayerType> secondInputTypes = {
        ov::test::utils::InputLayerType::CONSTANT,
        ov::test::utils::InputLayerType::PARAMETER,
};

std::vector<ov::element::Type> netPrecisions = {
        ov::element::boolean,
};

std::map<std::string, std::string> additional_config = {};

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs,
                         LogicalLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(combine_shapes(inputShapes))),
                                            ::testing::ValuesIn(logicalOpTypes),
                                            ::testing::ValuesIn(secondInputTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         LogicalLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefsNot,
                         LogicalLayerTest,
                         ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(combine_shapes(inputShapesNot))),
                                            ::testing::Values(ov::test::utils::LogicalTypes::LOGICAL_NOT),
                                            ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU),
                                            ::testing::Values(additional_config)),
                         LogicalLayerTest::getTestCaseName);

}  // namespace
