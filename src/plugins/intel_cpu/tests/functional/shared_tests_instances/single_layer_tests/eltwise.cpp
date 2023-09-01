// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {
std::vector<std::vector<ov::Shape>> inShapesStatic = {
        {{2}},
        {{2, 200}},
        {{10, 200}},
        {{1, 10, 100}},
        {{4, 4, 16}},
        {{1, 1, 1, 3}},
        {{2, 17, 5, 4}, {1, 17, 1, 1}},
        {{2, 17, 5, 1}, {1, 17, 1, 4}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{16, 16, 16, 16, 16}},
        {{16, 16, 16, 16, 1}},
        {{16, 16, 16, 1, 16}},
        {{16, 32, 1, 1, 1}},
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<ov::Shape>> inShapesStaticCheckCollapse = {
        {{16, 16, 16, 16}, {16, 16, 16, 1}},
        {{16, 16, 16, 1}, {16, 16, 16, 1}},
        {{16, 16, 16, 16}, {16, 16, 1, 16}},
        {{16, 16, 1, 16}, {16, 16, 1, 16}},
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}},
         {{ngraph::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamicLargeUpperBound = {
        {{{ngraph::Dimension(1, 1000000000000), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

std::vector<ov::helpers::InputLayerType> secondaryInputTypes = {
        ov::helpers::InputLayerType::CONSTANT,
        ov::helpers::InputLayerType::PARAMETER,
};

std::vector<ov::helpers::InputLayerType> secondaryInputTypesDynamic = {
        ov::helpers::InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::SCALAR,
        ov::test::utils::OpType::VECTOR,
};

std::vector<ov::test::utils::OpType> opTypesDynamic = {
        ov::test::utils::OpType::VECTOR,
};

std::vector<ov::helpers::EltwiseTypes> eltwiseOpTypes = {
        ov::helpers::EltwiseTypes::ADD,
        ov::helpers::EltwiseTypes::MULTIPLY,
        ov::helpers::EltwiseTypes::SUBTRACT,
        ov::helpers::EltwiseTypes::DIVIDE,
        ov::helpers::EltwiseTypes::FLOOR_MOD,
        ov::helpers::EltwiseTypes::SQUARED_DIFF,
        ov::helpers::EltwiseTypes::POWER,
        ov::helpers::EltwiseTypes::MOD
};

std::vector<ov::helpers::EltwiseTypes> eltwiseOpTypesDynamic = {
        ov::helpers::EltwiseTypes::ADD,
        ov::helpers::EltwiseTypes::MULTIPLY,
        ov::helpers::EltwiseTypes::SUBTRACT,
};

ov::test::Config additional_config = {};

const auto multiply_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic)),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto collapsing_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStaticCheckCollapse)),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::Values(opTypes[1]),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto multiply_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(eltwiseOpTypesDynamic),
        ::testing::ValuesIn(secondaryInputTypesDynamic),
        ::testing::ValuesIn(opTypesDynamic),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto multiply_params_dynamic_large_upper_bound = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamicLargeUpperBound),
        ::testing::Values(ov::helpers::EltwiseTypes::ADD),
        ::testing::ValuesIn(secondaryInputTypesDynamic),
        ::testing::ValuesIn(opTypesDynamic),
        ::testing::Values(ov::element::f32),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static_check_collapsing, EltwiseLayerTest, collapsing_params, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, EltwiseLayerTest, multiply_params_dynamic, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic_large_upper_bound,
                         EltwiseLayerTest,
                         multiply_params_dynamic_large_upper_bound,
                         EltwiseLayerTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> inShapesSingleThread = {
        {{1, 2, 3, 4}},
        {{2, 2, 2, 2}},
        {{2, 1, 2, 1, 2, 2}},
};

std::vector<ov::helpers::EltwiseTypes> eltwiseOpTypesSingleThread = {
        ov::helpers::EltwiseTypes::ADD,
        ov::helpers::EltwiseTypes::POWER,
};

ov::AnyMap additional_config_single_thread = {
        {"CPU_THREADS_NUM", "1"}
};

const auto single_thread_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesSingleThread)),
        ::testing::ValuesIn(eltwiseOpTypesSingleThread),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread, EltwiseLayerTest, single_thread_params, EltwiseLayerTest::getTestCaseName);


} // namespace
