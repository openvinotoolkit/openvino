// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "single_op_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::EltwiseLayerTest;
using ov::test::utils::InputLayerType;
using ov::test::utils::OpType;
using ov::test::utils::EltwiseTypes;

std::vector<std::vector<ov::Shape>> in_shapes_static = {
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

std::vector<std::vector<ov::Shape>> in_shapes_static_check_collapse = {
        {{16, 16, 16, 16}, {16, 16, 16, 1}},
        {{16, 16, 16, 1}, {16, 16, 16, 1}},
        {{16, 16, 16, 16}, {16, 16, 1, 16}},
        {{16, 16, 1, 16}, {16, 16, 1, 16}},
};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}},
         {{ngraph::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<std::vector<ov::test::InputShape>> in_shapes_dynamic_large_upper_bound = {
        {{{ngraph::Dimension(1, 1000000000000), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> model_types = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

std::vector<InputLayerType> secondary_input_types = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::vector<InputLayerType> secondary_input_types_dynamic = {
        InputLayerType::PARAMETER,
};

std::vector<OpType> op_types = {
        OpType::SCALAR,
        OpType::VECTOR,
};

std::vector<OpType> op_types_dynamic = {
        OpType::VECTOR,
};

std::vector<EltwiseTypes> eltwise_op_types = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::SUBTRACT,
        EltwiseTypes::DIVIDE,
        EltwiseTypes::FLOOR_MOD,
        EltwiseTypes::SQUARED_DIFF,
        EltwiseTypes::POWER,
        EltwiseTypes::MOD
};

std::vector<EltwiseTypes> eltwise_op_types_dynamic = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::SUBTRACT,
};

ov::test::Config additional_config = {};

const auto multiply_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_static)),
        ::testing::ValuesIn(eltwise_op_types),
        ::testing::ValuesIn(secondary_input_types),
        ::testing::ValuesIn(op_types),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto collapsing_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(in_shapes_static_check_collapse)),
        ::testing::ValuesIn(eltwise_op_types),
        ::testing::ValuesIn(secondary_input_types),
        ::testing::Values(op_types[1]),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto multiply_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(in_shapes_dynamic),
        ::testing::ValuesIn(eltwise_op_types_dynamic),
        ::testing::ValuesIn(secondary_input_types_dynamic),
        ::testing::ValuesIn(op_types_dynamic),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config));

const auto multiply_params_dynamic_large_upper_bound = ::testing::Combine(
        ::testing::ValuesIn(in_shapes_dynamic_large_upper_bound),
        ::testing::Values(EltwiseTypes::ADD),
        ::testing::ValuesIn(secondary_input_types_dynamic),
        ::testing::ValuesIn(op_types_dynamic),
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

std::vector<EltwiseTypes> eltwise_op_typesSingleThread = {
        EltwiseTypes::ADD,
        EltwiseTypes::POWER,
};

ov::AnyMap additional_config_single_thread = {
        {"CPU_THREADS_NUM", "1"}
};

const auto single_thread_params = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesSingleThread)),
        ::testing::ValuesIn(eltwise_op_typesSingleThread),
        ::testing::ValuesIn(secondary_input_types),
        ::testing::ValuesIn(op_types),
        ::testing::ValuesIn(model_types),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread, EltwiseLayerTest, single_thread_params, EltwiseLayerTest::getTestCaseName);


} // namespace
