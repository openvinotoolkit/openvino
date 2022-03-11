// Copyright (C) 2018-2022 Intel Corporation
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
        {{1, 1, 1, 1, 1, 1, 3}},
        {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}},
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}},
         {{ngraph::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypesDynamic = {
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<CommonTestUtils::OpType> opTypesDynamic = {
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::MOD
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesDynamic = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
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
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(additional_config));

const auto multiply_params_dynamic = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(eltwiseOpTypesDynamic),
        ::testing::ValuesIn(secondaryInputTypesDynamic),
        ::testing::ValuesIn(opTypesDynamic),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static, EltwiseLayerTest, multiply_params, EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic, EltwiseLayerTest, multiply_params_dynamic, EltwiseLayerTest::getTestCaseName);


std::vector<std::vector<ov::Shape>> inShapesSingleThread = {
        {{1, 2, 3, 4}},
        {{2, 2, 2, 2}},
        {{2, 1, 2, 1, 2, 2}},
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypesSingleThread = {
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::POWER,
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
        ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE),
        ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread, EltwiseLayerTest, single_thread_params, EltwiseLayerTest::getTestCaseName);


} // namespace