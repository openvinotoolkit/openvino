// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test::subgraph;

namespace {
std::vector<std::vector<ov::Shape>> inShapes = {
    {{2}},
    {{8}},
    {{1, 200}},
    {{1, 1, 1, 3}},
    {{1, 2, 4}},
    {{1, 4, 4}},
    {{1, 4, 4, 1}},
};

std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

std::vector<ov::test::utils::OpType> opTypes = {
    ov::test::utils::OpType::SCALAR,
    ov::test::utils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseOpTypes = {ngraph::helpers::EltwiseTypes::MULTIPLY,
                                                             ngraph::helpers::EltwiseTypes::SUBTRACT,
                                                             ngraph::helpers::EltwiseTypes::ADD};

std::vector<ov::AnyMap> additional_config_inputs_1 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1638.4"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_SCALE_FACTOR_0", "1638.4"}}};

std::vector<ov::AnyMap> additional_config_inputs_2 = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_SCALE_FACTOR_0", "1638.4"}, {"GNA_SCALE_FACTOR_1", "1638.4"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}, {"GNA_SCALE_FACTOR_0", "1638.4"}, {"GNA_SCALE_FACTOR_1", "1638.4"}}};

const auto multiply_params_1 =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                       ::testing::ValuesIn(eltwiseOpTypes),
                       ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::test::utils::DEVICE_GNA),
                       ::testing::ValuesIn(additional_config_inputs_1));

const auto multiply_params_2 =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                       ::testing::ValuesIn(eltwiseOpTypes),
                       ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::test::utils::DEVICE_GNA),
                       ::testing::ValuesIn(additional_config_inputs_2));

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseLayerConstTest,
                         EltwiseLayerTest,
                         multiply_params_1,
                         EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseLayerParamTest,
                         EltwiseLayerTest,
                         multiply_params_2,
                         EltwiseLayerTest::getTestCaseName);

}  // namespace
