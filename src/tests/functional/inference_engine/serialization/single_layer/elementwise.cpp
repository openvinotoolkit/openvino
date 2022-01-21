// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/eltwise.hpp"

using namespace ov::test::subgraph;

namespace {
TEST_P(EltwiseLayerTest, Serialize) {
    serialize();
}

const std::vector<ov::test::ElementType> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
        };

std::vector<std::vector<ov::Shape>> inputShapes = {
        {{2}},
        {{1, 5, 50}},
        {{2, 10, 1, 4}, {2, 10, 1, 1}}
};

std::vector<std::vector<ov::test::InputShape>> inShapesDynamic = {
        {{{ngraph::Dimension(1, 10), 200}, {{6, 200}, {1, 200}}},
         {{ngraph::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ngraph::helpers::InputLayerType> secondaryInputTypes = {
        ngraph::helpers::InputLayerType::CONSTANT,
        ngraph::helpers::InputLayerType::PARAMETER,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
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

ov::AnyMap additionalConfig = {};

const auto elementiwiseParams = ::testing::Combine(
        ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inputShapes)),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additionalConfig));

const auto elementiwiseParamsDyn = ::testing::Combine(
        ::testing::ValuesIn(inShapesDynamic),
        ::testing::ValuesIn(eltwiseOpTypes),
        ::testing::ValuesIn(secondaryInputTypes),
        ::testing::ValuesIn(opTypes),
        ::testing::ValuesIn(inputPrecisions),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(ov::element::undefined),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::Values(additionalConfig));

INSTANTIATE_TEST_SUITE_P(smoke_ElementwiseSerialization_static, EltwiseLayerTest,
                        elementiwiseParams,
                        EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_ElementwiseSerialization_dynamic, EltwiseLayerTest,
                         elementiwiseParamsDyn,
                         EltwiseLayerTest::getTestCaseName);
} // namespace
