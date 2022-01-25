// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/eltwise.hpp"
#include "common_test_utils/test_constants.hpp"
#include "common/myriad_common_test_utils.hpp"
#include <vpu/private_plugin_config.hpp>

#include <vector>

using namespace ov::test::subgraph;

namespace {

using Config = ov::AnyMap;

std::vector<std::vector<ov::Shape>>  inShapes = {
        {{2}},
        {{1, 1, 1, 3}},
        {{1, 2, 4}},
        {{1, 4, 4}},
        {{1, 4, 4, 1}},
        {{16, 16, 96}, {96}},
        {{52, 1, 52, 3, 2}, {2}},
};

std::vector<ov::test::ElementType> fpTypes = {
        ov::element::f32,
        ov::element::f16,
};

std::vector<ov::test::ElementType> intTypes = {
        ov::element::i32,
        ov::element::u32,
        ov::element::i64,
        ov::element::u64,
};

std::vector<CommonTestUtils::OpType> opTypes = {
        CommonTestUtils::OpType::SCALAR,
        CommonTestUtils::OpType::VECTOR,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseMathTypesFP = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::SUBTRACT,
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::DIVIDE,
        ngraph::helpers::EltwiseTypes::SQUARED_DIFF,
        ngraph::helpers::EltwiseTypes::POWER,
        ngraph::helpers::EltwiseTypes::FLOOR_MOD,
};

std::vector<ngraph::helpers::EltwiseTypes> eltwiseMathTypesINT = {
        ngraph::helpers::EltwiseTypes::MULTIPLY,
        ngraph::helpers::EltwiseTypes::ADD,
        ngraph::helpers::EltwiseTypes::DIVIDE,
};

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseMathFP,
                        EltwiseLayerTest,
                        ::testing::Combine(
                            ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                            ::testing::ValuesIn(eltwiseMathTypesFP),
                            ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                            ::testing::ValuesIn(opTypes),
                            ::testing::ValuesIn(fpTypes),
                            ::testing::Values(ov::element::undefined),
                            ::testing::Values(ov::element::undefined),
                            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                            ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_EltwiseMathInt,
                        EltwiseLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                                ::testing::ValuesIn(eltwiseMathTypesINT),
                                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                                ::testing::ValuesIn(opTypes),
                                ::testing::ValuesIn(intTypes),
                                ::testing::Values(ov::element::undefined),
                                ::testing::Values(ov::element::undefined),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD),
                                ::testing::Values(Config{{InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH, CONFIG_VALUE(NO)}})),
                        EltwiseLayerTest::getTestCaseName);
}  // namespace
