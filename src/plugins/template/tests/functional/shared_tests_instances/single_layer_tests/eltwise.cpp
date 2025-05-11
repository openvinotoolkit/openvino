// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/eltwise.hpp"

#include <vector>

#include "common_test_utils/test_constants.hpp"

using namespace ov::test;

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
    {{{ov::Dimension(1, 10), 200}, {{2, 200}, {1, 200}}}, {{ov::Dimension(1, 10), 200}, {{2, 200}, {5, 200}}}},
};

std::vector<ov::test::ElementType> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
    ov::element::i32,
};

std::vector<ov::test::utils::InputLayerType> secondaryInputTypes = {
    ov::test::utils::InputLayerType::CONSTANT,
    ov::test::utils::InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::InputLayerType> secondaryInputTypesDynamic = {
    ov::test::utils::InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::OpType> opTypes = {
    ov::test::utils::OpType::SCALAR,
    ov::test::utils::OpType::VECTOR,
};

std::vector<ov::test::utils::OpType> opTypesDynamic = {
    ov::test::utils::OpType::VECTOR,
};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypes = {ov::test::utils::EltwiseTypes::ADD,
                                                             ov::test::utils::EltwiseTypes::MULTIPLY,
                                                             ov::test::utils::EltwiseTypes::SUBTRACT,
                                                             ov::test::utils::EltwiseTypes::DIVIDE,
                                                             ov::test::utils::EltwiseTypes::FLOOR_MOD,
                                                             ov::test::utils::EltwiseTypes::SQUARED_DIFF,
                                                             ov::test::utils::EltwiseTypes::POWER,
                                                             ov::test::utils::EltwiseTypes::MOD};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypesDynamic = {
    ov::test::utils::EltwiseTypes::ADD,
    ov::test::utils::EltwiseTypes::MULTIPLY,
    ov::test::utils::EltwiseTypes::SUBTRACT,
};

ov::test::Config additional_config = {};

const auto multiply_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesStatic)),
                       ::testing::ValuesIn(eltwiseOpTypes),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                       ::testing::Values(additional_config));

const auto multiply_params_dynamic = ::testing::Combine(::testing::ValuesIn(inShapesDynamic),
                                                        ::testing::ValuesIn(eltwiseOpTypesDynamic),
                                                        ::testing::ValuesIn(secondaryInputTypesDynamic),
                                                        ::testing::ValuesIn(opTypesDynamic),
                                                        ::testing::ValuesIn(netPrecisions),
                                                        ::testing::Values(ov::element::dynamic),
                                                        ::testing::Values(ov::element::dynamic),
                                                        ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                                                        ::testing::Values(additional_config));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_static,
                         EltwiseLayerTest,
                         multiply_params,
                         EltwiseLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_dynamic,
                         EltwiseLayerTest,
                         multiply_params_dynamic,
                         EltwiseLayerTest::getTestCaseName);

std::vector<std::vector<ov::Shape>> inShapesSingleThread = {
    {{1, 2, 3, 4}},
    {{2, 2, 2, 2}},
    {{2, 1, 2, 1, 2, 2}},
};

std::vector<ov::test::utils::EltwiseTypes> eltwiseOpTypesSingleThread = {
    ov::test::utils::EltwiseTypes::ADD,
    ov::test::utils::EltwiseTypes::POWER,
};

ov::AnyMap additional_config_single_thread = {ov::inference_num_threads(1)};

const auto single_thread_params =
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapesSingleThread)),
                       ::testing::ValuesIn(eltwiseOpTypesSingleThread),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_TEMPLATE),
                       ::testing::Values(additional_config_single_thread));

INSTANTIATE_TEST_SUITE_P(smoke_SingleThread, EltwiseLayerTest, single_thread_params, EltwiseLayerTest::getTestCaseName);

}  // namespace
