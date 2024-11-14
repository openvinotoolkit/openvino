// Copyright (C) 2018-2024 Intel Corporation
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

std::vector<std::vector<ov::Shape>>  inShapes = {
        {{2}},
        {{}, {34100}},
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
        {{1, 4, 3, 2, 1, 3}},
        {{1, 3, 1, 1, 1, 3}, {1, 3, 1, 1, 1, 1}},
        {{1, 3, 2, 2, 2, 3, 2, 3}, {1, 3, 1, 1, 1, 1, 1, 1}},
        {{1, 3, 2, 2, 2, 3, 2, 3}, {3}},
        {{1, 3, 2, 2, 2, 3, 2, 3}, {1, 3, 2, 2, 2, 3, 2, 3}},
        {{1, 3, 2, 2, 2, 3, 2}, {1, 3, 2, 2, 2, 3, 2}},
};

std::vector<ov::test::ElementType> netPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i64,
};

std::vector<InputLayerType> secondaryInputTypes = {
        InputLayerType::CONSTANT,
        InputLayerType::PARAMETER,
};

std::vector<ov::test::utils::OpType> opTypes = {
        ov::test::utils::OpType::SCALAR,
        ov::test::utils::OpType::VECTOR,
};

std::vector<EltwiseTypes> smoke_eltwiseOpTypes = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
};

std::vector<EltwiseTypes> eltwiseOpTypes = {
        EltwiseTypes::ADD,
        EltwiseTypes::MULTIPLY,
        EltwiseTypes::SUBTRACT,
        EltwiseTypes::DIVIDE,
        EltwiseTypes::FLOOR_MOD,
        EltwiseTypes::SQUARED_DIFF,
        EltwiseTypes::POWER,
        EltwiseTypes::MOD
};

std::vector<EltwiseTypes> smoke_intOnly_eltwiseOpTypes = {
        EltwiseTypes::RIGHT_SHIFT,
        EltwiseTypes::BITWISE_AND
};

std::vector<ov::test::ElementType> intOnly_netPrecisions = {
        ov::element::i32,
        ov::element::i16,
        ov::element::u16
};

ov::AnyMap additional_config = {};

INSTANTIATE_TEST_SUITE_P(
    smoke_intOnly_CompareWithRefs,
    EltwiseLayerTest,
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                       ::testing::ValuesIn(smoke_intOnly_eltwiseOpTypes),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(intOnly_netPrecisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(additional_config)),
    EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_CompareWithRefs,
    EltwiseLayerTest,
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                       ::testing::ValuesIn(smoke_eltwiseOpTypes),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(additional_config)),
    EltwiseLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    CompareWithRefs,
    EltwiseLayerTest,
    ::testing::Combine(::testing::ValuesIn(ov::test::static_shapes_to_test_representation(inShapes)),
                       ::testing::ValuesIn(eltwiseOpTypes),
                       ::testing::ValuesIn(secondaryInputTypes),
                       ::testing::ValuesIn(opTypes),
                       ::testing::ValuesIn(netPrecisions),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::element::undefined),
                       ::testing::Values(ov::test::utils::DEVICE_GPU),
                       ::testing::Values(additional_config)),
    EltwiseLayerTest::getTestCaseName);

}  // namespace
