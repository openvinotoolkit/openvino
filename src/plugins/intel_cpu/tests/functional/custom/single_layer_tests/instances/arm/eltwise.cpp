// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {
namespace {

static const std::vector<std::vector<InputShape>> bitwise_in_shapes_4D = {
    // operations with scalar for nchw
    {
        {
            {1, -1, -1, -1},
            {
                {1, 3, 2, 2},
                {1, 3, 1, 1}
            }
        },
        {{1, 3, 2, 2}, {{1, 3, 2, 2}}}
    },
    // operations with vector for nchw
    {
        {
            {1, -1, -1, -1},
            {
                {1, 64, 2, 2},
                {1, 64, 1, 1}
            }
        },
        {{1, 64, 2, 2}, {{1, 64, 2, 2}}}
    },
};

const auto params_4D_bitwise = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::BITWISE_AND,
                             ov::test::utils::EltwiseTypes::BITWISE_OR,
                             ov::test::utils::EltwiseTypes::BITWISE_XOR}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn(
        {CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, "ref"), CPUSpecificParams({nchw, nchw}, {nchw}, {}, "ref")}),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise, EltwiseLayerCPUTest, params_4D_bitwise, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_i32 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::BITWISE_AND,
                             ov::test::utils::EltwiseTypes::BITWISE_OR,
                             ov::test::utils::EltwiseTypes::BITWISE_XOR}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, "ref_I32$/"),
                         CPUSpecificParams({nchw, nchw}, {nchw}, {}, "ref_I32$/")}),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_i32, EltwiseLayerCPUTest, params_4D_bitwise_i32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_NOT = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::BITWISE_NOT}),
        ::testing::ValuesIn({ov::test::utils::InputLayerType::CONSTANT}),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({CPUSpecificParams({nhwc}, {nhwc}, {}, "ref"), CPUSpecificParams({nchw}, {nchw}, {}, "ref")}),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT, EltwiseLayerCPUTest, params_4D_bitwise_NOT, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_NOT_i32 =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(bitwise_in_shapes_4D),
                                          ::testing::ValuesIn({ov::test::utils::EltwiseTypes::BITWISE_NOT}),
                                          ::testing::ValuesIn({ov::test::utils::InputLayerType::CONSTANT}),
                                          ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
                                          ::testing::ValuesIn({ov::element::Type_t::i16}),
                                          ::testing::Values(ov::element::Type_t::dynamic),
                                          ::testing::Values(ov::element::Type_t::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(ov::AnyMap())),
                       ::testing::ValuesIn({CPUSpecificParams({nhwc}, {nhwc}, {}, "ref_I32$/"),
                                            CPUSpecificParams({nchw}, {nchw}, {}, "ref_I32$/")}),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT_i32, EltwiseLayerCPUTest, params_4D_bitwise_NOT_i32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_int_jit = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
        ::testing::ValuesIn({utils::EltwiseTypes::ADD, utils::EltwiseTypes::MULTIPLY}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn(opTypes()),
        ::testing::ValuesIn({ElementType::i8, ElementType::u8, ElementType::f16, ElementType::i32, ElementType::f32}),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::element::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_int_jit, EltwiseLayerCPUTest, params_4D_int_jit, EltwiseLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
