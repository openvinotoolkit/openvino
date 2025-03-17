// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/bitwise_shift.hpp"

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {

std::vector<int32_t> shift_1{1};
uint32_t max_val = 3;
auto val_map_1_2 = ov::AnyMap{{"shift", shift_1}, {"max_val", max_val}};

static const std::vector<std::vector<InputShape>> bitwise_in_shapes_4D = {
    {{{-1, -1, -1, -1}, {{1, 3, 2, 2}, {1, 3, 1, 1}}}, {{1, 3, 2, 2}, {{1, 3, 2, 2}}}},
    {{{-1, -1, -1, -1}, {{2, 64, 2, 2}, {1, 64, 1, 1}}}, {{1, 64, 2, 2}, {{1, 64, 2, 2}}}},
};

const auto params_4D_bitwise_shift = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
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
    ::testing::Values(false),
    ::testing::Values(val_map_1_2));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_BitwiseShift,
                         BitwiseShiftLayerCPUTest,
                         params_4D_bitwise_shift,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

const auto params_4D_bitwise_shift_i32_cast = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
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
    ::testing::Values(false),
    ::testing::Values(val_map_1_2));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_BitwiseShift_i32_cast,
                         BitwiseShiftLayerCPUTest,
                         params_4D_bitwise_shift_i32_cast,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

std::vector<int32_t> shift_14{15};
uint32_t max_val_63 = 63;
auto val_map_overflow_cast = ov::AnyMap{{"shift", shift_14}, {"max_val", max_val_63}};

const auto params_4D_bitwise_shift_overflow_i32_cast = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn(
            {ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, "ref_I32$/"),
                         CPUSpecificParams({nchw, nchw}, {nchw}, {}, "ref_I32$/")}),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_overflow_cast));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_BitwiseShift_overflow_i32_cast,
                         BitwiseShiftLayerCPUTest,
                         params_4D_bitwise_shift_overflow_i32_cast,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

std::vector<int32_t> shift_7{7};
uint32_t max_val_15 = 15;
auto val_map_overflow_8_cast = ov::AnyMap{{"shift", shift_7}, {"max_val", max_val_15}};

const auto params_4D_bitwise_shift_overflow_8 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn(
        {CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, "ref"), CPUSpecificParams({nchw, nchw}, {nchw}, {}, "ref")}),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_overflow_8_cast));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_BitwiseShift_overflow_8,
                         BitwiseShiftLayerCPUTest,
                         params_4D_bitwise_shift_overflow_8,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

std::vector<int32_t> multi_shift_5{0, 1, 2, 3, 4};
uint32_t max_val_7 = 7;
auto val_map_multi_shift_5 = ov::AnyMap{{"shift", multi_shift_5}, {"max_val", max_val_7}};

static const std::vector<std::vector<ov::Shape>> bitwise_in_shapes_5D_1D = {
    {{2, 17, 8, 4, 5}, {5}},
};

const auto params_5D_1D_bitwise_shift = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(bitwise_in_shapes_5D_1D)),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, {})),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_multi_shift_5));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_BitwiseShift,
                         BitwiseShiftLayerCPUTest,
                         params_5D_1D_bitwise_shift,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

const auto params_5D_1D_bitwise_shift_cast_i32 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(bitwise_in_shapes_5D_1D)),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({ncdhw, x}, {ncdhw}, {}, "ref_I32$/")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_multi_shift_5));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_BitwiseShift_cast_i32,
                         BitwiseShiftLayerCPUTest,
                         params_5D_1D_bitwise_shift_cast_i32,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

static const std::vector<std::vector<ov::Shape>> bitwise_in_shapes_4D_1D = {
    {{2, 3, 4, 5}, {5}},
};

const auto params_4D_1D_bitwise_shift = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(bitwise_in_shapes_4D_1D)),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({nchw, x}, {nchw}, {}, {})),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_multi_shift_5));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_BitwiseShift,
                         BitwiseShiftLayerCPUTest,
                         params_4D_1D_bitwise_shift,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

const auto params_4D_1D_bitwise_shift_cast_i32 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(bitwise_in_shapes_4D_1D)),
        ::testing::ValuesIn({ov::test::utils::EltwiseTypes::LEFT_SHIFT, ov::test::utils::EltwiseTypes::RIGHT_SHIFT}),
        ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32}),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::element::Type_t::dynamic),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::Values(CPUSpecificParams({nchw, x}, {nchw}, {}, "ref_I32$/")),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false),
    ::testing::Values(val_map_multi_shift_5));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_BitwiseShift_cast_i32,
                         BitwiseShiftLayerCPUTest,
                         params_4D_1D_bitwise_shift_cast_i32,
                         BitwiseShiftLayerCPUTest::getTestCaseName);

}  // namespace Eltwise
}  // namespace test
}  // namespace ov
