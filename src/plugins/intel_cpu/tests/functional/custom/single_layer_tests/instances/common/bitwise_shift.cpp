// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/eltwise.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

#include "custom/single_layer_tests/classes/bitwise_shift.hpp"


using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Eltwise {

std::vector<int32_t> shift_1{1};
uint32_t max_val_2 = 2;
auto val_map_1_2 = ov::AnyMap{{"shift", shift_1}, {"max_val", max_val_2}};

static const std::vector<std::vector<InputShape>> bitwise_in_shapes_4D = {
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

const auto params_4D_bitwise_shift = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn(
            {ov::test::utils::EltwiseTypes::BITWISE_LEFT_SHIFT, ov::test::utils::EltwiseTypes::BITWISE_RIGHT_SHIFT}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32}),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({CPUSpecificParams({nhwc, nhwc}, {nhwc}, {}, "ref"),
                         CPUSpecificParams({nchw, nchw}, {nchw}, {}, "ref")}),
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
        ::testing::ValuesIn(
            {ov::test::utils::EltwiseTypes::BITWISE_LEFT_SHIFT, ov::test::utils::EltwiseTypes::BITWISE_RIGHT_SHIFT}),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ov::test::utils::OpType::VECTOR}),
        ::testing::ValuesIn({ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32}),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
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
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
