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
