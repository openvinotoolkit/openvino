// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/activation.hpp"
#include "utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Activation {

const auto basicCasesFp32 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
        ::testing::Values(activationShapes()),
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesF16())),
        ::testing::ValuesIn({ov::element::f32}),
        ::testing::ValuesIn({ov::element::f32}),
        ::testing::ValuesIn({ov::element::f32}),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
        ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Eltwise_CPU_f32, ActivationLayerCPUTest, basicCasesFp32, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesFp16 = ::testing::Combine(
        ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
        ::testing::Values(activationShapes()),
        ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesF16())),
        ::testing::ValuesIn({ov::element::f16}),
        ::testing::ValuesIn({ov::element::f16}),
        ::testing::ValuesIn({ov::element::f16}),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
        ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation_Eltwise_CPU_f16, ActivationLayerCPUTest, basicCasesFp16, ActivationLayerCPUTest::getTestCaseName);

} // namespace Activation
}  // namespace test
}  // namespace ov
