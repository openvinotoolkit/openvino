// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/activation.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Activation {

/* ============= Activation (1D) ============= */
const auto basicCases3D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic3D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3D()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Eltwise_CPU, ActivationLayerCPUTest, basicCases3D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (2D) ============= */
const auto basicCases4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (5D) ============= */
const auto basicCases5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D()))
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

const auto dynamicMathBasicCases = ::testing::Combine(
    ::testing::ValuesIn(dynamicMathBasic()),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesDynamicMath())),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(cpuParamsDynamicMath())
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_dynamicMath_CPU, ActivationLayerCPUTest, dynamicMathBasicCases, ActivationLayerCPUTest::getTestCaseName);

} // namespace Activation
}  // namespace test
}  // namespace ov
