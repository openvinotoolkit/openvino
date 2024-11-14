// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/activation.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

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
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Eltwise_CPU, ActivationLayerCPUTest, basicCases3D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets3D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic3D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams3D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation3D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets3D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (2D) ============= */
const auto basicCases4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Eltwise_CPU, ActivationLayerCPUTest, basicCases4D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets4D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic4D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams4D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation4D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets4D, ActivationLayerCPUTest::getTestCaseName);

/* ============= Activation (5D) ============= */
const auto basicCases5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypes())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Eltwise_CPU, ActivationLayerCPUTest, basicCases5D, ActivationLayerCPUTest::getTestCaseName);

const auto basicCasesSnippets5D = ::testing::Combine(
    ::testing::ValuesIn(static_shapes_to_test_representation(basic5D())),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesSnippets())),
    ::testing::ValuesIn(netPrc()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams5D())),
    ::testing::Values(true)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_Snippets_CPU, ActivationLayerCPUTest, basicCasesSnippets5D, ActivationLayerCPUTest::getTestCaseName);

const auto dynamicMathBasicCases = ::testing::Combine(
    ::testing::ValuesIn(dynamicMathBasic()),
    ::testing::Values(activationShapes()),
    ::testing::ValuesIn(ov::test::utils::combineParams(activationTypesDynamicMath())),
    ::testing::ValuesIn(netPrecisions()),
    ::testing::Values(ov::element::f32),
    ::testing::Values(ov::element::f32),
    ::testing::ValuesIn(cpuParamsDynamicMath()),
    ::testing::Values(false)
);

INSTANTIATE_TEST_SUITE_P(smoke_Activation5D_dynamicMath_CPU, ActivationLayerCPUTest, dynamicMathBasicCases, ActivationLayerCPUTest::getTestCaseName);

} // namespace Activation
}  // namespace test
}  // namespace ov
