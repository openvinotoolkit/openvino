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
const auto params_4D =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                                          ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder, EltwiseLayerCPUTest, params_4D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_Snippets =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                                          ::testing::ValuesIn(eltwiseOpTypesBinInpSnippets()),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::Values(additional_config()[0])),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(true));
INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_Snippets, EltwiseLayerCPUTest, params_4D_Snippets, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_emptyCPUSpec =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                                          ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::Values(emptyCPUSpec),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                                          ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder, EltwiseLayerCPUTest, params_5D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_emptyCPUSpec =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                                          ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                                          ::testing::ValuesIn(secondaryInputTypes()),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::Values(emptyCPUSpec),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D, EltwiseLayerCPUTest, params_5D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_constant_mode = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                       ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::ValuesIn(netType()),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Constant_mode())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Constant, EltwiseLayerCPUTest, params_4D_1D_constant_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_parameter_mode = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                       ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::ValuesIn(netType()),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Parameter_mode())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Parameter, EltwiseLayerCPUTest, params_4D_1D_parameter_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_constant = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                       ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::ValuesIn(netType()),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_constant())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Constant, EltwiseLayerCPUTest, params_5D_1D_constant, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_parameter = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                       ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::ValuesIn(netType()),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_parameter())),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Parameter, EltwiseLayerCPUTest, params_5D_1D_parameter, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_const =
    ::testing::Combine(::testing::Combine(::testing::ValuesIn(inShapes_4D_dyn_const()),
                                          ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                                          ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_4D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_param =
    ::testing::Combine(::testing::Combine(::testing::Values(inShapes_4D_dyn_param()),
                                          ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                                          ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_4D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_const =
    ::testing::Combine(::testing::Combine(::testing::Values(inShapes_5D_dyn_const()),
                                          ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                                          ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_5D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_param =
    ::testing::Combine(::testing::Combine(::testing::Values(inShapes_5D_dyn_param()),
                                          ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                                          ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                                          ::testing::ValuesIn(opTypes()),
                                          ::testing::ValuesIn(netType()),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::element::dynamic),
                                          ::testing::Values(ov::test::utils::DEVICE_CPU),
                                          ::testing::ValuesIn(additional_config())),
                       ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
                       ::testing::Values(emptyFusingSpec),
                       ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_5D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

const auto params_fma_4D = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_fusing_4D())),
                       ::testing::Values(utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
    ::testing::ValuesIn({fusingMultiplyAddPerChannel}),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_fma_4D, EltwiseLayerCPUTest, params_fma_4D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_fma_5D = ::testing::Combine(
    ::testing::Combine(::testing::ValuesIn(static_shapes_to_test_representation(inShapes_fusing_5D())),
                       ::testing::Values(utils::EltwiseTypes::MULTIPLY),
                       ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                       ::testing::ValuesIn(opTypes()),
                       ::testing::Values(ElementType::f32),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::element::dynamic),
                       ::testing::Values(ov::test::utils::DEVICE_CPU),
                       ::testing::ValuesIn(additional_config())),
    ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
    ::testing::ValuesIn({fusingMultiplyAddPerChannel}),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_fma_5D, EltwiseLayerCPUTest, params_fma_5D, EltwiseLayerCPUTest::getTestCaseName);

}  // namespace Eltwise
}  // namespace test
}  // namespace ov
