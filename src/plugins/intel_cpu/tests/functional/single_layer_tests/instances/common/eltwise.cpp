// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/eltwise.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include "test_utils/filter_cpu_params.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Eltwise {

const auto params_4D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder, EltwiseLayerCPUTest, params_4D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder, EltwiseLayerCPUTest, params_5D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D, EltwiseLayerCPUTest, params_5D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_constant_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Constant_mode())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Constant, EltwiseLayerCPUTest, params_4D_1D_constant_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_parameter_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Parameter_mode())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Parameter, EltwiseLayerCPUTest, params_4D_1D_parameter_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_constant = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_constant())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Constant, EltwiseLayerCPUTest, params_5D_1D_constant, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_parameter = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_parameter())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Parameter, EltwiseLayerCPUTest, params_5D_1D_parameter, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_4D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_4D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_const, EltwiseLayerCPUTest, params_5D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_param, EltwiseLayerCPUTest, params_5D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions
