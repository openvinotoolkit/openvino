// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/eltwise.hpp"
#include "shared_test_classes/single_layer/eltwise.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/fusing_test_utils.hpp"
#include <ngraph_functions/builders.hpp>
#include <common_test_utils/ov_tensor_utils.hpp>

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;


namespace CPULayerTestsDefinitions {
namespace Eltwise {

std::vector<ElementType> netType_x64 = {
        ElementType::bf16};

const std::vector<CPUSpecificParams>& cpuParams_4D_Blocked_Planar() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Planar = {
                CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {}),
        };
        return cpuParams_4D_Blocked_Planar;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_Planar_Blocked() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_Planar_Blocked = {
                CPUSpecificParams({nchw, nChw16c}, {nChw16c}, {}, {}),
        };
        return cpuParams_4D_Planar_Blocked;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_Blocked_Planar() {
        static const std::vector<CPUSpecificParams> cpuParams_5D_Blocked_Planar = {
                CPUSpecificParams({nCdhw16c, ncdhw}, {nCdhw16c}, {}, {}),
        };
        return cpuParams_5D_Blocked_Planar;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_Planar_Blocked() {
        static const std::vector<CPUSpecificParams> cpuParams_5D_Planar_Blocked = {
                CPUSpecificParams({ncdhw, nCdhw16c}, {nCdhw16c}, {}, {}),
        };
        return cpuParams_5D_Planar_Blocked;
}

const std::vector<CPUSpecificParams> & cpuParams_4D_1D_Constant_mode_x64() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_1D_Constant_mode = {
                CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {})
        };
        return cpuParams_4D_1D_Constant_mode;
}

const std::vector<fusingSpecificParams> fusingParamsSet_x64{
    // eltwise
    fusingSigmoid,
    fusingPRelu1D,
    // depthwise
    fusingReluScaleShift,
    // fake quantize
    fusingFakeQuantizePerTensorRelu,
    fusingFakeQuantizePerChannelRelu,
    fusingFQPerChannelSigmoidFQPerTensor
};

const auto params_4D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_x64, EltwiseLayerCPUTest, params_4D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_fusing())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::ValuesIn(fusingParamsSet_x64));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Fusing_x64, EltwiseLayerCPUTest, params_4D_fusing, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D())),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec_x64, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_x64, EltwiseLayerCPUTest, params_5D, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_emptyCPUSpec = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                ::testing::ValuesIn(eltwiseOpTypesDiffInp()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_x64, EltwiseLayerCPUTest, params_5D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const std::vector<fusingSpecificParams> fusingParamsSet_x64I32_x64{
    fusingMultiplyAddPerChannel
};

const auto params_5D_emptyCPUSpec_I32 = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D())),
                ::testing::ValuesIn(eltwiseOpTypesI32()),
                ::testing::ValuesIn(secondaryInputTypes()),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::i32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(fusingParamsSet_x64I32_x64));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_I32_x64, EltwiseLayerCPUTest, params_5D_emptyCPUSpec_I32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Blocked_Planar())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Planar())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Planar, EltwiseLayerCPUTest, params_4D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Planar_Blocked())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar_Blocked())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_Blocked, EltwiseLayerCPUTest, params_4D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Blocked_Planar())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Planar())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Blocked_Planar, EltwiseLayerCPUTest, params_5D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Planar_Blocked())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Planar_Blocked())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Planar_Blocked_x64, EltwiseLayerCPUTest, params_5D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_constant_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Constant_mode_x64())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Constant_x64, EltwiseLayerCPUTest, params_4D_1D_constant_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_parameter_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Parameter_mode())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Parameter_x64, EltwiseLayerCPUTest, params_4D_1D_parameter_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_constant = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_constant())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Constant_x64, EltwiseLayerCPUTest, params_5D_1D_constant, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_parameter = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ngraph::helpers::EltwiseTypes::ADD, ngraph::helpers::EltwiseTypes::MULTIPLY),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_parameter())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Parameter_x64, EltwiseLayerCPUTest, params_5D_1D_parameter, EltwiseLayerCPUTest::getTestCaseName);

//// ============================================ 4D ============================================

const auto params_4D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_const_x64, EltwiseLayerCPUTest, params_4D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_dyn_param_x64, EltwiseLayerCPUTest, params_4D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_param_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::ValuesIn(fusingParamsSet_x64));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_dyn_param_fusing_x64, EltwiseLayerCPUTest, params_4D_dyn_param_fusing, EltwiseLayerCPUTest::getTestCaseName);

//// ============================================ 5D ============================================

const auto params_5D_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ngraph::helpers::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_const_x64, EltwiseLayerCPUTest, params_5D_dyn_const, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ngraph::helpers::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType_x64),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(CommonTestUtils::DEVICE_CPU),
                ::testing::Values(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D())),
        ::testing::Values(emptyFusingSpec));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_param_x64, EltwiseLayerCPUTest, params_5D_dyn_param, EltwiseLayerCPUTest::getTestCaseName);

} // namespace Eltwise
} // namespace CPULayerTestsDefinitions