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

const std::vector<ElementType>& netType() {
        static const std::vector<ElementType> netType = {
                ElementType::bf16};
        return netType;
}

const std::vector<InputShape>& inShapes_4D_dyn_param_fusing() {
        static const std::vector<InputShape> inShapes_4D_dyn_param_fusing = {
        {
                // dynamic
                {-1, 7, -1, -1},
                // target
                {
                {3, 7, 1, 1},
                {1, 7, 5, 1},
                {3, 7, 1, 1},
                {3, 7, 4, 11},
                }
        },
        {
                // dynamic
                {-1, 7, -1, -1},
                // target
                {
                {1, 7, 5, 1},
                {3, 7, 1, 10},
                {1, 7, 5, 1},
                {3, 7, 4, 11}
                }
        }
        };
        return inShapes_4D_dyn_param_fusing;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D_Planar_Blocked() {
        static const std::vector<std::vector<ov::Shape>> inShapes_4D_Planar_Blocked = {
                {{2, 1, 31, 3}, {2, 17, 31, 3}},
                {{2, 1, 1, 4}, {2, 17, 5, 1}},
        };
        return inShapes_4D_Planar_Blocked;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D_fusing() {
        static const std::vector<std::vector<ov::Shape>> inShapes_4D_fusing = {
                {{2, 4, 4, 1}},
                {{2, 17, 5, 4}},
                {{2, 17, 5, 1}, {1, 17, 1, 4}},
        };
        return inShapes_4D_fusing;
}

const std::vector<std::vector<ov::Shape>>& inShapes_4D_Blocked_Planar() {
        static const std::vector<std::vector<ov::Shape>> inShapes_4D_Blocked_Planar = {
                {{2, 17, 31, 3}, {2, 1, 31, 3}},
                {{2, 17, 5, 1}, {2, 1, 1, 4}},
        };
        return inShapes_4D_Blocked_Planar;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_Blocked_Blocked() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Blocked = {
                CPUSpecificParams({nChw16c, nChw16c}, {nChw16c}, {}, {})
        };
        return cpuParams_4D_Blocked_Blocked;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_Blocked_Planar() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_Blocked_Planar = {
                CPUSpecificParams({nChw16c, nchw}, {nChw16c}, {}, {})
        };
        return cpuParams_4D_Blocked_Planar;
}

const std::vector<CPUSpecificParams>& cpuParams_4D_Planar_Blocked() {
        static const std::vector<CPUSpecificParams> cpuParams_4D_Planar_Blocked = {
                CPUSpecificParams({nchw, nChw16c}, {nChw16c}, {}, {})
        };
        return cpuParams_4D_Planar_Blocked;
}

const std::vector<CPUSpecificParams>& cpuParams_5D_Blocked_Blocked() {
        static const std::vector<CPUSpecificParams> cpuParams_5D_Blocked_Blocked = {
                CPUSpecificParams({nCdhw16c, nCdhw16c}, {nCdhw16c}, {}, {})
        };
        return cpuParams_5D_Blocked_Blocked;
}

const std::vector<std::vector<ov::Shape>>& inShapes_5D_Blocked_Planar() {
        static const std::vector<std::vector<ov::Shape>> inShapes_5D_Blocked_Planar = {
                {{2, 17, 31, 4, 3}, {2, 1, 31, 1, 3}},
                {{2, 17, 5, 3, 1}, {2, 1, 1, 3, 4}},
        };
        return inShapes_5D_Blocked_Planar;
}

const std::vector<std::vector<ov::Shape>>& inShapes_5D_Planar_Blocked() {
        static const std::vector<std::vector<ov::Shape>> inShapes_5D_Planar_Blocked = {
                {{2, 1, 31, 1, 3}, {2, 17, 31, 4, 3}},
                {{2, 1, 1, 3, 4}, {2, 17, 5, 3, 1}},
        };
        return inShapes_5D_Planar_Blocked;
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

const auto params_4D_Blocked_Blocked = ::testing::Combine(
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
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_MemOrder_Blocked_Blocked, EltwiseLayerCPUTest, params_4D_Blocked_Blocked,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_fusing())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Fusing, EltwiseLayerCPUTest, params_4D_fusing, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_fusing_blocked_blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_fusing())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Fusing_Blocked_Blocked, EltwiseLayerCPUTest, params_4D_fusing_blocked_blocked,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_blocked_blocked_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_fusing())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Blocked_Fusing, EltwiseLayerCPUTest, params_4D_blocked_blocked_fusing,
                         EltwiseLayerCPUTest::getTestCaseName);

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
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_emptyCPUSpec_x64, EltwiseLayerCPUTest, params_4D_emptyCPUSpec, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_Blocked_Blocked = ::testing::Combine(
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
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_Blocked_Blocked, EltwiseLayerCPUTest, params_5D_Blocked_Blocked,
                         EltwiseLayerCPUTest::getTestCaseName);

const std::vector<fusingSpecificParams> fusingParamsSet_I32{
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
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::Values(emptyCPUSpec),
        ::testing::ValuesIn(fusingParamsSet_I32),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_I32, EltwiseLayerCPUTest, params_5D_emptyCPUSpec_I32, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Blocked_Planar())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Planar())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Planar, EltwiseLayerCPUTest, params_4D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_Planar_Blocked())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_Blocked, EltwiseLayerCPUTest, params_4D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_Blocked_Planar = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Blocked_Planar())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Planar())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Blocked_Planar, EltwiseLayerCPUTest, params_5D_Blocked_Planar, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_Planar_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_Planar_Blocked())),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Planar_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_Planar_Blocked_x64, EltwiseLayerCPUTest, params_5D_Planar_Blocked, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_constant_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Constant_mode_x64())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Constant_x64, EltwiseLayerCPUTest, params_4D_1D_constant_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_1D_parameter_mode = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_4D_1D())),
                ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_1D_Parameter_mode())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_1D_Parameter_x64, EltwiseLayerCPUTest, params_4D_1D_parameter_mode, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_constant = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_constant())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Constant_x64, EltwiseLayerCPUTest, params_5D_1D_constant, EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_1D_parameter = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(static_shapes_to_test_representation(inShapes_5D_1D())),
                ::testing::Values(ov::test::utils::EltwiseTypes::ADD, ov::test::utils::EltwiseTypes::MULTIPLY),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_1D_parameter())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_1D_Parameter_x64, EltwiseLayerCPUTest, params_5D_1D_parameter, EltwiseLayerCPUTest::getTestCaseName);

//// ============================================ 4D ============================================

const auto params_4D_planar_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_MemOrder_dyn_const_x64, EltwiseLayerCPUTest, params_4D_planar_dyn_const,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_per_channel_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_PerChannel())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_PerChannel_MemOrder_dyn_const_x64, EltwiseLayerCPUTest, params_4D_per_channel_dyn_const,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_blocked_blocked_dyn_const = ::testing::Combine(
        ::testing::Combine(
                ::testing::ValuesIn(inShapes_4D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Blocked_MemOrder_dyn_const_x64, EltwiseLayerCPUTest, params_4D_blocked_blocked_dyn_const,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_planar_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar())),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Planar_MemOrder_dyn_param_x64, EltwiseLayerCPUTest, params_4D_planar_dyn_param,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_perchannel_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_PerChannel())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_PerChannel_MemOrder_dyn_param_x64, EltwiseLayerCPUTest, params_4D_perchannel_dyn_param,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_blocked_blocked_dyn_param = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Blocked_Blocked_MemOrder_dyn_param_x64, EltwiseLayerCPUTest, params_4D_blocked_blocked_dyn_param,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_planar_dyn_param_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Planar())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::ValuesIn(enforceSnippets()));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_planar_dyn_param_fusing, EltwiseLayerCPUTest, params_4D_planar_dyn_param_fusing,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_perchannel_dyn_param_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_PerChannel())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_perchannel_dyn_param_fusing, EltwiseLayerCPUTest, params_4D_perchannel_dyn_param_fusing,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_dyn_param_fusing_Blocked_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_dyn_param_fusing_Blocked_Blocked, EltwiseLayerCPUTest, params_4D_dyn_param_fusing_Blocked_Blocked,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_4D_blocked_blocked_dyn_param_fusing = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_4D_dyn_param_fusing()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::Values(ElementType::f32),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_4D_Blocked_Blocked())),
        ::testing::ValuesIn(fusingParamsSet_x64),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_blocked_blocked_dyn_param_fusing, EltwiseLayerCPUTest, params_4D_blocked_blocked_dyn_param_fusing,
                         EltwiseLayerCPUTest::getTestCaseName);

//// ============================================ 5D ============================================

const auto params_5D_dyn_const_Blocked_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_const()),
                ::testing::ValuesIn(eltwiseOpTypesBinInp()),
                ::testing::Values(ov::test::utils::InputLayerType::CONSTANT),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_const_Blocked_Blocked, EltwiseLayerCPUTest, params_5D_dyn_const_Blocked_Blocked,
                         EltwiseLayerCPUTest::getTestCaseName);

const auto params_5D_dyn_param_Blocked_Blocked = ::testing::Combine(
        ::testing::Combine(
                ::testing::Values(inShapes_5D_dyn_param()),
                ::testing::ValuesIn(eltwiseOpTypesBinDyn()),
                ::testing::Values(ov::test::utils::InputLayerType::PARAMETER),
                ::testing::ValuesIn(opTypes()),
                ::testing::ValuesIn(netType()),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::element::undefined),
                ::testing::Values(ov::test::utils::DEVICE_CPU),
                ::testing::ValuesIn(additional_config())),
        ::testing::ValuesIn(filterCPUSpecificParams(cpuParams_5D_Blocked_Blocked())),
        ::testing::Values(emptyFusingSpec),
        ::testing::Values(false)); // CPU Plugin supports only planar layout for dynamic Subgraphs

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_5D_MemOrder_dyn_param_Blocked_Blocked, EltwiseLayerCPUTest, params_5D_dyn_param_Blocked_Blocked,
                         EltwiseLayerCPUTest::getTestCaseName);


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
        ::testing::ValuesIn({
            ov::test::utils::EltwiseTypes::BITWISE_AND,
            ov::test::utils::EltwiseTypes::BITWISE_OR,
            ov::test::utils::EltwiseTypes::BITWISE_XOR
        }),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
        ::testing::ValuesIn({
            CPUSpecificParams({ nhwc, nhwc }, { nhwc }, {}, {}),
            CPUSpecificParams({ nchw, nchw }, { nchw }, {}, {})
        }),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise, EltwiseLayerCPUTest, params_4D_bitwise, EltwiseLayerCPUTest::getTestCaseName);


const auto params_4D_bitwise_i32 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({
            ov::test::utils::EltwiseTypes::BITWISE_AND,
            ov::test::utils::EltwiseTypes::BITWISE_OR,
            ov::test::utils::EltwiseTypes::BITWISE_XOR
        }),
        ::testing::ValuesIn(secondaryInputTypes()),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i16, ov::element::Type_t::u16, ov::element::Type_t::u32 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({
        CPUSpecificParams({ nhwc, nhwc }, { nhwc }, {}, "*_I32"),
        CPUSpecificParams({ nchw, nchw }, { nchw }, {}, "*_I32")
    }),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_i32, EltwiseLayerCPUTest, params_4D_bitwise_i32, EltwiseLayerCPUTest::getTestCaseName);


const auto params_4D_bitwise_NOT = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ ov::test::utils::EltwiseTypes::BITWISE_NOT }),
        ::testing::ValuesIn({ ov::test::utils::InputLayerType::CONSTANT }),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i8, ov::element::Type_t::u8, ov::element::Type_t::i32 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({
        CPUSpecificParams({ nhwc }, { nhwc }, {}, {}),
        CPUSpecificParams({ nchw }, { nchw }, {}, {})
    }),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT, EltwiseLayerCPUTest, params_4D_bitwise_NOT, EltwiseLayerCPUTest::getTestCaseName);


const auto params_4D_bitwise_NOT_i32 = ::testing::Combine(
    ::testing::Combine(
        ::testing::ValuesIn(bitwise_in_shapes_4D),
        ::testing::ValuesIn({ ov::test::utils::EltwiseTypes::BITWISE_NOT }),
        ::testing::ValuesIn({ ov::test::utils::InputLayerType::CONSTANT }),
        ::testing::ValuesIn({ ov::test::utils::OpType::VECTOR }),
        ::testing::ValuesIn({ ov::element::Type_t::i16 }),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::element::Type_t::undefined),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values(ov::AnyMap())),
    ::testing::ValuesIn({
        CPUSpecificParams({ nhwc }, { nhwc }, {}, "*_I32"),
        CPUSpecificParams({ nchw }, { nchw }, {}, "*_I32")
    }),
    ::testing::Values(emptyFusingSpec),
    ::testing::Values(false));

INSTANTIATE_TEST_SUITE_P(smoke_CompareWithRefs_4D_Bitwise_NOT_i32, EltwiseLayerCPUTest, params_4D_bitwise_NOT_i32, EltwiseLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Eltwise
}  // namespace test
}  // namespace ov
