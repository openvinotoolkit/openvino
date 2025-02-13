// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Convolution {
OPENVINO_SUPPRESS_DEPRECATED_START
/* ============= Convolution (Gemm 1D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32,
                         ConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(convParams_ExplicitPadding_GEMM_1D(),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inShapesGemm1D()),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_1D())),
                                            ::testing::ValuesIn(fusingParamsSetWithEmpty()),
                                            ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapesPlain2Blocked3d = {
        {{}, {{ 1, 1, 7, 7, 7 }}},
        {{}, {{ 1, 2, 7, 7, 7 }}},
        {{}, {{ 1, 3, 7, 7, 7 }}},
        {
            //dynamic shapes
            { -1, 1, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 1, 7, 7, 7 },
                { 1, 1, 9, 9, 9 }
            }
        },
        {
            //dynamic shapes
            { -1, 3, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 3, 7, 7, 7 },
                { 1, 3, 9, 9, 9 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inShapesGemm2D_cache()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_FP32_dilated_empty_fusing,
                         ConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(convParams_ExplicitPadding_GEMM_2D_dilated(),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inShapesGemm2D()),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                            ::testing::Values(emptyFusingSpec),
                                            ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (2D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_FP32_empty_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes2d_cache()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_FP32_dilated_empty_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_2D_plain_to_blocked = {
        conv_avx2_plain_to_blocked_2D,
        conv_avx512_plain_to_blocked_2D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Convolution ============= */
const auto convParams_Reorder_2D = ::testing::Combine(
        ::testing::Values(ov::Shape{1, 1}),
        ::testing::Values(ov::Shape{2, 2}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(ov::Shape{1, 1}),
        ::testing::Values(64),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

std::vector<InputShape> inputShapes_Reorder_2D = {
        {
            // dynamic shape
            { -1, 32, -1, -1 },
            // target static shapes
            {
                { 1, 32, 39, 40 },
                { 2, 32, 20, 20 },
                { 1, 32, 39, 40 },
                { 2, 32, 20, 20 }
            }
        }
};

const std::vector<fusingSpecificParams> fusingParamsSet_reorder{
        emptyFusingSpec,
        fusingReluScaleShift,
        fusingAddPerChannel
};

INSTANTIATE_TEST_SUITE_P(smoke_reorder_Conv_2D, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Reorder_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes_Reorder_2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D_1x1})),
                                 ::testing::ValuesIn(fusingParamsSet_reorder),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_3D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_FP32_fusingScaleShiftAndFakeQuantizePerChannel, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_3D())),
                                 ::testing::Values(fusingScaleShiftAndFakeQuantizePerChannel),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_3D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_3D_plain_to_blocked = {
        conv_avx2_plain_to_blocked_3D,
        conv_avx512_plain_to_blocked_3D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (1D) ============= */

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_FP32_empty_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1x1_1D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_FP32_empty_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1x1_2D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution auto padding tests ============= */

const auto convParams_AutoPadding_2D = ::testing::Combine(
        ::testing::Values(kernels2d().front()),
        ::testing::ValuesIn(strides2d()),
        ::testing::ValuesIn(padBegins2d()),
        ::testing::ValuesIn(padEnds2d()),
        ::testing::ValuesIn(dilations2d()),
        ::testing::ValuesIn(numOutChannels()),
        ::testing::Values(ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_AutoPad_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_AutoPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Large Filter Test ============= */
namespace {

const size_t outChannels = 80;

const ov::Shape kernel = { 251 };
const ov::Shape stride = { 10 };
const std::vector<ptrdiff_t> padBegins = { 0 };
const std::vector<ptrdiff_t> padEnds = { 0 };
const ov::Shape dilations = { 1 };

const auto convParams_1D = ::testing::Combine(
        ::testing::Values(kernel),
        ::testing::Values(stride),
        ::testing::Values(padBegins),
        ::testing::Values(padEnds),
        ::testing::Values(dilations),
        ::testing::Values(outChannels),
        ::testing::Values(ov::op::PadType::EXPLICIT)
);

std::vector<InputShape> inShapes = {
    {{}, {{ 1, 1, 600 }}},
    {
        //dynamic shape
        { -1, 1, -1 },
        { //target static shapes
            { 1, 1, 600 },
            { 10, 1, 700 },
            { 1, 1, 600 }
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_Large_Filter, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::dynamic),
                                         ::testing::ValuesIn(inShapes),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

}  // namespace
}  // namespace Convolution
}  // namespace test
}  // namespace ov
