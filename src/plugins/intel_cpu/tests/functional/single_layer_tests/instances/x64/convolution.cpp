// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/convolution.hpp"
#include "shared_test_classes/single_layer/convolution.hpp"
#include "test_utils/cpu_test_utils.hpp"
#include "test_utils/filter_cpu_info.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions  {
namespace Convolution {
namespace {

std::vector<CPUSpecificParams> filterCPUInfoForDevice_BF16(std::vector<CPUSpecificParams> allParams) {
    std::vector<CPUSpecificParams> specificParams;
    bool with_bf16 = with_cpu_x86_bfloat16();
    std::copy_if(allParams.begin(), allParams.end(), std::back_inserter(specificParams), [with_bf16](const CPUSpecificParams& item) {
        const auto &selected = std::get<3>(item);
        // when no bf16 hardware brgconv will not work
        if (!with_bf16 && selected.find("brgconv") != std::string::npos) {
            return false;
        }
        return true;
    });

    return filterCPUInfoForDevice(specificParams);
}

const std::vector<fusingSpecificParams> fusingParamsSetWithoutEmpty{
        // eltwise
        fusingRelu,
        fusingPRelu1DScaleShift,
        // depthwise
        fusingReluScaleShift,
        // fake quantize
        fusingFakeQuantizePerTensorRelu,
        fusingFakeQuantizePerChannelRelu,
        // sum
        fusingSumEluFQ,
        fusingSum,
        // bias
        fusingAddPerChannel
};

const std::vector<fusingSpecificParams> fusingParamsSetBF16{
        emptyFusingSpec,
        // eltwise
        fusingRelu,
        // depthwise
        fusingPRelu1DScaleShift,
        // sum
        fusingSum,
        // bias
        fusingAddPerChannel
};

/* ============= Convolution (Gemm 1D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_1D())),
                                 ::testing::ValuesIn(fusingParamsSetWithEmpty()),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32_ImproperPriorityList, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                             ::testing::Combine(
                                 convParams_ExplicitPadding_GEMM_1D(),
                                 ::testing::Values(ElementType::f32),
                                 ::testing::Values(ElementType::undefined),
                                 ::testing::Values(ElementType::undefined),
                                 ::testing::ValuesIn(inShapesGemm1D()),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfo({conv_gemm_1D})),
                             ::testing::Values(emptyFusingSpec),
                             ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_gemm_1D})), // todo: [AV] what about conv_gemm_1D_nspc?
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_1D())),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_FP32_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D_cache()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_FP32_dilated_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_2D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_FP32_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d_cache()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_FP32_dilated_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_2D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (1D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_FP32_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1x1_1D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_FP32_fusing, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1x1_2D())),
                                 ::testing::ValuesIn(fusingParamsSetWithoutEmpty),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (1D) ============= */
const auto convParams_ExplicitPadding_1D = ::testing::Combine(
        ::testing::ValuesIn(kernels1d()),
        ::testing::ValuesIn(strides1d()),
        ::testing::ValuesIn(padBegins1d()),
        ::testing::ValuesIn(padEnds1d()),
        ::testing::ValuesIn(dilations1d()),
        ::testing::ValuesIn(numOutChannels()),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1D = {
        conv_sse42_1D,
        conv_avx2_1D,
        conv_avx512_1D,
        conv_sse42_1D_nspc,
        conv_avx2_1D_nspc,
        conv_avx2_1D_nspc_brgconv,
        conv_avx512_1D_nspc,
        conv_avx512_1D_nspc_brgconv
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1D)),
                                 ::testing::ValuesIn(fusingParamsSetWithEmpty()),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_1D,
                                        conv_avx512_1D_nspc_brgconv, conv_avx512_1D_nspc_brgconv_amx})), // todo: [AV] what about conv_avx512_1D_nspc?
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_1D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_1D,
        conv_avx2_plain_to_blocked_1D,
        conv_avx512_plain_to_blocked_1D,
};

std::vector<InputShape> inputShapesPlain2Blocked1d = {
        {{}, {{1, 1, 7}}},
        {{}, {{1, 2, 7}}},
        {{}, {{1, 3, 7}}},
        {
        //dynamic shapes
            {-1, 1, {1, 200}},
            { //target static shapes
                {2, 1, 7},
                {1, 1, 9}
            }
        },
        {
        //dynamic shapes
            {-1, 3, {1, 200}},
            { //target static shapes
                {2, 3, 7},
                {1, 3, 9}
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_1D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_plain_to_blocked_1D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D, conv_avx512_2D_nspc,
                                        conv_avx512_2D_nspc_brgconv, conv_avx512_2D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D, conv_avx512_2D_nspc,
                                        conv_avx512_2D_nspc_brgconv, conv_avx512_2D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_1D_1x1, conv_avx512_2D_1x1_nspc,
                                        conv_avx512_1D_1x1_nspc_brgconv, conv_avx512_1D_1x1_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D_1x1, conv_avx512_2D_1x1_nspc,
                                        conv_avx512_2D_1x1_nspc_brgconv, conv_avx512_2D_1x1_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Jit Planar ============= */
/* ============= Convolution planar params (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_2D = {
        // sse42 is not supported
        conv_avx2_planar_2D,
        conv_avx512_planar_2D,
};

const auto convParams_Planar_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d()),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d()),
        ::testing::ValuesIn(padEnds2d()),
        ::testing::ValuesIn(dilations2d()),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d()),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d()),
        ::testing::ValuesIn(padEnds2d()),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_Jit_Planar_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_Jit_Planar_2D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_Jit_Planar_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_Jit_Planar_2D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (GEMM 3D) ============= */
INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution planar params (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_3D = {
        // sse42 is not supported
        conv_avx2_planar_3D,
        conv_avx512_planar_3D,
};

const auto convParams_Planar_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d()),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d()),
        ::testing::ValuesIn(padEnds3d()),
        ::testing::ValuesIn(dilations3d()),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d()),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d()),
        ::testing::ValuesIn(padEnds3d()),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_GEMM_3D())),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_3D, conv_avx512_3D_nspc,
                                        conv_avx512_3D_nspc_brgconv, conv_avx512_3D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated(),
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_3D, conv_avx512_3D_nspc,
                                        conv_avx512_3D_nspc_brgconv, conv_avx512_3D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_Jit_Planar_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_Jit_Planar_3D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_Jit_Planar_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d()),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfo(CPUParams_Jit_Planar_3D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

} // namespace
} // namespace Convolution
} // namespace CPULayerTestsDefinitions