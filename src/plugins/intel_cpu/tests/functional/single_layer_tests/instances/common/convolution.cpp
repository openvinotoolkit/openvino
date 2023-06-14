// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/convolution.hpp"
#include "shared_test_classes/single_layer/convolution.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;
using namespace ov::intel_cpu;

namespace CPULayerTestsDefinitions {
namespace Convolution {

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

/* COMMON PARAMS */
const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
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

/* ============= Convolution params (GEMM layout) ============= */
const SizeVector numOutChannels_Gemm = { 6 };

/* ============= Convolution params (blocked and nspc layout) ============= */
const SizeVector numOutChannels = { 64, 63 };

/* ============= Convolution params (1D) ============= */
const std::vector<SizeVector> kernels1d = { {3}, {1} };
const std::vector<SizeVector> strides1d = { {1}, {2} };
const std::vector<std::vector<ptrdiff_t>> padBegins1d = { {0}, {1} };
const std::vector<std::vector<ptrdiff_t>> padEnds1d = { {0} };
const std::vector<SizeVector> dilations1d = { {1}, {2} };
std::vector<InputShape> inputShapes1d = {
        {{}, {{ 2, 64, 7 }}},
        {{}, {{ 1, 67, 7 }}},
        {
            //dynamic shape
            { -1, 64, {1, 200} },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 9 }
            }
        },
        {
            //dynamic shape
            { -1, 67, {1, 200} },
            { //target static shapes
                { 2, 67, 7 },
                { 1, 67, 9 }
            }
        },
        {
            //dynamic shape
            { {1, 200}, 64, -1 },
            { //target static shapes
                { 2, 64, 7 },
                { 1, 64, 5 }
            }
        }
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

/* ============= Convolution params (2D) ============= */
const std::vector<SizeVector> kernels2d = { {3, 3}, {1, 1} };
const std::vector<SizeVector> strides2d = { {1, 1}, {2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins2d = { {0, 0}, {1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds2d = { {0, 0} };
const std::vector<SizeVector> dilations2d = { {1, 1} };

std::vector<InputShape> inputShapes2d = {
        {{}, {{ 1, 64, 7, 7 }}},
        {{}, {{ 1, 67, 7, 7 }}},
        {
            //dynamic shape
            { -1, 64, -1, {1, 200} },
            { //target static shapes
                { 2, 64, 7, 7 },
                { 1, 64, 9, 9}
            }
        },
        {
            //dynamic shape
            { -1, 67, -1, {1, 200} },
            { //target static shapes
                { 2, 67, 7, 7 },
                { 1, 67, 9, 9}
            }
        }
};

std::vector<InputShape> inputShapesPlain2Blocked2d = {
        {{}, {{ 1, 1, 7, 7 }}},
        {{}, {{ 1, 2, 7, 7 }}},
        {{}, {{ 1, 3, 7, 7 }}},
        {
            //dynamic shape
            { -1, 1, -1, {1, 200} },
            { //target static shapes
                { 2, 1, 7, 7 },
                { 1, 1, 9, 9}
            }
        },
        {
            //dynamic shape
            { -1, 3, -1, {1, 200} },
            { //target static shapes
                { 2, 3, 7, 7 },
                { 1, 3, 9, 9}
            }
        }
};

/* ============= Convolution params (3D) ============= */
const std::vector<SizeVector> kernels3d = { {3, 3, 3}, {1, 1, 1} };
const std::vector<SizeVector> strides3d = { {1, 1, 1}, {2, 2, 2} };
const std::vector<std::vector<ptrdiff_t>> padBegins3d = { {0, 0, 0}, {1, 1, 1} };
const std::vector<std::vector<ptrdiff_t>> padEnds3d = { {0, 0, 0} };
const std::vector<SizeVector> dilations3d = { {1, 1, 1} };
std::vector<InputShape> inputShapes3d = {
        {{}, {{ 1, 64, 7, 7, 7 }}},
        {{}, {{ 1, 67, 7, 7, 7 }}},
        {
            //dynamic shapes
            { -1, 64, -1, {1, 200}, -1 },
            { //target static shapes
                { 1, 64, 7, 7, 7 },
                { 1, 64, 9, 9, 9}
            }
        },
        {
            //dynamic shapes
            { -1, 67, -1, {1, 200}, -1 },
            { //target static shapes
                { 1, 67, 7, 7, 7 },
                { 1, 67, 9, 9, 9}
            }
        }
};
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
/* ============= */

/* INSTANCES */
/* ============= Convolution (Gemm 1D) ============= */
const auto convParams_ExplicitPadding_GEMM_1D = ::testing::Combine(
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_1D = {
        conv_gemm_1D,
        conv_gemm_1D_nspc
};

std::vector<InputShape> inShapesGemm1D = {
        {{}, {{ 2, 12, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, {1, 200} },
            { //target static shapes
                { 2, 12, 7 },
                { 1, 12, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

// Verify that even if primitive is missed in custom priority list there is still a fallback to the default priority list
const auto conv_gemm_1D_improperPriorityList = CPUSpecificParams{{ncw}, {ncw}, {"unknown"}, "jit_gemm"};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_FP32_ImproperPriorityList, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                             ::testing::Combine(
                                 convParams_ExplicitPadding_GEMM_1D,
                                 ::testing::Values(ElementType::f32),
                                 ::testing::Values(ElementType::undefined),
                                 ::testing::Values(ElementType::undefined),
                                 ::testing::ValuesIn(inShapesGemm1D),
                                 ::testing::Values(ov::test::utils::DEVICE_CPU)),
                             ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_1D})),
                             ::testing::Values(emptyFusingSpec),
                             ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_gemm_1D})), // todo: [AV] what about conv_gemm_1D_nspc?
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm1D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (Gemm 2D) ============= */
const auto convParams_ExplicitPadding_GEMM_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_GEMM_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_2D = {
        conv_gemm_2D,
        conv_gemm_2D_nspc
};

std::vector<InputShape> inShapesGemm2D = {
        {{}, {{ 2, 12, 7, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, -1, {1, 200} },
            { //target static shapes
                { 2, 12, 7, 7 },
                { 1, 12, 5, 5 }
            }
        }
};

std::vector<InputShape> inShapesGemm2D_cache = {
        {{}, {{ 2, 12, 7, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, -1, {1, 200} },
            { //target static shapes
                { 1, 12, 5, 5 },
                { 1, 12, 7, 7 },
                { 1, 12, 5, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D_cache),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_GEMM_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (GEMM 3D) ============= */
const auto convParams_ExplicitPadding_GEMM_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_GEMM_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::ValuesIn(numOutChannels_Gemm),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_GEMM_3D = {
        conv_gemm_3D,
        conv_gemm_3D_nspc
};

std::vector<InputShape> inShapesGemm3D = {
        {{}, {{ 2, 12, 7, 7, 7 }}},
        {
            //dynamic shape
            { {1, 200}, 12, -1, {1, 200}, -1 },
            { //target static shapes
                { 2, 12, 7, 7, 7 },
                { 1, 12, 5, 5, 5 }
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_GEMM_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_GEMM_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_GEMM_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapesGemm3D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_GEMM_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (1D) ============= */
const auto convParams_ExplicitPadding_1D = ::testing::Combine(
        ::testing::ValuesIn(kernels1d),
        ::testing::ValuesIn(strides1d),
        ::testing::ValuesIn(padBegins1d),
        ::testing::ValuesIn(padEnds1d),
        ::testing::ValuesIn(dilations1d),
        ::testing::ValuesIn(numOutChannels),
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
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
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
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_1D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_1D,
        conv_avx2_plain_to_blocked_1D,
        conv_avx512_plain_to_blocked_1D,
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
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1D_plain_to_blocked)),
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
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_1D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (2D) ============= */
const auto convParams_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_2D = {
        conv_sse42_2D,
        conv_avx2_2D,
        conv_avx512_2D,
        conv_sse42_2D_nspc,
        conv_avx2_2D_nspc,
        conv_avx2_2D_nspc_brgconv,
        conv_avx512_2D_nspc,
        conv_avx512_2D_nspc_brgconv
};

std::vector<InputShape> inputShapes2d_cache = {
        {{}, {{ 1, 64, 7, 7 }}},
        {{}, {{ 1, 67, 7, 7 }}},
        {
            //dynamic shape
            { -1, 64, -1, {1, 200} },
            { //target static shapes
                { 1, 64, 7, 7 },
                { 1, 64, 9, 9 },
                { 1, 64, 7, 7 }
            }
        },
        {
            //dynamic shape
            { -1, 67, -1, {1, 200} },
            { //target static shapes
                { 1, 67, 7, 7 },
                { 1, 67, 9, 9}
            }
        }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d_cache),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

std::vector<InputShape> inputShapes2d_dynBatch = {
        {
            //dynamic shape
            { {1, 10}, 64, 7, 7 },
            { //target static shapes
                { 2, 64, 7, 7 },
                { 1, 64, 7, 7 }
            }
        },
};

const std::vector<fusingSpecificParams> fusingParamsSet_dynBatch{
        emptyFusingSpec,
        fusingReluScaleShift,
        fusingSum,
        fusingAddPerChannel
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_FP32_dynBatch, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d_dynBatch),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::ValuesIn(fusingParamsSet_dynBatch),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D, conv_avx512_2D_nspc,
                                        conv_avx512_2D_nspc_brgconv, conv_avx512_2D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D, conv_avx512_2D_nspc,
                                        conv_avx512_2D_nspc_brgconv, conv_avx512_2D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_2D_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_2D_plain_to_blocked = {
        conv_sse42_plain_to_blocked_2D,
        conv_avx2_plain_to_blocked_2D,
        conv_avx512_plain_to_blocked_2D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_2D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_2D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_2D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Convolution ============= */
const auto convParams_Reorder_2D = ::testing::Combine(
        ::testing::Values(SizeVector{1, 1}),
        ::testing::Values(SizeVector{2, 2}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(std::vector<ptrdiff_t>{0, 0}),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::Values(64),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
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
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes_Reorder_2D),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1})),
                                 ::testing::ValuesIn(fusingParamsSet_reorder),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution (3D) ============= */
const auto convParams_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_ExplicitPadding_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_3D = {
        //conv_sse42_3D, // not supported jit_sse42 for 3d
        conv_avx2_3D,
        conv_avx512_3D,
        conv_avx2_3D_nspc,
        conv_avx2_3D_nspc_brgconv,
        conv_avx512_3D_nspc,
        conv_avx512_3D_nspc_brgconv
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_FP32_fusingScaleShiftAndFakeQuantizePerChannel, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::Values(fusingScaleShiftAndFakeQuantizePerChannel),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_3D, conv_avx512_3D_nspc,
                                        conv_avx512_3D_nspc_brgconv, conv_avx512_3D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_3D, conv_avx512_3D_nspc,
                                        conv_avx512_3D_nspc_brgconv, conv_avx512_3D_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_3D_I8_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

const std::vector<CPUSpecificParams> CPUParams_3D_plain_to_blocked = {
        conv_avx2_plain_to_blocked_3D,
        conv_avx512_plain_to_blocked_3D,
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_PlainToBlocked_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_FP32_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_3D_plain_to_blocked)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Conv_PlainToBlocked_3D_BF16_dilated, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_3D_dilated,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapesPlain2Blocked3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_plain_to_blocked_3D})),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (1D) ============= */

const auto convParams_ExplicitPadding_1x1_1D = ::testing::Combine(
        ::testing::Values(SizeVector({1})),
        ::testing::Values(SizeVector({1})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(std::vector<ptrdiff_t>({0})),
        ::testing::Values(SizeVector({1})),
        ::testing::Values(63),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_1D = {
        conv_sse42_1D_1x1,
        conv_avx2_1D_1x1,
        conv_avx512_1D_1x1,
        conv_sse42_1D_1x1_nspc,
        conv_avx2_1D_1x1_nspc,
        conv_avx2_1D_1x1_nspc_brgconv,
        conv_avx512_1D_1x1_nspc,
        conv_avx512_1D_1x1_nspc_brgconv
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_1D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_1D_1x1, conv_avx512_2D_1x1_nspc,
                                        conv_avx512_1D_1x1_nspc_brgconv, conv_avx512_1D_1x1_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_1x1_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_1D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes1d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_1D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */

const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(SizeVector({1, 1})),
        ::testing::Values(63),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const std::vector<CPUSpecificParams> CPUParams_1x1_2D = {
        conv_sse42_2D_1x1,
        conv_avx2_2D_1x1,
        conv_avx512_2D_1x1,
        conv_sse42_2D_1x1_nspc,
        conv_avx2_2D_1x1_nspc,
        conv_avx2_2D_1x1_nspc_brgconv,
        conv_avx512_2D_1x1_nspc,
        conv_avx512_2D_1x1_nspc_brgconv
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
                                 ::testing::ValuesIn(fusingParamsSet),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_BF16, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice_BF16({conv_avx512_2D_1x1, conv_avx512_2D_1x1_nspc,
                                        conv_avx512_2D_1x1_nspc_brgconv, conv_avx512_2D_1x1_nspc_brgconv_amx})),
                                 ::testing::ValuesIn(fusingParamsSetBF16),
                                 ::testing::Values(cpuBF16PluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_1x1_I8, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_ExplicitPadding_1x1_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::i8),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_1x1_2D)),
                                 ::testing::Values(fusingSum),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Jit Planar ============= */

/* ============= Convolution planar params (2D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_2D = {
        // sse42 is not supported
        conv_avx2_planar_2D,
        conv_avx512_planar_2D,
};

const auto convParams_Planar_ExplicitPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_2D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::Values(SizeVector{1, 1}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
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
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_2D)),
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
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_2D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution planar params (3D) ============= */
const std::vector<CPUSpecificParams> CPUParams_Jit_Planar_3D = {
        // sse42 is not supported
        conv_avx2_planar_3D,
        conv_avx512_planar_3D,
};

const auto convParams_Planar_ExplicitPadding_3D = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

const auto convParams_Planar_ExplicitPadding_3D_dilated = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::Values(SizeVector{1, 1, 1}),
        ::testing::ValuesIn(padBegins3d),
        ::testing::ValuesIn(padEnds3d),
        ::testing::Values(SizeVector{2, 2, 2}),
        ::testing::Values(1),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_3D_Jit_Planar_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_Planar_ExplicitPadding_3D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_3D)),
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
                                         ::testing::ValuesIn(inputShapes3d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_Jit_Planar_3D)),
                                 ::testing::Values(emptyFusingSpec, fusingRelu),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Convolution auto padding tests ============= */

const auto convParams_AutoPadding_2D = ::testing::Combine(
        ::testing::Values(kernels2d.front()),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER)
);

INSTANTIATE_TEST_SUITE_P(smoke_Conv_2D_AutoPad_FP32, ConvolutionLayerCPUTest,
                         ::testing::Combine(
                                 ::testing::Combine(
                                         convParams_AutoPadding_2D,
                                         ::testing::Values(ElementType::f32),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inputShapes2d),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams_2D)),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

/* ============= Large Filter Test ============= */
namespace {

const size_t outChannels = 80;

const SizeVector kernel = { 251 };
const SizeVector stride = { 10 };
const std::vector<ptrdiff_t> padBegins = { 0 };
const std::vector<ptrdiff_t> padEnds = { 0 };
const SizeVector dilations = { 1 };

const auto convParams_1D = ::testing::Combine(
        ::testing::Values(kernel),
        ::testing::Values(stride),
        ::testing::Values(padBegins),
        ::testing::Values(padEnds),
        ::testing::Values(dilations),
        ::testing::Values(outChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
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
                                         ::testing::Values(ElementType::undefined),
                                         ::testing::ValuesIn(inShapes),
                                         ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                 ::testing::Values(CPUSpecificParams{{}, {}, {}, CPUTestsBase::any_type}),
                                 ::testing::Values(emptyFusingSpec),
                                 ::testing::Values(cpuEmptyPluginConfig)),
                         ConvolutionLayerCPUTest::getTestCaseName);

} // namespace
} // namespace Convolution
} // namespace CPULayerTestsDefinitions