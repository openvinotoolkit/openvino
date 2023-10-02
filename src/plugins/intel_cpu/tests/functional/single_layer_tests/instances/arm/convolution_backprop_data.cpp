// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_layer_tests/classes/convolution_backprop_data.hpp"
#include "shared_test_classes/single_layer/convolution_backprop_data.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;
using namespace ngraph::helpers;
using namespace ov::test;

namespace CPULayerTestsDefinitions {
namespace Deconvolution {

const std::vector<fusingSpecificParams> fusingParamsSet{
        emptyFusingSpec,
};

/* INSTANCES */
/* ============= Deconvolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Planar),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> Planar_2D_inputs_smoke = {
        DeconvInputData{
                InputShape{{}, {{ 2, 12, 7, 7 }}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 12, -1, -1}, {{ 1, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 7, 7}}},
                ngraph::helpers::InputLayerType::PARAMETER,
                {{15, 15}, {9, 10}, {15, 15}}
        }
};

const std::vector<DeconvInputData> Planar_2D_inputs_nightly = {
        DeconvInputData{
                InputShape{{-1, 12, -1, -1}, {{ 2, 12, 7, 7}, { 2, 12, 5, 7}, { 1, 12, 9, 4}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 12, 7, 7}, {{ 1, 12, 7, 7}, { 2, 12, 7, 7}, { 1, 12, 7, 7}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {{15, 15}}
        },
        DeconvInputData{
                InputShape{{{1, 10}, 12, 7, 7}, {{ 1, 12, 7, 7}, { 2, 12, 7, 7}, { 3, 12, 7, 7}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {{15, 15}}
        },
};

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfo({conv_gemm_acl_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Planar_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_Planar_2D,
        ::testing::ValuesIn(Planar_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfo({conv_gemm_acl_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Blocked 2D) ============= */
const std::vector<DeconvInputData> Blocked_2D_inputs_smoke = {
        DeconvInputData{
                InputShape{{}, {{ 2, 67, 7, 7 }}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
                ngraph::helpers::InputLayerType::PARAMETER,
                {{15, 15}, {9, 10}, {9, 9}}
        }
};


const auto convParams_ExplicitPadding_Blocked_2D_nightly = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        // Use 7x7 with stride 1 is too small to generate 15x15 output. It needs a big negative pad which will result
        //  avx512 kernel not to be selected.
        ::testing::ValuesIn({strides2d[1]}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const std::vector<DeconvInputData> Blocked_2D_inputs_nightly = {
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}, { 2, 67, 7, 7}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {{15, 15}}
        },
        DeconvInputData{
                InputShape{{ {1, 10}, 67, 7, 7}, {{ 2, 67, 7, 7}, { 3, 67, 7, 7}, { 1, 67, 7, 7}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {{15, 15}}
        }
};

const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

const auto convParams_ExplicitPadding_AMX_2D = ::testing::Combine(
        ::testing::ValuesIn(deconvAmxKernels2d),
        ::testing::ValuesIn(deconvAmxStrides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D, conv_avx2_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_INT8_AMX, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_AMX_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::i8),
        ::testing::ValuesIn({emptyFusingSpec, fusingClampRoundAddRelu}),
        ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D_nspc_amx})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Blocked_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_Blocked_2D_nightly,
        ::testing::ValuesIn(Blocked_2D_inputs_nightly),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D, conv_avx2_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */
const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(InferenceEngine::SizeVector({1, 1})),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_1x1_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        convParams_ExplicitPadding_1x1_2D,
        ::testing::ValuesIn(Blocked_2D_inputs_smoke),
        ::testing::Values(ElementType::f32),
        ::testing::ValuesIn(fusingParamsSet),
        ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D_1x1, conv_avx2_2D_1x1})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Deconvolution ============= */
INSTANTIATE_TEST_SUITE_P(smoke_reorder_Deconv_2D, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        ::testing::Combine(::testing::ValuesIn(kernels2d),
                           ::testing::Values(InferenceEngine::SizeVector{1, 1}),
                           ::testing::ValuesIn(padBegins2d),
                           ::testing::ValuesIn(padEnds2d),
                           ::testing::ValuesIn(dilations2d),
                           ::testing::ValuesIn(numOutChannels_Blocked),
                           ::testing::Values(ngraph::op::PadType::EXPLICIT),
                           ::testing::ValuesIn(emptyOutputPadding)),
        ::testing::Values(DeconvInputData{InputShape{{-1, 67, -1, -1}, {{ 1, 67, 7, 7}, { 1, 67, 9, 4}, { 1, 67, 5, 7}, { 1, 67, 7, 7}, { 1, 67, 9, 4}}},
                                          ngraph::helpers::InputLayerType::PARAMETER,
                                          {{15, 15}, {9, 9}, {9, 10}, {15, 15}, {9, 9}}}),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfo({conv_avx512_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution auto padding tests ============= */
const std::vector<DeconvInputData> inputs_2D_AutoPadding = {
        DeconvInputData{
                InputShape{{}, {{ 2, 67, 7, 7 }}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 1, 67, 9, 4}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {}
        },
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 2, 67, 7, 7}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
                ngraph::helpers::InputLayerType::CONSTANT,
                {{15, 15}}
        },
        DeconvInputData{
                InputShape{{-1, 67, -1, -1}, {{ 1, 67, 9, 4}, { 2, 67, 5, 7}, { 1, 67, 9, 4}}},
                ngraph::helpers::InputLayerType::PARAMETER,
                {{9, 9}, {9, 10}, {9, 9}}
        }
};

const auto deconvParams_AutoPadding_2D = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        ::testing::ValuesIn(strides2d),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ngraph::op::PadType::SAME_UPPER, ngraph::op::PadType::SAME_LOWER),
        ::testing::ValuesIn(emptyOutputPadding)
);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_AutoPadding_FP32, DeconvolutionLayerCPUTest,
        ::testing::Combine(
        deconvParams_AutoPadding_2D,
        ::testing::ValuesIn(inputs_2D_AutoPadding),
        ::testing::Values(ElementType::f32),
        ::testing::Values(emptyFusingSpec),
        ::testing::ValuesIn(filterCPUInfo({conv_gemm_acl_2D})),
        ::testing::Values(CPUTestUtils::cpuEmptyPluginConfig)),
        DeconvolutionLayerCPUTest::getTestCaseName);

const std::vector<DeconvInputData> inputs_3D_AutoPadding = {
        DeconvInputData{
                InputShape{{-1, 2, 4, {32, 64}, {32, 64}}, {{1, 2, 4, 32, 32}, {1, 2, 4, 40, 40}}},
                ngraph::helpers::InputLayerType::PARAMETER,
                {{8, 64, 64}, {8, 80, 80}}
        },
        DeconvInputData{
                InputShape{
                        {1, 64, 5, {1, std::numeric_limits<ov::Dimension::value_type>::max()}, {1, std::numeric_limits<ov::Dimension::value_type>::max()}},
                        {{1, 64, 5, 8, 8}}
                },
                ngraph::helpers::InputLayerType::CONSTANT,
                {{10, 16, 16}}
        },
};

} // namespace Deconvolution
} // namespace CPULayerTestsDefinitions
