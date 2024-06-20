// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution_backprop_data.hpp"
#include "shared_test_classes/single_op/convolution_backprop_data.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace ov {
namespace test {
namespace DeConvolution {
/* COMMON PARAMS */
const std::vector<fusingSpecificParams> brgDeconvFusingParamsSet{
    emptyFusingSpec,
    //bias fusing
    fusingAddPerChannel,
    fusingMultiplyPerChannel
};

/* INSTANCES */
/* ============= Deconvolution (Planar 2D) ============= */
const auto convParams_ExplicitPadding_Planar_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                     ::testing::ValuesIn(strides2d),
                                                                     ::testing::ValuesIn(padBegins2d),
                                                                     ::testing::ValuesIn(padEnds2d),
                                                                     ::testing::ValuesIn(dilations2d),
                                                                     ::testing::ValuesIn(numOutChannels_Planar),
                                                                     ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                     ::testing::ValuesIn(emptyOutputPadding));
const std::vector<DeconvInputData> Planar_2D_inputs_nightly = {
        DeconvInputData{InputShape{{-1, 12, -1, -1}, {{2, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 9, 4}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{-1, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {1, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15}}},
        DeconvInputData{InputShape{{{1, 10}, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {3, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15}}},
        DeconvInputData{InputShape{{}, {{2, 12, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 12, -1, -1}, {{1, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{15, 15}, {9, 10}, {15, 15}}}
};

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Planar_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                           ::testing::ValuesIn(Planar_2D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Planar_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                           ::testing::ValuesIn(Planar_2D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution (Planar 3D) ============= */
const std::vector<DeconvInputData> Planar_3D_inputs_nightly = {
        DeconvInputData{
                // -1 will result deconv use 64 to infer output shape, for 3d output shape is too big for gemm bwd kernel
                //  to buffer the intermedia results
                InputShape{{-1, 12, {5, 9}, {4, 7}, {7, 9}}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {}},
        DeconvInputData{
                InputShape{{-1, 12, -1, -1, -1}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}, {2, 12, 7, 7, 7}}},
                ov::test::utils::InputLayerType::CONSTANT,
                {{10, 16, 16}}},
        DeconvInputData{InputShape{{{1, 10}, 12, 7, 7, 7}, {{2, 12, 7, 7, 7}, {1, 12, 7, 7, 7}, {3, 12, 7, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15, 15}}},
        DeconvInputData{InputShape{{}, {{2, 12, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 12, -1, -1, -1}, {{2, 12, 7, 7, 7}, {2, 12, 5, 7, 7}, {1, 12, 9, 4, 9}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{15, 15, 15}, {9, 10, 10}, {9, 9, 9}}}
};

const auto convParams_ExplicitPadding_Planar_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                     ::testing::ValuesIn(strides3d),
                                                                     ::testing::ValuesIn(padBegins3d),
                                                                     ::testing::ValuesIn(padEnds3d),
                                                                     ::testing::ValuesIn(dilations3d),
                                                                     ::testing::ValuesIn(numOutChannels_Planar),
                                                                     ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                     ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Planar_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Planar_3D,
                           ::testing::ValuesIn(Planar_3D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_3D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Planar_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Planar_3D,
                           ::testing::ValuesIn(Planar_3D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_3D})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution 2D ============= */
const std::vector<DeconvInputData> smoke_2D_inputs = {
        DeconvInputData{InputShape{{}, {{2, 67, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{2, 67, 7, 7}, {2, 67, 10, 10}, {1, 67, 9, 9}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{15, 15}, {9, 10}, {9, 9}}}};

const auto convParams_ExplicitPadding_Blocked_2D_nightly = ::testing::Combine(
        ::testing::ValuesIn(kernels2d),
        // Use 7x7 with stride 1 is too small to generate 15x15 output. It needs a big negative pad which will result
        //  avx512 kernel not to be selected.
        ::testing::ValuesIn({strides2d[1]}),
        ::testing::ValuesIn(padBegins2d),
        ::testing::ValuesIn(padEnds2d),
        ::testing::ValuesIn(dilations2d),
        ::testing::ValuesIn(numOutChannels_Blocked),
        ::testing::Values(ov::op::PadType::EXPLICIT),
        ::testing::ValuesIn(emptyOutputPadding));

const std::vector<DeconvInputData> Blocked_2D_inputs_nightly = {
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{2, 67, 7, 7}, {2, 67, 5, 7}, {1, 67, 9, 4}, {2, 67, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{2, 67, 7, 7}, {2, 67, 5, 7}, {1, 67, 9, 4}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15}}},
        DeconvInputData{InputShape{{{1, 10}, 67, 7, 7}, {{2, 67, 7, 7}, {3, 67, 7, 7}, {1, 67, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15}}},
        DeconvInputData{InputShape{{}, {{2, 67, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{2, 67, 7, 7}, {2, 67, 5, 7}, {1, 67, 9, 4}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{15, 15}, {9, 10}, {9, 9}}}};

const auto convParams_ExplicitPadding_Blocked_2D = ::testing::Combine(::testing::ValuesIn(kernels2d),
                                                                      ::testing::ValuesIn(strides2d),
                                                                      ::testing::ValuesIn(padBegins2d),
                                                                      ::testing::ValuesIn(padEnds2d),
                                                                      ::testing::ValuesIn(dilations2d),
                                                                      ::testing::ValuesIn(numOutChannels_Blocked),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                      ::testing::ValuesIn(emptyOutputPadding));

const auto convParams_ExplicitPadding_NSPC_2D = ::testing::Combine(::testing::ValuesIn(deconvBrgKernels2d),
                                                                  ::testing::ValuesIn(deconvBrgStrides2d),
                                                                  ::testing::ValuesIn(padBegins2d),
                                                                  ::testing::ValuesIn(padEnds2d),
                                                                  ::testing::ValuesIn(dilations2d),
                                                                  ::testing::ValuesIn(numOutChannels_Blocked),
                                                                  ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                  ::testing::ValuesIn(emptyOutputPadding));


const auto convParams_ExplicitPadding_NSPC_2D_1x1 = ::testing::Combine(::testing::ValuesIn(deconvBrgKernels2d_1x1),
                                                                  ::testing::ValuesIn(deconvBrgStrides2d),
                                                                  ::testing::ValuesIn(padBegins2d),
                                                                  ::testing::ValuesIn(padEnds2d),
                                                                  ::testing::ValuesIn(dilations2d),
                                                                  ::testing::ValuesIn(numOutChannels_Blocked),
                                                                  ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                  ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Blocked_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Blocked_2D,
                           ::testing::ValuesIn(Blocked_2D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D, block8c_2D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_Blocked_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Blocked_2D,
                           ::testing::ValuesIn(Blocked_2D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_BF16_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_FP16_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDeviceWithFP16({conv_avx512_2D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_f16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_BF16_AMX_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D_1x1,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1_nspc_brgconv_amx})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_FP16_AMX_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D_1x1,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDeviceWithFP16({conv_avx512_2D_1x1_nspc_brgconv_amx})),
                           ::testing::Values(cpu_f16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_BRG,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv, conv_avx2_2D_nspc_brgconv})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_BRG_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D_1x1,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_1x1_nspc_brgconv,
                                                                        conv_avx2_2D_1x1_nspc_brgconv})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_2D_NSPC_INT8_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::i8),
                           ::testing::ValuesIn({emptyFusingSpec, fusingClampRoundAddRelu}),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_amx})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution 3D ============= */
const std::vector<DeconvInputData> smoke_3D_inputs = {
        DeconvInputData{InputShape{{}, {{2, 35, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 35, -1, -1, -1}, {{1, 35, 5, 5, 5}, {2, 35, 5, 7, 5}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{7, 7, 7}, {7, 9, 7}}}};

const auto convParams_ExplicitPadding_Blocked_3D_nightly =
        ::testing::Combine(::testing::ValuesIn(kernels3d),
                           ::testing::ValuesIn({strides3d[0]}),
                           ::testing::ValuesIn(padBegins3d),
                           ::testing::ValuesIn(padEnds3d),
                           ::testing::ValuesIn(dilations3d),
                           ::testing::Values(32),
                           ::testing::Values(ov::op::PadType::EXPLICIT),
                           ::testing::ValuesIn(emptyOutputPadding));

const std::vector<DeconvInputData> Blocked_3D_inputs_nightly = {
        DeconvInputData{InputShape{{-1, 35, -1, -1, -1}, {{1, 35, 5, 5, 5}, {2, 35, 5, 7, 5}, {1, 35, 5, 5, 5}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{-1, 35, -1, -1, -1}, {{1, 35, 5, 5, 5}, {2, 35, 5, 7, 5}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{7, 7, 7}}},
        DeconvInputData{InputShape{{{1, 10}, 35, 5, 5, 5}, {{1, 35, 5, 5, 5}, {2, 35, 5, 5, 5}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{7, 7, 7}}},
        DeconvInputData{InputShape{{}, {{2, 35, 7, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 35, -1, -1, -1}, {{1, 35, 5, 5, 5}, {2, 35, 5, 7, 5}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{7, 7, 7}, {7, 9, 7}}}
};

const auto convParams_ExplicitPadding_Blocked_3D = ::testing::Combine(::testing::ValuesIn(kernels3d),
                                                                      ::testing::ValuesIn(strides3d),
                                                                      ::testing::ValuesIn(padBegins3d),
                                                                      ::testing::ValuesIn(padEnds3d),
                                                                      ::testing::ValuesIn(dilations3d),
                                                                      ::testing::Values(32),
                                                                      ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                      ::testing::ValuesIn(emptyOutputPadding));

const auto convParams_ExplicitPadding_NSPC_3D = ::testing::Combine(::testing::ValuesIn(deconvBrgKernels3d),
                                                                  ::testing::ValuesIn(deconvBrgStrides3d),
                                                                  ::testing::ValuesIn(padBegins3d),
                                                                  ::testing::ValuesIn(padEnds3d),
                                                                  ::testing::ValuesIn(dilations3d),
                                                                  ::testing::Values(32),
                                                                  ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                  ::testing::ValuesIn(emptyOutputPadding));

const auto convParams_ExplicitPadding_NSPC_3D_1x1 = ::testing::Combine(::testing::ValuesIn(deconvBrgKernels3d_1x1),
                                                                  ::testing::ValuesIn(deconvBrgStrides3d),
                                                                  ::testing::ValuesIn(padBegins3d),
                                                                  ::testing::ValuesIn(padEnds3d),
                                                                  ::testing::ValuesIn(dilations3d),
                                                                  ::testing::Values(32),
                                                                  ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                  ::testing::ValuesIn(emptyOutputPadding));
INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Blocked_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Blocked_3D,
                           ::testing::ValuesIn(Blocked_3D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_3D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_Blocked_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_Blocked_3D,
                           ::testing::ValuesIn(Blocked_3D_inputs_nightly),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_3D})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_BF16_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_FP16_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDeviceWithFP16({conv_avx512_3D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_f16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_BF16_AMX_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D_1x1,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_1x1_nspc_brgconv_amx})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_FP16_AMX_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D_1x1,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDeviceWithFP16({conv_avx512_3D_1x1_nspc_brgconv_amx})),
                           ::testing::Values(cpu_f16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_BRG,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_brgconv, conv_avx2_3D_nspc_brgconv})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_BRG_1x1,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D_1x1,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(brgDeconvFusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_1x1_nspc_brgconv,
                                                                        conv_avx2_3D_1x1_nspc_brgconv})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Deconv_3D_NSPC_INT8_AMX,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_NSPC_3D,
                           ::testing::ValuesIn(smoke_3D_inputs),
                           ::testing::Values(ElementType::i8),
                           ::testing::ValuesIn({emptyFusingSpec, fusingClampRoundAddRelu}),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_3D_nspc_amx})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Kernel_1x1 (2D) ============= */
const auto convParams_ExplicitPadding_1x1_2D = ::testing::Combine(::testing::Values(std::vector<size_t>({1, 1})),
                                                                  ::testing::Values(std::vector<size_t>({1, 1})),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::Values(std::vector<size_t>({1, 1})),
                                                                  ::testing::ValuesIn(numOutChannels_Blocked),
                                                                  ::testing::Values(ov::op::PadType::EXPLICIT),
                                                                  ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_1x1_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_1x1_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D,
                                                                       block8c_2D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_1x1_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(convParams_ExplicitPadding_1x1_2D,
                           ::testing::ValuesIn(smoke_2D_inputs),
                           ::testing::Values(ElementType::f32),
                           ::testing::ValuesIn(fusingParamsSet),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D,
                                                                       block8c_2D})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Reorder + Deconvolution ============= */
INSTANTIATE_TEST_SUITE_P(
        nightly_reorder_Deconv_2D,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(::testing::Combine(::testing::ValuesIn(kernels2d),
                                              ::testing::Values(std::vector<size_t>{1, 1}),
                                              ::testing::ValuesIn(padBegins2d),
                                              ::testing::ValuesIn(padEnds2d),
                                              ::testing::ValuesIn(dilations2d),
                                              ::testing::ValuesIn(numOutChannels_Blocked),
                                              ::testing::Values(ov::op::PadType::EXPLICIT),
                                              ::testing::ValuesIn(emptyOutputPadding)),
                           ::testing::Values(DeconvInputData{
                                   InputShape{{-1, 67, -1, -1},
                                              {{1, 67, 7, 7}, {1, 67, 9, 4}, {1, 67, 5, 7}, {1, 67, 7, 7}, {1, 67, 9, 4}}},
                                   ov::test::utils::InputLayerType::PARAMETER,
                                   {{15, 15}, {9, 9}, {9, 10}, {15, 15}, {9, 9}}}),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(emptyFusingSpec),
                           ::testing::ValuesIn(filterCPUInfoForDevice({block16c_2D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

/* ============= Deconvolution auto padding tests ============= */
const std::vector<DeconvInputData> inputs_2D_AutoPadding = {
        DeconvInputData{InputShape{{}, {{2, 67, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{1, 67, 9, 4}, {2, 67, 5, 7}, {1, 67, 9, 4}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{2, 67, 7, 7}, {2, 67, 5, 7}, {1, 67, 9, 4}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{15, 15}}},
        DeconvInputData{InputShape{{-1, 67, -1, -1}, {{1, 67, 9, 4}, {2, 67, 5, 7}, {1, 67, 9, 4}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{9, 9}, {9, 10}, {9, 9}}}};

const auto deconvParams_AutoPadding_2D =
        ::testing::Combine(::testing::ValuesIn(kernels2d),
                           ::testing::ValuesIn(strides2d),
                           ::testing::ValuesIn(padBegins2d),
                           ::testing::ValuesIn(padEnds2d),
                           ::testing::ValuesIn(dilations2d),
                           ::testing::ValuesIn(numOutChannels_Blocked),
                           ::testing::Values(ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER),
                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_2D_AutoPadding_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(deconvParams_AutoPadding_2D,
                           ::testing::ValuesIn(inputs_2D_AutoPadding),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(emptyFusingSpec),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_2D, block16c_2D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

const std::vector<DeconvInputData> inputs_3D_AutoPadding = {
        DeconvInputData{InputShape{{-1, 2, 4, {32, 64}, {32, 64}}, {{1, 2, 4, 32, 32}, {1, 2, 4, 40, 40}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {{8, 64, 64}, {8, 80, 80}}},
        DeconvInputData{InputShape{{1,
                                           64,
                                           5,
                                           {1, std::numeric_limits<ov::Dimension::value_type>::max()},
                                           {1, std::numeric_limits<ov::Dimension::value_type>::max()}},
                                   {{1, 64, 5, 8, 8}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {{10, 16, 16}}},
};

const auto deconvParams_AutoPadding_3D =
        ::testing::Combine(::testing::Values(kernels3d[0]),
                           ::testing::Values(strides3d[1]),
                           ::testing::ValuesIn(padBegins3d),
                           ::testing::ValuesIn(padEnds3d),
                           ::testing::ValuesIn(dilations3d),
                           ::testing::Values(1),
                           ::testing::Values(ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER),
                           ::testing::ValuesIn(emptyOutputPadding));

INSTANTIATE_TEST_SUITE_P(nightly_Deconv_3D_AutoPadding_FP32,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(deconvParams_AutoPadding_3D,
                           ::testing::ValuesIn(inputs_3D_AutoPadding),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(emptyFusingSpec),
                           ::testing::ValuesIn(filterCPUInfoForDevice({planar_3D, block16c_3D})),
                           ::testing::Values(CPUTestUtils::empty_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

const auto deconvParams_AutoPadding_2D_AMX =
        ::testing::Combine(::testing::ValuesIn(deconvBrgKernels2d),
                           ::testing::ValuesIn(deconvBrgStrides2d),
                           ::testing::ValuesIn(padBegins2d),
                           ::testing::ValuesIn(padEnds2d),
                           ::testing::ValuesIn(dilations2d),
                           ::testing::Values(256),
                           ::testing::Values(ov::op::PadType::SAME_UPPER, ov::op::PadType::SAME_LOWER),
                           ::testing::ValuesIn(emptyOutputPadding));

const DeconvInputData inputs_2D_AutoPadding_AMX = {InputShape{{-1, 512, -1, -1}, {{1, 512, 32, 51}, {1, 512, 68, 101}}},
                                                   ov::test::utils::InputLayerType::PARAMETER,
                                                   {{64, 101}, {135, 202}}};

INSTANTIATE_TEST_SUITE_P(
        smoke_Deconv_2D_AutoPadding_AMX_BF16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(deconvParams_AutoPadding_2D_AMX,
                           ::testing::Values(inputs_2D_AutoPadding_AMX),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(emptyFusingSpec),
                           ::testing::ValuesIn(filterCPUInfoForDevice({conv_avx512_2D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_bf16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        smoke_Deconv_2D_AutoPadding_AMX_FP16,
        DeconvolutionLayerCPUTest,
        ::testing::Combine(deconvParams_AutoPadding_2D_AMX,
                           ::testing::Values(inputs_2D_AutoPadding_AMX),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(emptyFusingSpec),
                           ::testing::ValuesIn(filterCPUInfoForDeviceWithFP16({conv_avx512_2D_nspc_brgconv_amx})),
                           ::testing::Values(cpu_f16_plugin_config)),
        DeconvolutionLayerCPUTest::getTestCaseName);

}  // namespace DeConvolution
}  // namespace test
}  // namespace ov
