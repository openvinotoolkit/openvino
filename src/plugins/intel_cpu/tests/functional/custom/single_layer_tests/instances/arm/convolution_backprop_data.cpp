// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution_backprop_data.hpp"
#include "shared_test_classes/single_op/convolution_backprop_data.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"

using namespace CPUTestUtils;
using namespace ov::test;

namespace {

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

const std::vector<DeconvInputData> Planar_2D_inputs_smoke = {
        DeconvInputData{InputShape{{}, {{2, 12, 7, 7}}}, ov::test::utils::InputLayerType::CONSTANT, {}},
        DeconvInputData{InputShape{{-1, 12, -1, -1}, {{1, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::PARAMETER,
                        {}}};

const std::vector<DeconvInputData> Planar_2D_inputs_nightly = {
        DeconvInputData{InputShape{{-1, 12, -1, -1}, {{2, 12, 7, 7}, {2, 12, 5, 7}, {1, 12, 9, 4}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{-1, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {1, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
        DeconvInputData{InputShape{{{1, 10}, 12, 7, 7}, {{1, 12, 7, 7}, {2, 12, 7, 7}, {3, 12, 7, 7}}},
                        ov::test::utils::InputLayerType::CONSTANT,
                        {}},
};

INSTANTIATE_TEST_SUITE_P(smoke_arm_Deconv_2D_Planar_FP16,
                         DeconvolutionLayerCPUTest,
                         ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_smoke),
                                            ::testing::Values(ElementType::f16),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfo({conv_gemm_2D, conv_gemm_acl_2D, conv_gemm_acl_2D_nspc})),
                                            ::testing::Values(cpu_f16_plugin_config)),
                         DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_arm_Deconv_2D_Planar_FP32,
                         DeconvolutionLayerCPUTest,
                         ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_smoke),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfo({conv_gemm_2D, conv_gemm_acl_2D, conv_gemm_acl_2D_nspc})),
                                            ::testing::Values(empty_plugin_config)),
                         DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_arm_Deconv_2D_Planar_FP16,
                         DeconvolutionLayerCPUTest,
                         ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f16),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfo({conv_gemm_2D, conv_gemm_acl_2D, conv_gemm_acl_2D_nspc})),
                                            ::testing::Values(cpu_f16_plugin_config)),
                         DeconvolutionLayerCPUTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_arm_Deconv_2D_Planar_FP32,
                         DeconvolutionLayerCPUTest,
                         ::testing::Combine(convParams_ExplicitPadding_Planar_2D,
                                            ::testing::ValuesIn(Planar_2D_inputs_nightly),
                                            ::testing::Values(ElementType::f32),
                                            ::testing::ValuesIn(fusingParamsSet),
                                            ::testing::ValuesIn(filterCPUInfo({conv_gemm_2D, conv_gemm_acl_2D, conv_gemm_acl_2D_nspc})),
                                            ::testing::Values(empty_plugin_config)),
                         DeconvolutionLayerCPUTest::getTestCaseName);
} // namespace
