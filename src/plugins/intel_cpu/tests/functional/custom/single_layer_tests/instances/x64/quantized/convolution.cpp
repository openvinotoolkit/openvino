// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution.hpp"

#include "common_test_utils/node_builders/constant.hpp"
#include "internal_properties.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "utils/cpu_test_utils.hpp"
#include "utils/filter_cpu_info.hpp"
#include "utils/fusing_test_utils.hpp"
#include "utils/quantization_utils.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {
namespace Convolution {
namespace {

ExtraOperationsParams prelu_u8i8f32 = {
    fusingPReluPerTensor,
    QuantizationInfo {
        {
            {0, QuantizationData{0, 8, 0, 8, 256}},
            {1, QuantizationData{-1, 1, -1, 1, 255}},
        },
    }
};

ExtraOperationsParams prelu_u8i8u8 = {
    fusingSigmoid,
    QuantizationInfo {
        {
            {0, QuantizationData{0, 8, 0, 8, 256}},
            {1, QuantizationData{-1, 1, -1, 1, 255}},
        },
        {
            {0, QuantizationData{0, 1, 0, 1, 256}},
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_Conv_1D_FP32_Quantization,
                         ConvolutionLayerCPUTest,
                         ::testing::Combine(::testing::Combine(convParams_ExplicitPadding_GEMM_1D(),
                                                               ::testing::Values(ElementType::f32),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::Values(ElementType::dynamic),
                                                               ::testing::ValuesIn(inShapesGemm1D()),
                                                               ::testing::Values(ov::test::utils::DEVICE_CPU)),
                                            ::testing::ValuesIn(filterCPUInfoVNNI({conv_avx2_1D_nspc,
                                                                                   conv_avx512_1D_nspc})),
                                            ::testing::ValuesIn({prelu_u8i8f32, prelu_u8i8u8}),
                                            ::testing::Values(empty_plugin_config)),
                         ConvolutionLayerCPUTest::getTestCaseName);

// Feed the Convolution output to a per-channel Multiply and a Relu, then add their results.
// Folding the Multiply into the Convolution's DQ scales would also scale the Relu input.
const auto fusingMultiplyPerChannelWithSecondConsumer = fusingSpecificParams{
    std::make_shared<postNodesMgr>(std::vector<postNodeBuilder>{
        {[](postNodeConfig& cfg) {
             auto scale = ov::test::utils::make_constant(cfg.type, generatePerChannelShape(cfg.input));
             auto multiply = std::make_shared<ov::op::v1::Multiply>(cfg.input, scale);
             auto relu = std::make_shared<ov::op::v0::Relu>(cfg.input);
             return std::make_shared<ov::op::v1::Add>(multiply, relu);
         },
         "Add(Multiply(PerChannel),Relu)"}}),
    {}};

const ExtraOperationsParams dqScalesMultipleConsumers{
    fusingMultiplyPerChannelWithSecondConsumer,
    QuantizationInfo{{{0, QuantizationData{0, 8, 256}}, {1, QuantizationData{-1, 1, 255}}}}};

INSTANTIATE_TEST_SUITE_P(
    smoke_Conv_1D_INT8_DQScalesMultipleConsumers,
    ConvolutionLayerCPUTest,
    ::testing::Combine(
        ::testing::Combine(::testing::Values(convSpecificParams{{1}, {1}, {0}, {0}, {1}, 8, ov::op::PadType::EXPLICIT}),
                           ::testing::Values(ElementType::f32),
                           ::testing::Values(ElementType::dynamic),
                           ::testing::Values(ElementType::dynamic),
                           ::testing::ValuesIn(inShapesGemm1D()),
                           ::testing::Values(ov::test::utils::DEVICE_CPU)),
        ::testing::ValuesIn(filterCPUInfoVNNI({conv_avx2_1D_nspc, conv_avx512_1D_nspc})),
        ::testing::Values(dqScalesMultipleConsumers),
        // Pin f32 inference precision across CPUs. Disable Snippets so the static case exercises DQ-scale fusion.
        ::testing::Values(ov::AnyMap{ov::hint::inference_precision(ov::element::f32),
                                     ov::intel_cpu::snippets_mode(ov::intel_cpu::SnippetsMode::DISABLE)})),
    ConvolutionLayerCPUTest::getTestCaseName);
}  // namespace
}  // namespace Convolution
}  // namespace test
}  // namespace ov
