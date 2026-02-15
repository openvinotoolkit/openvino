// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "custom/single_layer_tests/classes/convolution.hpp"
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
}  // namespace
}  // namespace Convolution
}  // namespace test
}  // namespace ov
