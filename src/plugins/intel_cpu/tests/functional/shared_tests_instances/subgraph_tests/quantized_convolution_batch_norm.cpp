// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/quantized_convolution_batch_norm.hpp"

using namespace ov::test;

namespace {

INSTANTIATE_TEST_SUITE_P(
    smoke_QuantizedConvolutionBatchNorm,
    QuantizedConvolutionBatchNorm,
    ::testing::Combine(::testing::ValuesIn({ConvType::CONVOLUTION, ConvType::CONVOLUTION_BACKPROP}),
                       ::testing::ValuesIn({QuantizeType::FAKE_QUANTIZE,
                                            QuantizeType::QUANTIZE_DEQUANTIZE,
                                            QuantizeType::COMPRESSED_WEIGHTS}),
                       ::testing::ValuesIn({IntervalsType::PER_TENSOR, IntervalsType::PER_CHANNEL}),
                       ::testing::Values(false),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    QuantizedConvolutionBatchNorm::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_QuantizedConvolutionBatchNormTransposeOnWeights,
    QuantizedConvolutionBatchNorm,
    ::testing::Combine(::testing::ValuesIn({ConvType::CONVOLUTION, ConvType::CONVOLUTION_BACKPROP}),
                       ::testing::ValuesIn({QuantizeType::FAKE_QUANTIZE}),
                       ::testing::ValuesIn({IntervalsType::PER_TENSOR, IntervalsType::PER_CHANNEL}),
                       ::testing::Values(true),
                       ::testing::Values(ov::test::utils::DEVICE_CPU)),
    QuantizedConvolutionBatchNorm::getTestCaseName);

}  // namespace
