// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/binary_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(BinaryConvolutionLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};
const std::vector<std::vector<size_t>> kernels = {{3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 3}};
const std::vector<std::vector<size_t>> dilations = {{3, 1}};
const std::vector<size_t> numOutChannels = {5};
const std::vector<float> pad_values = {0, 1};

const auto binConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(pad_values));
const auto binConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::ValuesIn(pad_values));

INSTANTIATE_TEST_SUITE_P(
    smoke_BinaryConvolution2D_Serialization_ExplicitPadding, BinaryConvolutionLayerTest,
    ::testing::Combine(
        binConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    BinaryConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BinaryConvolution2D__Serialization_AutoPadValid, BinaryConvolutionLayerTest,
    ::testing::Combine(
        binConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    BinaryConvolutionLayerTest::getTestCaseName);
}  // namespace
