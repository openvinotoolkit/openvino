// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(ConvolutionLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I16, InferenceEngine::Precision::I32,
    InferenceEngine::Precision::I64};
const std::vector<std::vector<size_t>> kernels = {{3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 3}};
const std::vector<std::vector<size_t>> dilations = {{3, 1}};
const std::vector<size_t> numOutChannels = {5};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_Convolution2D_Serialization_ExplicitPadding, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_Convolution2D__Serialization_AutoPadValid, ConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConvolutionLayerTest::getTestCaseName);
}  // namespace
