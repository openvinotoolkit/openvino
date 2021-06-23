// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/binary_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32};

/* ============= 2D Binary Convolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padsBegin = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padsEnd = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<float> padValues = {0, 1};

const auto binConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padsBegin),
    ::testing::ValuesIn(padsEnd),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT),
    ::testing::ValuesIn(padValues));

const auto binConv2DParams_ValidPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels),
    ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID),
    ::testing::ValuesIn(padValues));

INSTANTIATE_TEST_SUITE_P(
    smoke_BinaryConvolution2D_ExplicitPadding, BinaryConvolutionLayerTest,
    ::testing::Combine(
        binConv2DParams_ExplicitPadding,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    BinaryConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_BinaryConvolution2D_AutoPadValid, BinaryConvolutionLayerTest,
    ::testing::Combine(
        binConv2DParams_ValidPadding,
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    BinaryConvolutionLayerTest::getTestCaseName);

}   // namespace
