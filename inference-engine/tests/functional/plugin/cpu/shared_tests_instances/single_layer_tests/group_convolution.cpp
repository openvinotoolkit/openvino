// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/group_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32};

/* ============= 1D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels1d = {{3}};
const std::vector<std::vector<size_t>> strides1d = {{1}};
const std::vector<std::vector<ptrdiff_t>> padBegins1d = {{0}};
const std::vector<std::vector<ptrdiff_t>> padEnds1d = {{0}};
const std::vector<std::vector<size_t>> dilations1d = {{1}};
const std::vector<size_t> numOutChannels1d = {8, 16};
const std::vector<size_t> numGroups1d = {2, 8};
const auto inputShapes1d = std::vector<size_t>({1, 16, 30});

const auto groupConv1DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d),
    ::testing::ValuesIn(padBegins1d), ::testing::ValuesIn(padEnds1d),
    ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
    ::testing::ValuesIn(numGroups1d),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv1DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels1d), ::testing::ValuesIn(strides1d),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::Values(std::vector<ptrdiff_t>({0})),
    ::testing::ValuesIn(dilations1d), ::testing::ValuesIn(numOutChannels1d),
    ::testing::ValuesIn(numGroups1d),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution1D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv1DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>(inputShapes1d)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution1D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv1DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 16, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};
const auto inputShapes = std::vector<size_t>({1, 16, 30, 30});

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>(inputShapes)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 16, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}};
const auto inputShapes3d = std::vector<size_t>({1, 4, 10, 10, 10});

const auto groupConv3DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(paddings3d),
    ::testing::ValuesIn(dilations3d), ::testing::Values(4),
    ::testing::Values(2), ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto groupConv3DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::ValuesIn(dilations3d), ::testing::Values(4),
    ::testing::Values(2), ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution3D_ExplicitPadding, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>(inputShapes3d)),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvolution3D_AutoPadValid, GroupConvolutionLayerTest,
    ::testing::Combine(
        groupConv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 4, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
