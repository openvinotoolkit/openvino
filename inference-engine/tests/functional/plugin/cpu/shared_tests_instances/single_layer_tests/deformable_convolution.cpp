// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/deformable_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16};

/* ============= 2D DeformableConvolution ============= */
const std::vector<std::vector<size_t>> deformable_vals = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> kernels = {{3, 3}, {3, 5}};
const std::vector<std::vector<size_t>> strides = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}, {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}, {0, 3}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}, {3, 1}};
const std::vector<size_t> groups = {1, 2};
const std::vector<size_t> defor_groups = {1, 2};
const std::vector<size_t> numOutChannels = {1, 5};

const auto deformableConv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(deformable_vals),
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto deformableConv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(deformable_vals),
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution2D_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution2D_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

/* ============= 3D DeformableConvolution ============= */
const std::vector<std::vector<size_t>> deformable_vals3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<size_t>> kernels3d = {{3, 3, 3}, {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}, {0, 2, 0}};
const std::vector<std::vector<size_t>> strides3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<std::vector<size_t>> dilations3d = {{1, 1, 1}, {1, 2, 1}};
const std::vector<size_t> numOutChannels3D = {1, 5};

const auto deformableConv3DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(deformable_vals3d),
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::ValuesIn(paddings3d), ::testing::ValuesIn(paddings3d),
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto deformableConv3DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(deformable_vals3d),
    ::testing::ValuesIn(kernels3d), ::testing::ValuesIn(strides3d),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
    ::testing::ValuesIn(dilations3d), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels3D),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution3D_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv3DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution3D_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv3DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 3, 10, 10, 10})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

}  // namespace
