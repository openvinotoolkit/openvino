// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/deformable_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32, InferenceEngine::Precision::I16};

/* ============= 2D DeformableConvolution ============= */
const std::vector<std::vector<size_t>> deformable_vals = {{1, 18, 28, 28}};
const std::vector<std::vector<size_t>> kernels = {{1, 1, 3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds ={{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> groups = {1};
const std::vector<size_t> defor_groups = {1};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<size_t> multiple_defor_groups = {4};
const std::vector<std::vector<size_t>> deform_vals = {{1, 200, 220, 220}};
const std::vector<std::vector<size_t>> kernel = {{64, 4, 5, 5}};

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

const auto deformableConv2DParams_DeformableGroups_AutoPadExplicit = ::testing::Combine(
    ::testing::ValuesIn(deform_vals),
    ::testing::ValuesIn(kernel), ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(multiple_defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 30, 30})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    smoke_DeformableConvolution2D_DeformableGroups_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        deformableConv2DParams_DeformableGroups_AutoPadExplicit, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 4, 224, 224})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);
}  // namespace
