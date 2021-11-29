// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {
TEST_P(DeformableConvolutionLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32, InferenceEngine::Precision::FP16,
    InferenceEngine::Precision::I32, InferenceEngine::Precision::I16};
const std::vector<std::vector<size_t>> offsets = {{1, 18, 26, 26}};
const std::vector<std::vector<size_t>> filters = {{1, 1, 3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds ={{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> groups = {1};
const std::vector<size_t> defor_groups = {1};
const std::vector<size_t> numOutChannels = {1};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
    ::testing::ValuesIn(offsets), ::testing::ValuesIn(filters),
    ::testing::ValuesIn(strides), ::testing::ValuesIn(padBegins),
    ::testing::ValuesIn(padEnds), ::testing::ValuesIn(dilations),
    ::testing::ValuesIn(groups), ::testing::ValuesIn(defor_groups),
    ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid = ::testing::Combine(
    ::testing::ValuesIn(offsets), ::testing::ValuesIn(filters),
    ::testing::ValuesIn(strides),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(groups),
    ::testing::ValuesIn(defor_groups), ::testing::ValuesIn(numOutChannels),
    ::testing::Values(ngraph::op::PadType::VALID));

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution2D_Serialization_ExplicitPadding, DeformableConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_ExplicitPadding, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 28, 28})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(
    smoke_DeformableConvolution2D__Serialization_AutoPadValid, DeformableConvolutionLayerTest,
    ::testing::Combine(
        conv2DParams_AutoPadValid, ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(std::vector<size_t>({1, 1, 28, 28})),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    DeformableConvolutionLayerTest::getTestCaseName);
}  // namespace
