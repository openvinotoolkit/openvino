// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/group_convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32
};

/* ============= 2D GroupConvolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 3}};
const std::vector<std::vector<size_t >> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t >> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};

const auto groupConv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto groupConv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::ValuesIn(numGroups),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 16, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution2D_AutoPadValid, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 16, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

/* ============= 3D GroupConvolution ============= */
const std::vector<std::vector<size_t >> kernels3d = {{3, 3, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0}};

const std::vector<std::vector<size_t >> strides3d = {{1, 1, 1}};
const std::vector<std::vector<size_t >> dilations3d = {{1, 1, 1}};

const auto groupConv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(4),
        ::testing::Values(2),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto groupConv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(4),
        ::testing::Values(2),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D_ExplicitPadding, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv3DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 4, 10, 10, 10})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution3D_AutoPadValid, GroupConvolutionLayerTest,
                        ::testing::Combine(
                                groupConv3DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 4, 10, 10, 10})),
                                ::testing::Values(CommonTestUtils::DEVICE_GPU)),
                        GroupConvolutionLayerTest::getTestCaseName);

}  // namespace
