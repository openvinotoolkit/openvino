// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

// ! [test_convolution:declare_parameters]
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16,
};

/* ============= 2D Convolution ============= */

const std::vector<std::vector<size_t >> kernels = {{3, 3},
                                                   {3, 5}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                   {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0},
                                                       {0, 3}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0},
                                                     {0, 3}};
const std::vector<std::vector<size_t >> dilations = {{1, 1},
                                                     {3, 1}};
const std::vector<size_t> numOutChannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::VALID
};

const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
// ! [test_convolution:declare_parameters]

const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutChannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

// ! [test_convolution:instantiate]
INSTANTIATE_TEST_SUITE_P(Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_ExplicitPadding,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                         ConvolutionLayerTest::getTestCaseName);
// ! [test_convolution:instantiate]

INSTANTIATE_TEST_SUITE_P(Convolution2D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv2DParams_AutoPadValid,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                         ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */

const std::vector<std::vector<size_t >> kernels3d = {{3, 3, 3},
                                                     {3, 5, 3}};
const std::vector<std::vector<ptrdiff_t>> paddings3d = {{0, 0, 0},
                                                        {0, 2, 0}};

const std::vector<std::vector<size_t >> strides3d = {{1, 1, 1},
                                                     {1, 2, 1}};
const std::vector<std::vector<size_t >> dilations3d = {{1, 1, 1},
                                                       {1, 2, 1}};

const auto conv3DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(paddings3d),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(5),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv3DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels3d),
        ::testing::ValuesIn(strides3d),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0, 0})),
        ::testing::ValuesIn(dilations3d),
        ::testing::Values(5),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution3D_ExplicitPadding, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv3DParams_ExplicitPadding,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 10, 10, 10})),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                         ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(nightly_Convolution3D_AutoPadValid, ConvolutionLayerTest,
                         ::testing::Combine(
                                 conv3DParams_AutoPadValid,
                                 ::testing::ValuesIn(netPrecisions),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(InferenceEngine::Layout::ANY),
                                 ::testing::Values(std::vector<size_t >({1, 3, 10, 10, 10})),
                                 ::testing::Values(CommonTestUtils::DEVICE_TEMPLATE)),
                         ConvolutionLayerTest::getTestCaseName);

}  // namespace