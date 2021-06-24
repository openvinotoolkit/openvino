// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
// Layer specific
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
const std::vector<size_t> numOutCannels = {1, 5};
const std::vector<ngraph::op::PadType> padTypes = {
        ngraph::op::PadType::EXPLICIT,
        ngraph::op::PadType::VALID  // FAILURE: Cannot create convolution forward descriptor for layer: Convolution_19010
};
const auto conv2DParams_ExplicitPadding = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::ValuesIn(padBegins),
        ::testing::ValuesIn(padEnds),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid = ::testing::Combine(
        ::testing::ValuesIn(kernels),
        ::testing::ValuesIn(strides),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilations),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);
const auto conv2DParams_BigDimensionValid = ::testing::Combine(
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1}})),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1}})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(std::vector<std::vector<size_t>>({{1, 1}})),
        ::testing::ValuesIn(std::vector<size_t>({64})),
        ::testing::Values(ngraph::op::PadType::VALID)
);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 30})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConvolutionLayerTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_BigDimensionValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_BigDimensionValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(std::vector<size_t >({1, 3, 1, 2500})),
                                ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
                        ConvolutionLayerTest::getTestCaseName);

/* ============= 3D Convolution ============= */
// TODO: 3D convolution fails with sigabort
}  // namespace
