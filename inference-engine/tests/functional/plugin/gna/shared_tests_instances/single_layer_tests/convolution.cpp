// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_layer_tests/convolution.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels = {{3, 1},
                                                          {5, 1}};
const std::vector<std::vector<size_t >> strides = {{1, 1},
                                                          {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 1},
                                                       {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 1},
                                                     {3, 1}};
const std::vector<std::vector<size_t >> dilations = {{1, 1},
                                                            {3, 1}};
const std::vector<size_t> numOutCannels = {1, 5};
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

// TODO: Issue:  26417
INSTANTIATE_TEST_CASE_P(DISABLED_Convolution2D_ExplicitPadding, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 1})),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_Convolution2D_AutoPadValid, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(std::vector<size_t >({1, 3, 30, 1})),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);
}  // namespace
