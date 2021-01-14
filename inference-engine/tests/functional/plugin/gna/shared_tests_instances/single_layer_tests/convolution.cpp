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
const std::vector<std::vector<size_t >> kernelsH1 = {{1, 3},
                                                     {1, 5}};
const std::vector<std::vector<size_t >> stridesH1 = {{1, 1},
                                                     {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padBeginsH1 = {{1, 0},
                                                         {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padEndsH1 = {{1, 0},
                                                       {1, 3}};
const std::vector<std::vector<size_t >> dilationsH1 = {{1, 1},
                                                       {1, 3}};
// TODO: Currently C != 1 is not supported for graphs with native NCHW layout (will be fixed in 40496)
const std::vector<std::vector<size_t>> inputShapesH1 = {{1, 1, 1, 32},
                                                        {1, 1, 1, 160},
                                                        {1, 1, 1, 64}};
const std::vector<std::vector<size_t >> kernelsW1 = {{3, 1},
                                                     {5, 1}};
const std::vector<std::vector<size_t >> stridesW1 = {{1, 1},
                                                     {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padBeginsW1 = {{0, 1},
                                                         {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padEndsW1 = {{0, 1},
                                                       {3, 1}};
const std::vector<std::vector<size_t >> dilationsW1 = {{1, 1},
                                                       {3, 1}};
// TODO: Currently C != 1 is not supported for graphs with native NCHW layout (will be fixed in 40496)
const std::vector<std::vector<size_t>> inputShapesW1 = {{1, 1, 32, 1},
                                                        {1, 1, 160, 1},
                                                        {1, 1, 64, 1}};
const std::vector<size_t> numOutCannels = {4, 8, 12};

const auto conv2DParams_ExplicitPadding_Height1 = ::testing::Combine(
        ::testing::ValuesIn(kernelsH1),
        ::testing::ValuesIn(stridesH1),
        ::testing::ValuesIn(padBeginsH1),
        ::testing::ValuesIn(padEndsH1),
        ::testing::ValuesIn(dilationsH1),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_ExplicitPadding_Width1 = ::testing::Combine(
        ::testing::ValuesIn(kernelsW1),
        ::testing::ValuesIn(stridesW1),
        ::testing::ValuesIn(padBeginsW1),
        ::testing::ValuesIn(padEndsW1),
        ::testing::ValuesIn(dilationsW1),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParams_AutoPadValid_Height1 = ::testing::Combine(
        ::testing::ValuesIn(kernelsH1),
        ::testing::ValuesIn(stridesH1),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilationsH1),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);
const auto conv2DParams_AutoPadValid_Width1 = ::testing::Combine(
        ::testing::ValuesIn(kernelsW1),
        ::testing::ValuesIn(stridesW1),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
        ::testing::ValuesIn(dilationsW1),
        ::testing::ValuesIn(numOutCannels),
        ::testing::Values(ngraph::op::PadType::VALID)
);

// TODO: padding isn't currently supported in GNA
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Convolution2D_ExplicitPadding_Height1, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding_Height1,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesH1),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(DISABLED_smoke_Convolution2D_ExplicitPadding_Width1, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_ExplicitPadding_Width1,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesW1),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid_Height1, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid_Height1,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesH1),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);

INSTANTIATE_TEST_CASE_P(smoke_Convolution2D_AutoPadValid_Width1, ConvolutionLayerTest,
                        ::testing::Combine(
                                conv2DParams_AutoPadValid_Width1,
                                ::testing::ValuesIn(netPrecisions),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::Values(InferenceEngine::Layout::ANY),
                                ::testing::ValuesIn(inputShapesW1),
                                ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                        ConvolutionLayerTest::getTestCaseName);
}  // namespace
