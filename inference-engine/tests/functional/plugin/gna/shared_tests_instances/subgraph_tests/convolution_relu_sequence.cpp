// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/convolution_relu_sequence.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<size_t> inputShapeSimple = {
    {1, 32, 64, 16},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeq {
    {
        {2, 2},     // Kernel size
        {2, 2},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        3           // Num out channels
    },
    {
        {2, 5},     // Kernel size
        {2, 3},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8           // Num out channels
    },
};

const InferenceEngine::SizeVector inputShapeFB = {
    {1, 1, 5, 236},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsFBSeq = {
    {
        {5, 7},     // Kernel size
        {1, 1},     // Stride
        {2, 3},     // Pad begin
        {2, 3},     // Pad end
        32          // Num out channels
    },
    {
        {9, 5},     // Kernel size
        {1, 1},     // Stride
        {4, 2},     // Pad begin
        {4, 2},     // Pad end
        32           // Num out channels
    },
        {
        {1, 1},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8           // Num out channels
    },
};

const std::vector<convReluSpecificParamsAll> convReluSpecificParamsAllAll = {
    {
        inputShapeSimple,
        convReluSpecificParamsSimpleSeq
    },
    {
        inputShapeFB,
        convReluSpecificParamsFBSeq
    }
};

// Enable when using GNA 2.1 library
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_ConvolutionReluSequenceTest, ConvolutionReluSequenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(convReluSpecificParamsAllAll),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_GNA)),
    ConvolutionReluSequenceTest::getTestCaseName);

} // namespace
