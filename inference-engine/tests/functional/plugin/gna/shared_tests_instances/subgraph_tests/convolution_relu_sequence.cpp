// Copyright (C) 2019-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gna/gna_config.hpp"

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

const std::vector<size_t> inputShapeSimpleWithPooling = {
    {1, 32, 128, 32},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeq {
    {
        {2, 2},     // Kernel size
        {2, 2},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        3,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
    {
        {2, 5},     // Kernel size
        {2, 3},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeqWithPooling {
    {
        {3, 3},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        3,         // Num out channels
        {2, 3},     //Pooling window
        {2, 3}      //Pooling stride
    },
    {
        {2, 2},     // Kernel size
        {1, 2},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,         // Num out channels
        {2, 3},     //Pooling window
        {2, 2}      //Pooling stride
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
        32,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
    {
        {9, 5},     // Kernel size
        {1, 1},     // Stride
        {4, 2},     // Pad begin
        {4, 2},     // Pad end
        32,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
        {
        {1, 1},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
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
    },
    {
        inputShapeSimpleWithPooling,
        convReluSpecificParamsSimpleSeqWithPooling
    }
};

const std::vector<std::map<std::string, std::string> > configs = {
    {
        {InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_AUTO}
    },
    {
        {InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW_FP32}
    },
    {
        {InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE, InferenceEngine::GNAConfigParams::GNA_SW_EXACT}
    }
};

// Enable when using GNA 2.1 library
INSTANTIATE_TEST_CASE_P(DISABLED_smoke_ConvolutionReluSequenceTest, ConvolutionReluSequenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(convReluSpecificParamsAllAll),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    ConvolutionReluSequenceTest::getTestCaseName);

} // namespace
