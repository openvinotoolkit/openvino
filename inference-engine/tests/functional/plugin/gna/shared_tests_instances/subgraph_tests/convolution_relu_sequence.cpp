// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "gna/gna_config.hpp"

#include "subgraph_tests/convolution_relu_sequence.hpp"
#include "common_test_utils/test_constants.hpp"
#include "../skip_tests_check.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

class GnaConvolutionReluSequenceTest : public ConvolutionReluSequenceTest, GnaLayerTestCheck {
protected:
    void Run() override {
        GnaLayerTestCheck::SkipTestCheck();

        if (!GnaLayerTestCheck::skipTest) {
            ConvolutionReluSequenceTest::Run();
        }
    }

    void SetUp() override {
        ConvolutionReluSequenceTest::SetUp();
    }
};

TEST_P(GnaConvolutionReluSequenceTest, CompareWithRefs) {
    Run();
}


const std::vector<InferenceEngine::Precision> netPrecisions = {
    InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16
};

const std::vector<size_t> inputShapeSimple = {
    {1, 32, 64, 32},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeq {
    {
        {2, 2},     // Kernel size
        {2, 2},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        16,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
    {
        {2, 1},     // Kernel size
        {2, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,          // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
};

const std::vector<size_t> inputShapeSimpleWithPooling = {
    {1, 32, 53, 110},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeqWithPooling {
    {
        {3, 3},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        16,         // Num out channels
        {3, 3},     //Pooling window
        {3, 3}      //Pooling stride
    },
    {
        {2, 2},     // Kernel size
        {1, 2},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,         // Num out channels
        {2, 2},     //Pooling window
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

const InferenceEngine::SizeVector inputShape3 = {
    {1, 8, 18, 54},
};

const std::vector<convReluSpecificParams> convReluSpecificParams3Seq = {
    {
        {1, 3},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        32,         // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
    {
        {2, 1},     // Kernel size
        {1, 1},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,          // Num out channels
        {1, 1},     //Pooling window
        {1, 1}      //Pooling stride
    },
        {
        {3, 3},     // Kernel size
        {3, 3},     // Stride
        {0, 0},     // Pad begin
        {0, 0},     // Pad end
        8,          // Num out channels
        {3, 3},     //Pooling window
        {3, 3}      //Pooling stride
    },
};

const std::vector<convReluSpecificParamsAll> convReluSpecificParamsAllAll = {
    {
        inputShapeSimple,
        convReluSpecificParamsSimpleSeq
    },
    {
        inputShape3,
        convReluSpecificParams3Seq
    },
    // Enable when bigger kernels (e.g., 5x7, 9x5) and input padding supported
    // {
    //     inputShapeFB,
    //     convReluSpecificParamsFBSeq
    // },
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
INSTANTIATE_TEST_CASE_P(smoke_ConvolutionReluSequenceTest, GnaConvolutionReluSequenceTest,
    ::testing::Combine(
        ::testing::ValuesIn(convReluSpecificParamsAllAll),
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(CommonTestUtils::DEVICE_GNA),
        ::testing::ValuesIn(configs)),
    GnaConvolutionReluSequenceTest::getTestCaseName);

} // namespace
