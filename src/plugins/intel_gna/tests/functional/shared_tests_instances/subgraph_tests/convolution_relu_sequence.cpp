// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/convolution_relu_sequence.hpp"

#include <vector>

#include "../skip_tests_check.hpp"
#include "common_test_utils/test_constants.hpp"
#include "gna/gna_config.hpp"

using namespace SubgraphTestsDefinitions;

namespace {

class GnaConvolutionReluSequenceTest : public ConvolutionReluSequenceTest {
protected:
    void Run() override {
        ConvolutionReluSequenceTest::Run();
    }

    void SetUp() override {
        ConvolutionReluSequenceTest::SetUp();
    }
};

TEST_P(GnaConvolutionReluSequenceTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

const std::vector<size_t> inputShapeSimple = {
    {1, 32, 64, 32},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeq{
    {
        {2, 2},  // Kernel size
        {2, 2},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        16,      // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
    {
        {2, 1},  // Kernel size
        {2, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        8,       // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
};

const std::vector<size_t> inputShapeSimpleWithPooling = {
    {1, 32, 53, 110},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsSimpleSeqWithPooling{
    {
        {3, 3},  // Kernel size
        {1, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        16,      // Num out channels
        {3, 3},  // Pooling window
        {3, 3}   // Pooling stride
    },
    {
        {2, 2},  // Kernel size
        {1, 2},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        8,       // Num out channels
        {2, 2},  // Pooling window
        {2, 2}   // Pooling stride
    },
};

const InferenceEngine::SizeVector inputShapeFB = {
    {1, 1, 5, 236},
};

const std::vector<convReluSpecificParams> convReluSpecificParamsFBSeq = {
    {
        {5, 7},  // Kernel size
        {1, 1},  // Stride
        {2, 3},  // Pad begin
        {2, 3},  // Pad end
        32,      // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
    {
        {9, 5},  // Kernel size
        {1, 1},  // Stride
        {4, 2},  // Pad begin
        {4, 2},  // Pad end
        32,      // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
    {
        {1, 1},  // Kernel size
        {1, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        8,       // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
};

const InferenceEngine::SizeVector inputShape3 = {
    {1, 8, 18, 54},
};

const std::vector<convReluSpecificParams> convReluSpecificParams3Seq = {
    {
        {1, 3},  // Kernel size
        {1, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        32,      // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
    {
        {2, 1},  // Kernel size
        {1, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        8,       // Num out channels
        {1, 1},  // Pooling window
        {1, 1}   // Pooling stride
    },
    {
        {3, 3},  // Kernel size
        {3, 3},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        8,       // Num out channels
        {3, 3},  // Pooling window
        {3, 3}   // Pooling stride
    },
};

const std::vector<convReluSpecificParamsAll> convReluSpecificParamsAllAll = {
    {inputShapeSimple, convReluSpecificParamsSimpleSeq},
    {inputShape3, convReluSpecificParams3Seq},
    // Enable when bigger kernels (e.g., 5x7, 9x5) and input padding supported
    // {
    //     inputShapeFB,
    //     convReluSpecificParamsFBSeq
    // },
    {inputShapeSimpleWithPooling, convReluSpecificParamsSimpleSeqWithPooling}};

const std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_FP32"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const std::vector<std::map<std::string, std::string>> configs_allowing_pooling_stride_above_window = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_2_0"}}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionReluSequenceTest,
                         GnaConvolutionReluSequenceTest,
                         ::testing::Combine(::testing::ValuesIn(convReluSpecificParamsAllAll),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         GnaConvolutionReluSequenceTest::getTestCaseName);

const InferenceEngine::SizeVector inputShape1Even = {
    {1, 1, 48, 1},
};

const InferenceEngine::SizeVector inputShape1DOneAbove = {
    {1, 1, 41, 1},
};

const InferenceEngine::SizeVector inputShape1DOneBelow = {
    {1, 1, 47, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel4 = {
    {1, 4, 49, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel5 = {
    {1, 5, 49, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel6 = {
    {1, 6, 49, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel7 = {
    {1, 7, 49, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel8 = {
    {1, 8, 49, 1},
};

const InferenceEngine::SizeVector inputShape1DMultichannel9 = {
    {1, 9, 49, 1},
};

const std::vector<convReluSpecificParams> poolingStrideBelowWindow = {
    {
        {3, 1},  // Kernel size
        {2, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        4,       // Num out channels
        {4, 1},  // Pooling window
        {2, 1}   // Pooling stride
    },
};

const std::vector<convReluSpecificParams> poolingStrideAboveWindow = {
    {
        {3, 1},  // Kernel size
        {2, 1},  // Stride
        {0, 0},  // Pad begin
        {0, 0},  // Pad end
        4,       // Num out channels
        {2, 1},  // Pooling window
        {4, 1}   // Pooling stride
    },
};

const std::vector<convReluSpecificParamsAll> poolingStrideNotEqualWindow_Above = {
    {inputShape1Even, poolingStrideAboveWindow},
    {inputShape1DOneAbove, poolingStrideAboveWindow},
    {inputShape1DOneBelow, poolingStrideAboveWindow},
    {inputShape1DMultichannel4, poolingStrideAboveWindow},
    {inputShape1DMultichannel5, poolingStrideAboveWindow},
    {inputShape1DMultichannel6, poolingStrideAboveWindow},
    {inputShape1DMultichannel7, poolingStrideAboveWindow},
    {inputShape1DMultichannel8, poolingStrideAboveWindow},
    {inputShape1DMultichannel9, poolingStrideAboveWindow}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionPoolingStrideNotEqualWindowTest_Above,
                         ConvolutionReluSequenceTest,
                         ::testing::Combine(::testing::ValuesIn(poolingStrideNotEqualWindow_Above),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs_allowing_pooling_stride_above_window)),
                         ConvolutionReluSequenceTest::getTestCaseName);

const std::vector<convReluSpecificParamsAll> poolingStrideNotEqualWindow_Below = {
    {inputShape1Even, poolingStrideBelowWindow},
    {inputShape1DOneAbove, poolingStrideBelowWindow},
    {inputShape1DOneBelow, poolingStrideBelowWindow},
    {inputShape1DMultichannel4, poolingStrideBelowWindow},
    {inputShape1DMultichannel5, poolingStrideBelowWindow},
    {inputShape1DMultichannel6, poolingStrideBelowWindow},
    {inputShape1DMultichannel7, poolingStrideBelowWindow},
    {inputShape1DMultichannel8, poolingStrideBelowWindow},
    {inputShape1DMultichannel9, poolingStrideBelowWindow}};

INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionPoolingStrideNotEqualWindowTest_Below,
                         ConvolutionReluSequenceTest,
                         ::testing::Combine(::testing::ValuesIn(poolingStrideNotEqualWindow_Below),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionReluSequenceTest::getTestCaseName);
}  // namespace
