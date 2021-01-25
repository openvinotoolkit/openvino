// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/include/conv_dw_conv_fusing.hpp"

using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

    std::string ConvDWConvFusingSubgraphTest::getTestCaseName(testing::TestParamInfo<convDWConvFusingCPUParams> obj) {
        std::ostringstream result;
        commonConvParams convParams;
        CPUSpecificParams cpuParams;
        SizeVector inputShapes;
        std::tie(convParams, cpuParams, inputShapes) = obj.param;

        SizeVector kernelSize, strides, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels, numOfGroups, dwStride;
        ngraph::op::PadType paddingType;
        bool withBias, withDWBias;
        std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups, dwStride, withBias, withDWBias) = convParams;

        result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
        result << "K" << CommonTestUtils::vec2str(kernelSize) << "_";
        result << "S" << CommonTestUtils::vec2str(strides) << "_";
        result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
        result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
        result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
        result << "O=" << numOutChannels << "_";
        result << "G=" << numOfGroups << "_";
        result << "AP=" << paddingType << "_";
        result << "dwStride=" << dwStride << "_";
        result << "withBias=" << withBias << "_";
        result << "withDWBias=" << withDWBias << "_";

        result << CPUTestsBase::getTestCaseName(cpuParams);

        return result.str();
    }

    void ConvDWConvFusingSubgraphTest::SetUp() {
        targetDevice = CommonTestUtils::DEVICE_CPU;
        commonConvParams convParams;
        CPUSpecificParams cpuParams;
        SizeVector inputShapes;

        std::tie(convParams, cpuParams, inputShapes) = this->GetParam();
        SizeVector kernelSize, strides, dilation;
        std::vector<ptrdiff_t> padBegin, padEnd;
        size_t numOutChannels, numOfGroups, dwStride;
        ngraph::op::PadType paddingType;
        bool withBias, withDWBias;

        std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups, dwStride, withBias, withDWBias) = convParams;
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        selectedType += "_FP32";

        auto inputParams = ngraph::builder::makeParams(ngraph::element::f32, {inputShapes});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(inputParams));

        auto conv1 = ngraph::builder::makeConvolution(paramOuts[0], ngraph::element::f32, kernelSize, strides, padBegin,
                                                      padEnd, dilation, paddingType, numOutChannels, withBias);

        auto conv2 = ngraph::builder::makeGroupConvolution(conv1, ngraph::element::f32, {3, 3}, {dwStride, dwStride}, {1, 1}, {1, 1}, {1, 1},
                                                           ngraph::op::PadType::EXPLICIT, 512, 512, withDWBias);

        ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(conv2)};
        function = std::make_shared<ngraph::Function>(results, inputParams, "convolutionDWConvolutionFusingPattern");
    }

TEST_P(ConvDWConvFusingSubgraphTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()

    Run();
//    CheckPluginRelatedResults(executableNetwork);
};

/* ============= Kernel_1x1 (2D) ============= */
    const std::vector<CPUSpecificParams> CPUParams2DConv = {
//            conv_avx512_2D_1x1,
            conv_avx2_2D_1x1
    };

    std::vector<commonConvParams> convParams2D1x1_stride1_Set = {
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 1, true, true},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 1, true, false},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 1, false, true},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 1, false, false}
    };

    const SizeVector inputShapes2D{4, 512, 64, 64};

    const auto params2DConv = ::testing::Combine(
            ::testing::ValuesIn(convParams2D1x1_stride1_Set),
            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2DConv)),
            ::testing::Values(inputShapes2D));

    INSTANTIATE_TEST_CASE_P(smoke_Convolution2D1x1, ConvDWConvFusingSubgraphTest, params2DConv, ConvDWConvFusingSubgraphTest::getTestCaseName);

    std::vector<commonConvParams> convParams2D1x1_stride2_Set = {
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 2, true, true},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 2, true, false},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 2, false, true},
            commonConvParams{{1, 1}, {1, 1}, {0, 0}, {0, 0}, {1, 1}, 512, ngraph::op::PadType::EXPLICIT, 1, 2, false, false}
    };

    const auto params2DConv_2 = ::testing::Combine(
            ::testing::ValuesIn(convParams2D1x1_stride2_Set),
            ::testing::ValuesIn(filterCPUInfoForDevice(CPUParams2DConv)),
            ::testing::Values(inputShapes2D));

    INSTANTIATE_TEST_CASE_P(smoke_Convolution2D1x1_2, ConvDWConvFusingSubgraphTest, params2DConv_2, ConvDWConvFusingSubgraphTest::getTestCaseName);

}  // namespace SubgraphTestsDefinitions
