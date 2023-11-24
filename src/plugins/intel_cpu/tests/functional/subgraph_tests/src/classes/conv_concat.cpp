// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_concat.hpp"

using namespace ngraph;
using namespace InferenceEngine;
using namespace CPUTestUtils;

namespace SubgraphTestsDefinitions {

std::string ConvConcatSubgraphTest::getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj) {
    std::ostringstream result;
    nodeType type;
    commonConvParams convParams;
    CPUSpecificParams cpuParams;
    SizeVector inputShapes;
    int axis;
    std::tie(type, convParams, cpuParams, inputShapes, axis) = obj.param;

    result << "Type=" << nodeType2str(type) << "_";

    SizeVector kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ngraph::op::PadType paddingType;
    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups) = convParams;

    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernelSize) << "_";
    result << "S" << ov::test::utils::vec2str(strides) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << numOutChannels << "_";
    result << "G=" << numOfGroups << "_";
    result << "AP=" << paddingType << "_";

    result << CPUTestsBase::getTestCaseName(cpuParams);

    result << "_axis=" << axis;

    return result.str();
}

void ConvConcatSubgraphTest::SetUp() {
    targetDevice = ov::test::utils::DEVICE_CPU;
    nodeType type;
    commonConvParams convParams;
    CPUSpecificParams cpuParams;
    SizeVector inputShapes;
    int axis;

    std::tie(type, convParams, cpuParams, inputShapes, axis) = this->GetParam();
    pluginTypeNode = nodeType2PluginType(type);
    SizeVector kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ngraph::op::PadType paddingType;

    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups) = convParams;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    selectedType += "_FP32";

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShapes)),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape(inputShapes))};

    std::vector<std::shared_ptr<ngraph::Node>> convolutionNodes(2);
    switch (type) {
        case nodeType::convolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ngraph::builder::makeConvolution(inputParams[conv], ngraph::element::f32, kernelSize, strides, padBegin,
                                                                          padEnd, dilation, paddingType, numOutChannels);
            }
            break;
        }
        case nodeType::convolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ngraph::builder::makeConvolutionBackpropData(inputParams[conv], ngraph::element::f32, kernelSize, strides, padBegin,
                                                                                      padEnd, dilation, paddingType, numOutChannels);
            }
            break;
        }
        case nodeType::groupConvolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ngraph::builder::makeGroupConvolution(inputParams[conv], ngraph::element::f32, kernelSize, strides, padBegin,
                                                                                           padEnd, dilation, paddingType, numOutChannels, numOfGroups);
            }
            break;
        }
        case nodeType::groupConvolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ngraph::builder::makeGroupConvolutionBackpropData(inputParams[conv], ngraph::element::f32, kernelSize,
                                                                                           strides, padBegin, padEnd, dilation, paddingType,
                                                                                           numOutChannels, numOfGroups);
            }
            break;
        }
        default: {
            throw std::runtime_error("Subgraph concat test doesn't support this type of operation");
        }
    }
    for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
        convolutionNodes[conv]->get_rt_info() = getCPUInfo();
    }

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::NodeVector{convolutionNodes[0], convolutionNodes[1]}, axis);

    ngraph::ResultVector results{std::make_shared<ngraph::opset4::Result>(concat)};
    function = std::make_shared<ngraph::Function>(results, inputParams, "convolutionConcat");
}

TEST_P(ConvConcatSubgraphTest, CompareWithRefs) {
    Run();

    CheckPluginRelatedResults(executableNetwork, pluginTypeNode);
};

namespace ConvConcat {

const SizeVector inputShapes3D() {
    return SizeVector{1, 64, 8, 16, 16};
}

const int axis() {
    return 1;
}

const ngraph::op::PadType paddingType() {
    return ngraph::op::PadType::EXPLICIT;
}
const size_t numOutChannels() {
    return 32;
}

const SizeVector kernelSize3D() {
    return SizeVector{3, 3, 3};
}

const SizeVector strides3D() {
    return SizeVector{2, 2, 2};
}

const std::vector<ptrdiff_t> padBegin3D() {
    return std::vector<ptrdiff_t>{1, 1, 1};
}

const std::vector<ptrdiff_t> padEnd3D() {
    return std::vector<ptrdiff_t>{1, 1, 1};
}

const SizeVector dilation3D() {
    return SizeVector{1, 1, 1};
}

const commonConvParams convParams3D() {
    return commonConvParams{kernelSize3D(), strides3D(), padBegin3D(), padEnd3D(), dilation3D(), numOutChannels(), paddingType(), 1};
}

const commonConvParams groupConvParams3D() {
    return commonConvParams{kernelSize3D(), strides3D(), padBegin3D(), padEnd3D(), dilation3D(), numOutChannels(), paddingType(), 2};
}
} // namespace ConvConcat

} // namespace SubgraphTestsDefinitions
