// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "conv_concat.hpp"
#include "utils/convolution_params.hpp"
#include "common_test_utils/node_builders/convolution.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution_backprop_data.hpp"

using namespace CPUTestUtils;

namespace ov {
namespace test {

std::string ConvConcatSubgraphTest::getTestCaseName(testing::TestParamInfo<convConcatCPUParams> obj) {
    std::ostringstream result;
    nodeType type;
    commonConvParams convParams;
    CPUSpecificParams cpuParams;
    ov::Shape inputShapes;
    int axis;
    std::tie(type, convParams, cpuParams, inputShapes, axis) = obj.param;

    result << "Type=" << nodeType2str(type) << "_";

    std::vector<size_t> kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ov::op::PadType paddingType;
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
    ov::Shape inputShapes;
    int axis;

    std::tie(type, convParams, cpuParams, inputShapes, axis) = this->GetParam();
    pluginTypeNode = nodeType2PluginType(type);
    std::vector<size_t> kernelSize, strides, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t numOutChannels, numOfGroups;
    ov::op::PadType paddingType;

    std::tie(kernelSize, strides, padBegin, padEnd, dilation, numOutChannels, paddingType, numOfGroups) = convParams;
    std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

    selectedType += "_f32";

    ov::ParameterVector inputParams{std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShapes),
                                    std::make_shared<ov::op::v0::Parameter>(ov::element::f32, inputShapes)};

    std::vector<std::shared_ptr<ov::Node>> convolutionNodes(2);
    switch (type) {
        case nodeType::convolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_convolution(inputParams[conv],
                                                                           ov::element::f32,
                                                                           kernelSize,
                                                                           strides,
                                                                           padBegin,
                                                                           padEnd,
                                                                           dilation,
                                                                           paddingType,
                                                                           numOutChannels);
            }
            break;
        }
        case nodeType::convolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_convolution_backprop_data(inputParams[conv],
                                                                                         ov::element::f32,
                                                                                         kernelSize,
                                                                                         strides,
                                                                                         padBegin,
                                                                                         padEnd,
                                                                                         dilation,
                                                                                         paddingType,
                                                                                         numOutChannels);
            }
            break;
        }
        case nodeType::groupConvolution : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_group_convolution(inputParams[conv],
                                                                                 ov::element::f32,
                                                                                 kernelSize,
                                                                                 strides,
                                                                                 padBegin,
                                                                                 padEnd,
                                                                                 dilation,
                                                                                 paddingType,
                                                                                 numOutChannels,
                                                                                 numOfGroups);
            }
            break;
        }
        case nodeType::groupConvolutionBackpropData : {
            for (size_t conv = 0; conv < convolutionNodes.size(); conv++) {
                convolutionNodes[conv] = ov::test::utils::make_group_convolution_backprop_data(inputParams[conv],
                                                                                               ov::element::f32,
                                                                                               kernelSize,
                                                                                               strides,
                                                                                               padBegin,
                                                                                               padEnd,
                                                                                               dilation,
                                                                                               paddingType,
                                                                                               numOutChannels,
                                                                                               numOfGroups);
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

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(concat)};
    function = std::make_shared<ov::Model>(results, inputParams, "convolutionConcat");
}

TEST_P(ConvConcatSubgraphTest, CompareWithRefs) {
    run();

    CheckPluginRelatedResults(compiledModel, pluginTypeNode);
};

namespace ConvConcat {

const ov::Shape inputShapes2D() {
    return ov::Shape{1, 64, 16, 16};
}

const ov::Shape inputShapes3D() {
    return ov::Shape{1, 64, 8, 16, 16};
}

const int axis() {
    return 1;
}

const ov::op::PadType paddingType() {
    return ov::op::PadType::EXPLICIT;
}
const size_t numOutChannels() {
    return 32;
}

const ov::Shape kernelSize2D() {
    return ov::Shape{3, 3};
}

const ov::Shape strides2D() {
    return ov::Shape{2, 2};
}

const std::vector<ptrdiff_t> padBegin2D() {
    return std::vector<ptrdiff_t>{1, 1};
}

const std::vector<ptrdiff_t> padEnd2D() {
    return std::vector<ptrdiff_t>{1, 1};
}

const ov::Shape dilation2D() {
    return ov::Shape{1, 1};
}

const ov::Shape kernelSize3D() {
    return ov::Shape{3, 3, 3};
}

const ov::Shape strides3D() {
    return ov::Shape{2, 2, 2};
}

const std::vector<ptrdiff_t> padBegin3D() {
    return std::vector<ptrdiff_t>{1, 1, 1};
}

const std::vector<ptrdiff_t> padEnd3D() {
    return std::vector<ptrdiff_t>{1, 1, 1};
}

const ov::Shape dilation3D() {
    return ov::Shape{1, 1, 1};
}

const commonConvParams convParams2D() {
    return commonConvParams{kernelSize2D(), strides2D(), padBegin2D(), padEnd2D(), dilation2D(), numOutChannels(), paddingType(), 1};
}

const commonConvParams convParams3D() {
    return commonConvParams{kernelSize3D(), strides3D(), padBegin3D(), padEnd3D(), dilation3D(), numOutChannels(), paddingType(), 1};
}

const commonConvParams groupConvParams2D() {
    return commonConvParams{kernelSize2D(), strides2D(), padBegin2D(), padEnd2D(), dilation2D(), numOutChannels(), paddingType(), 2};
}

const commonConvParams groupConvParams3D() {
    return commonConvParams{kernelSize3D(), strides3D(), padBegin3D(), padEnd3D(), dilation3D(), numOutChannels(), paddingType(), 2};
}

const std::vector<CPUSpecificParams> blockedCPUParams2D() {
    static std::vector<CPUSpecificParams> resCPUParams = {block8c_2D};
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(block16c_2D);
    }
    return resCPUParams;
}

const std::vector<CPUSpecificParams> blockedCPUParams3D() {
    static std::vector<CPUSpecificParams> resCPUParams = {block8c_3D};
    if (ov::with_cpu_x86_avx512f()) {
        resCPUParams.push_back(block16c_3D);
    }
    return resCPUParams;
}

}  // namespace ConvConcat
}  // namespace test
}  // namespace ov