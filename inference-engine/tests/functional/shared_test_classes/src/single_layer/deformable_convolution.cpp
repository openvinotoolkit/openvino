// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/deformable_convolution.hpp"

namespace LayerTestsDefinitions {

std::string DeformableConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<deformableConvLayerTestParamsSet> obj) {
    deformableConvSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) =
        obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "G=" << groups << "_";
    result << "DG=" << deformable_groups << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

InferenceEngine::Blob::Ptr DeformableConvolutionLayerTest::GenerateInput(const InferenceEngine::InputInfo &info) const {
        InferenceEngine::Blob::Ptr blobPtr;
        const std::string name = info.name();
        if (name == "data") {
            auto data_shape = info.getTensorDesc().getDims();
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0, 0, 0, 7235346);
        } else {
            blobPtr = FuncTestUtils::createAndFillBlob(info.getTensorDesc(), 0, 0, 0, 7235346);
        }
        return blobPtr;
}

void DeformableConvolutionLayerTest::SetUp() {
    deformableConvSpecificParams convParams;
    std::vector<size_t> inputShape;
    auto netPrecision   = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) =
        this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t groups, deformable_groups, convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, groups, deformable_groups, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto deformable_values = std::make_shared<ngraph::op::Parameter>(ngPrc, ngraph::Shape(kernel));
    deformable_values->set_friendly_name("data");
    auto deformable_conv = std::make_shared<ngraph::opset1::DeformableConvolution>(paramOuts[0], deformable_values, paramOuts[2],
                                                              stride, padBegin, padEnd, dilation, padType, groups, deformable_groups);
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(deformable_conv)};
    function = std::make_shared<ngraph::Function>(results, params, "deformable_convolution");
}
}  // namespace LayerTestsDefinitions
