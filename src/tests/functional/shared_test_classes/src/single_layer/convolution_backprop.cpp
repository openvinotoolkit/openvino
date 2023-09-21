// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convolution_backprop.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionBackpropLayerTest::getTestCaseName(const testing::TestParamInfo<convBackpropLayerTestParamsSet>& obj) {
    convBackpropSpecificParams convBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    InferenceEngine::SizeVector outputShapes;
    std::string targetDevice;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = convBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "OS=" << ov::test::utils::vec2str(outputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "OP=" << ov::test::utils::vec2str(outPadding) << "_";
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

void ConvolutionBackpropLayerTest::SetUp() {
    convBackpropSpecificParams convBackpropDataParams;
    std::vector<size_t> inputShape;
    std::vector<size_t> outputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, outPadding) = convBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                                        padEnd, dilation, padType, convOutChannels, false, outPadding));
    if (!outputShape.empty()) {
        auto outShape = ngraph::opset3::Constant::create(ngraph::element::i64, {outputShape.size()}, outputShape);
        convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
        ngraph::builder::makeConvolutionBackpropData(paramOuts[0], outShape, ngPrc, kernel, stride, padBegin,
                                                        padEnd, dilation, padType, convOutChannels));
    }
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "convolutionBackpropData");
}
}  // namespace LayerTestsDefinitions
