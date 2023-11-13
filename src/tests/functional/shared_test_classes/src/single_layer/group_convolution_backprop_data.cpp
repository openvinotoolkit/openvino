// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/group_convolution_backprop_data.hpp"

namespace LayerTestsDefinitions {

// DEPRECATED, remove this old API when KMB (#58495) and ARM (#58496) plugins are migrated to new API

std::string GroupConvBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<groupConvBackpropDataLayerTestParamsSet>& obj) {
    groupConvBackpropDataSpecificParams groupConvBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "G=" << numGroups << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void GroupConvBackpropDataLayerTest::SetUp() {
    groupConvBackpropDataSpecificParams groupConvBackpropDataParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType) = groupConvBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto groupConvBackpropData = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
            ngraph::builder::makeGroupConvolutionBackpropData(params[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, numGroups));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(groupConvBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "GroupConvolutionBackpropData");
}

std::string GroupConvBackpropLayerTest::getTestCaseName(testing::TestParamInfo<groupConvBackpropLayerTestParamsSet> obj) {
    groupConvBackpropSpecificParams groupConvBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes, outputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, outputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType, outPadding) = groupConvBackpropDataParams;

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
    result << "G=" << numGroups << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    return result.str();
}

void GroupConvBackpropLayerTest::SetUp() {
    groupConvBackpropSpecificParams groupConvBackpropDataParams;
    std::vector<size_t> inputShape, outputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvBackpropDataParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, outputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd, outPadding;
    size_t convOutChannels, numGroups;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, padType, outPadding) = groupConvBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    std::shared_ptr<ngraph::op::v1::GroupConvolutionBackpropData> groupConvBackpropData;
    if (!outputShape.empty()) {
        auto outShape = ngraph::opset3::Constant::create(ngraph::element::i64, {outputShape.size()}, outputShape);
        groupConvBackpropData = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
        ngraph::builder::makeGroupConvolutionBackpropData(params[0], outShape, ngPrc, kernel, stride, padBegin,
                                                        padEnd, dilation, padType, convOutChannels, numGroups, false, outPadding));
    } else {
        groupConvBackpropData = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(
        ngraph::builder::makeGroupConvolutionBackpropData(params[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, numGroups, false, outPadding));
    }
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(groupConvBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "GroupConvolutionBackpropData");
}
}  // namespace LayerTestsDefinitions
