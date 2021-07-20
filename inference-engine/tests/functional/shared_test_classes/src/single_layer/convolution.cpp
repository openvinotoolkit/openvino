// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/single_layer/convolution.hpp"

namespace LayerTestsDefinitions {

std::string ConvolutionLayerTest::getTestCaseName(testing::TestParamInfo<convLayerTestParamsSet> obj) {
    convSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    std::vector<std::vector<size_t>> inputShape;
    InferenceEngine::SizeVector targetShape;
    std::string targetDevice;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetShape, targetDevice) =
        obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShape) << "_";
    result << "TS=" << CommonTestUtils::vec2str(targetShape) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
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

void ConvolutionLayerTest::SetUp() {
    convSpecificParams convParams;
    std::vector<std::vector<size_t>> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetStaticShape, targetDevice) =
        this->GetParam();
    inputDynamicShape = vec2partialshape(inputShape);
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {targetStaticShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(
            ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));
    std::vector<float> filter_weights;
    if (targetDevice == CommonTestUtils::DEVICE_GNA) {
        auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
        filter_weights = CommonTestUtils::generate_float_numbers(convOutChannels * targetStaticShape[1] * filter_size,
                                                                 -0.5f, 0.5f);
    }
    auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(
            ngraph::builder::makeConvolution(paramOuts[0], ngPrc, kernel, stride, padBegin,
                                             padEnd, dilation, padType, convOutChannels, false, filter_weights));
    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(conv)};
    function = std::make_shared<ngraph::Function>(results, params, "convolution");
}

ngraph::PartialShape ConvolutionLayerTest::vec2partialshape(std::vector<std::vector<size_t>> inputShape) {
    if (inputShape.empty()) {
        for (auto&& item : targetStaticShape) {
            inputShape.push_back({item, item});
        }
    }
    std::vector<ngraph::Dimension> dimensions;
    dimensions.reserve(inputShape.size());
    for (auto&& item : inputShape) {
        dimensions.emplace_back(item[0], item[1]);
    }
    return ngraph::PartialShape(dimensions);
}
}  // namespace LayerTestsDefinitions
