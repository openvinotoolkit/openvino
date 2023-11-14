// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_convolution_backprop_data.hpp"

namespace SubgraphTestsDefinitions {

std::string QuantConvBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<quantConvBackpropDataLayerTestParamsSet>& obj) {
    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, netPrecision, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "Levels=" << quantLevels << "_";
    result << "QG=" << quantGranularity << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantConvBackpropDataLayerTest::SetUp() {
    threshold = 0.5f;

    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvBackpropDataParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ngraph::builder::makeFakeQuantize(params[0], ngPrc, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {inputShape[1], convOutChannels};
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    std::vector<float> weightsData;
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShapes, weightsData, weightsData.empty());

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    auto weightsFq = ngraph::builder::makeFakeQuantize(weightsNode, ngPrc, quantLevels, weightsFqConstShapes);

    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(dataFq, weightsFq, ngPrc, stride, padBegin, padEnd, dilation, padType));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "QuantConvolutionBackpropData");
}
}  // namespace SubgraphTestsDefinitions
