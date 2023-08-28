// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_group_convolution.hpp"

using ngraph::helpers::QuantizationGranularity;

namespace SubgraphTestsDefinitions {

std::string QuantGroupConvLayerTest::getTestCaseName(const testing::TestParamInfo<quantGroupConvLayerTestParamsSet>& obj) {
    quantGroupConvSpecificParams groupConvParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(groupConvParams, netPrecision, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    bool quantizeWeights;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, quantLevels, quantGranularity, quantizeWeights) = groupConvParams;

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
    result << "Levels=" << quantLevels << "_";
    result << "QG=" << quantGranularity << "_";
    result << "QW=" << quantizeWeights << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantGroupConvLayerTest::SetUp() {
    threshold = 0.5f;

    quantGroupConvSpecificParams groupConvParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType = ngraph::op::PadType::AUTO;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    size_t quantLevels;
    size_t quantGranularity;
    bool quantizeWeights;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, quantLevels, quantGranularity, quantizeWeights) = groupConvParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ngraph::helpers::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1]};
    if (weightsShapes[0] % numGroups || weightsShapes[1] % numGroups)
        throw std::runtime_error("incorrect shape for QuantGroupConvolution");
    weightsShapes[0] /= numGroups;
    weightsShapes[1] /= numGroups;
    weightsShapes.insert(weightsShapes.begin(), numGroups);
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    std::vector<float> weightsData;
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShapes, weightsData, weightsData.empty());

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ngraph::helpers::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    std::shared_ptr<ngraph::Node> weights;
    if (quantizeWeights) {
        weights = ngraph::builder::makeFakeQuantize(weightsNode, ngPrc, quantLevels, weightsFqConstShapes);
    } else {
        weights = weightsNode;
    }

    auto groupConv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(
            ngraph::builder::makeGroupConvolution(dataFq, weights, ngPrc, stride, padBegin, padEnd, dilation, padType));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(groupConv)};
    function = std::make_shared<ngraph::Function>(results, params, "QuantGroupConvolution");
}
}  // namespace SubgraphTestsDefinitions
