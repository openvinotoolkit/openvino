// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <tuple>
#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <functional_test_utils/skip_tests_config.hpp>

#include "ie_core.hpp"

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"

#include "subgraph_tests/quantized_convolution_backprop_data.hpp"

using ngraph::helpers::QuantizationGranularity;

namespace LayerTestsDefinitions {

std::string QuantConvBackpropDataLayerTest::getTestCaseName(testing::TestParamInfo<quantConvBackpropDataLayerTestParamsSet> obj) {
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
    QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;

    std::ostringstream result;
    result << "IS=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "K" << CommonTestUtils::vec2str(kernel) << "_";
    result << "S" << CommonTestUtils::vec2str(stride) << "_";
    result << "PB" << CommonTestUtils::vec2str(padBegin) << "_";
    result << "PE" << CommonTestUtils::vec2str(padEnd) << "_";
    result << "D=" << CommonTestUtils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "Levels=" << quantLevels << "_";
    result << "QG=" << quantGranularity << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantConvBackpropDataLayerTest::SetUp() {
    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(groupConvBackpropDataParams, netPrecision, inputShape, targetDevice) = this->GetParam();
    ngraph::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    auto params = ngraph::builder::makeParams(ngPrc, {inputShape});
    auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::op::Parameter>(params));

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ngraph::helpers::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ngraph::builder::makeFakeQuantize(paramOuts[0], ngPrc, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {inputShape[1], convOutChannels};
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    std::vector<float> weightsData;
    auto weightsNode = ngraph::builder::makeConstant(ngPrc, weightsShapes, weightsData, weightsData.empty());

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ngraph::helpers::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    auto weightsFq = ngraph::builder::makeFakeQuantize(weightsNode, ngPrc, quantLevels, weightsFqConstShapes);

    auto convBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(dataFq, weightsFq, ngPrc, stride, padBegin, padEnd, dilation, padType));

    ngraph::ResultVector results{std::make_shared<ngraph::opset1::Result>(convBackpropData)};
    function = std::make_shared<ngraph::Function>(results, params, "QuantConvolutionBackpropData");
}

TEST_P(QuantConvBackpropDataLayerTest, CompareWithRefs) {
    Run();
}
}  // namespace LayerTestsDefinitions
