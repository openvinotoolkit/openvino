// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_convolution_backprop_data.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ov {
namespace test {

std::string QuantConvBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<quantConvBackpropDataLayerTestParamsSet>& obj) {
    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    ov::element::Type element_type;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, element_type, inputShapes, targetDevice) = obj.param;
    ngraph::op::PadType padType;
    ov::Shape kernel, stride, dilation;
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
    result << "netPRC=" << element_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantConvBackpropDataLayerTest::SetUp() {
    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    ov::Shape inputShape;
    ov::element::Type element_type = ov::element::undefined;
    std::tie(groupConvBackpropDataParams, element_type, inputShape, targetDevice) = this->GetParam();
    ov::op::PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, inputShape)};

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ngraph::builder::makeFakeQuantize(params[0], element_type, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {inputShape[1], convOutChannels};
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    std::vector<float> weightsData;
    auto weightsNode = ngraph::builder::makeConstant(element_type, weightsShapes, weightsData, weightsData.empty());

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    auto weightsFq = ngraph::builder::makeFakeQuantize(weightsNode, element_type, quantLevels, weightsFqConstShapes);

    auto convBackpropData = std::dynamic_pointer_cast<ov::op::v1::ConvolutionBackpropData>(
            ngraph::builder::makeConvolutionBackpropData(dataFq, weightsFq, element_type, stride, padBegin, padEnd, dilation, padType));

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convBackpropData)};
    function = std::make_shared<ov::Model>(results, params, "QuantConvolutionBackpropData");
}
}  // namespace test
}  // namespace ov
