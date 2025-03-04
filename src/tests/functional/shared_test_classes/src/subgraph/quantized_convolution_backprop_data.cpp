// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/convolution_backprop_data.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {

std::string QuantConvBackpropDataLayerTest::getTestCaseName(const testing::TestParamInfo<quantConvBackpropDataLayerTestParamsSet>& obj) {
    quantConvBackpropDataSpecificParams groupConvBackpropDataParams;
    ov::element::Type element_type;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(groupConvBackpropDataParams, element_type, inputShapes, targetDevice) = obj.param;
    ov::op::PadType padType;
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
    ov::element::Type element_type = ov::element::dynamic;
    std::tie(groupConvBackpropDataParams, element_type, inputShape, targetDevice) = this->GetParam();
    ov::op::PadType padType;
    std::vector<size_t> kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType, quantLevels, quantGranularity) = groupConvBackpropDataParams;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, inputShape)};

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ov::test::utils::make_fake_quantize(params[0], element_type, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {inputShape[1], convOutChannels};
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    auto weightsNode = ov::test::utils::make_constant(element_type, weightsShapes);

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    auto weightsFq = ov::test::utils::make_fake_quantize(weightsNode, element_type, quantLevels, weightsFqConstShapes);

    auto convBackpropData = ov::as_type_ptr<ov::op::v1::ConvolutionBackpropData>(
            ov::test::utils::make_convolution_backprop_data(dataFq, weightsFq, element_type, stride, padBegin, padEnd, dilation, padType));

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(convBackpropData)};
    function = std::make_shared<ov::Model>(results, params, "QuantConvolutionBackpropData");
}
}  // namespace test
}  // namespace ov
