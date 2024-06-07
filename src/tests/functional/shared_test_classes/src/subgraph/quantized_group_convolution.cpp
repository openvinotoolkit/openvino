// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/quantized_group_convolution.hpp"
#include "common_test_utils/node_builders/group_convolution.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace test {

std::string QuantGroupConvLayerTest::getTestCaseName(const testing::TestParamInfo<quantGroupConvLayerTestParamsSet>& obj) {
    quantGroupConvSpecificParams groupConvParams;
    ov::element::Type element_type;
    ov::Shape inputShapes;
    std::string targetDevice;
    std::tie(groupConvParams, element_type, inputShapes, targetDevice) = obj.param;
    ov::op::PadType padType = ov::op::PadType::AUTO;
    ov::Shape kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
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
    result << "ET=" << element_type.get_type_name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void QuantGroupConvLayerTest::SetUp() {
    quantGroupConvSpecificParams groupConvParams;
    ov::Shape inputShape;
    ov::element::Type element_type = ov::element::undefined;
    std::tie(groupConvParams, element_type, inputShape, targetDevice) = this->GetParam();
    ov::op::PadType padType = ov::op::PadType::AUTO;
    ov::Shape kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels, numGroups;
    size_t quantLevels;
    ov::test::utils::QuantizationGranularity quantGranularity;
    bool quantizeWeights;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, numGroups, quantLevels, quantGranularity, quantizeWeights) = groupConvParams;
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(element_type, ov::Shape(inputShape))};

    std::vector<size_t> dataFqConstShapes(inputShape.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        dataFqConstShapes[1] = inputShape[1];
    auto dataFq = ov::test::utils::make_fake_quantize(params[0], element_type, quantLevels, dataFqConstShapes);

    std::vector<size_t> weightsShapes = {convOutChannels, inputShape[1]};
    if (weightsShapes[0] % numGroups || weightsShapes[1] % numGroups)
        throw std::runtime_error("incorrect shape for QuantGroupConvolution");
    weightsShapes[0] /= numGroups;
    weightsShapes[1] /= numGroups;
    weightsShapes.insert(weightsShapes.begin(), numGroups);
    weightsShapes.insert(weightsShapes.end(), kernel.begin(), kernel.end());

    auto weightsNode = ov::test::utils::make_constant(element_type, weightsShapes);

    std::vector<size_t> weightsFqConstShapes(weightsShapes.size(), 1);
    if (quantGranularity == ov::test::utils::QuantizationGranularity::Perchannel)
        weightsFqConstShapes[0] = weightsShapes[0];

    std::shared_ptr<ov::Node> weights;
    if (quantizeWeights) {
        weights = ov::test::utils::make_fake_quantize(weightsNode, element_type, quantLevels, weightsFqConstShapes);
    } else {
        weights = weightsNode;
    }

    auto groupConv = std::dynamic_pointer_cast<ov::op::v1::GroupConvolution>(
            ov::test::utils::make_group_convolution(dataFq, weights, element_type, stride, padBegin, padEnd, dilation, padType));

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(groupConv)};
    function = std::make_shared<ov::Model>(results, params, "QuantGroupConvolution");
}
}  // namespace test
}  // namespace ov
