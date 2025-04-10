// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/convolution_backprop_data.hpp"

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "low_precision/network_helper.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

namespace {
const std::shared_ptr<ov::opset1::ConvolutionBackpropData> buildConvBackpropData(const Output<Node>& data, const Output<Node>& weights) {
    const auto rank = data.get_partial_shape().rank();
    const auto rank_value = rank.is_static() ? rank.get_length() : 4;
    OPENVINO_ASSERT(rank_value == 3 || rank_value == 4,
                    "ConvolutionBackpropData test class doesn't support input shape ",
                    data.get_partial_shape());
    return std::make_shared<ov::opset1::ConvolutionBackpropData>(
        data,
        weights,
        rank_value == 4 ? Strides{1, 1} : Strides{1},
        rank_value == 4 ? CoordinateDiff{0, 0} : CoordinateDiff{0},
        rank_value == 4 ? CoordinateDiff{0, 0} : CoordinateDiff{0},
        rank_value == 4 ? Strides{1, 1} : Strides{1});
}
}  // namespace

std::shared_ptr<ov::Model> ConvolutionBackpropDataFunction::get(const ov::element::Type netPrecision,
                                                                const PartialShape& inputShape,
                                                                const Shape& outputShape,
                                                                const builder::subgraph::FakeQuantizeOnData& fqOnData,
                                                                const std::shared_ptr<Node>& weights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(netPrecision, inputShape);
    const auto fq = makeFakeQuantize(input, netPrecision, fqOnData);

    const auto convolutionBackpropData = buildConvBackpropData(fq, weights);
    convolutionBackpropData->set_friendly_name("convolutionBackpropData");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolutionBackpropData) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

std::shared_ptr<Node> ConvolutionBackpropDataFunction::getWeights(
    const Shape& shape,
    const ov::element::Type& netPrecision,
    const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
    const std::shared_ptr<ov::opset1::Constant>& value) {
    const auto weights =
        value != nullptr
            ? value
            : std::make_shared<ov::opset1::Constant>(ov::element::i8, shape, std::vector<float>(shape_size(shape), 1));
    const auto convert = std::make_shared<ov::opset1::Convert>(weights, netPrecision);
    OutputVector convertedOutput(1);
    convert->constant_fold(convertedOutput, convert->input_values());
    const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();
    const auto fq = makeFakeQuantize(convertedWeights, netPrecision, fqOnWeights);

    return fq;
}

std::shared_ptr<Node> ConvolutionBackpropDataFunction::getWeights(
    const Shape& shape,
    const ov::element::Type& netPrecision,
    const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const std::shared_ptr<ov::opset1::Constant>& value) {
    auto weights =
        value != nullptr
            ? value
            : std::make_shared<ov::opset1::Constant>(ov::element::i8, shape, std::vector<float>(shape_size(shape), 1));
    auto dequantizationStructure = dequantizationOnWeights;
    dequantizationStructure.setPrecision(netPrecision);
    if (!dequantizationOnWeights.subtract.constantPrecision.is_real()) {
        dequantizationStructure.subtract.constantPrecision = dequantizationOnWeights.subtract.constantPrecision;
    }
    if (weights->get_element_type().is_real()) {
        weights = ov::as_type_ptr<ov::opset1::Constant>(fold<ov::opset1::Convert>(weights, netPrecision));
    }
    const auto dq = makeDequantization(weights, dequantizationStructure);

    return dq;
}

std::shared_ptr<Node> ConvolutionBackpropDataFunction::getWeights(
    const Shape& shape,
    const ov::element::Type& netPrecision,
    const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
    const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
    const std::shared_ptr<ov::opset1::Constant>& value) {
    const auto weights =
        value != nullptr
            ? value
            : std::make_shared<ov::opset1::Constant>(ov::element::i8, shape, std::vector<float>(shape_size(shape), 1));
    const auto convert = std::make_shared<ov::opset1::Convert>(weights, netPrecision);
    OutputVector convertedOutput(1);
    convert->constant_fold(convertedOutput, convert->input_values());
    const auto convertedWeights = convertedOutput[0].get_node_shared_ptr();
    const auto fq = makeFakeQuantizeTypeRelaxed(convertedWeights, netPrecision, fqOnWeights);

    auto dequantizationStructure = dequantizationOnWeights;
    dequantizationStructure.setPrecision(netPrecision);
    return makeDequantization(fq, dequantizationStructure);
}

std::shared_ptr<ov::Model> ConvolutionBackpropDataFunction::getOriginal(
    const ov::element::Type precision,
    const ov::element::Type netPrecision,
    const PartialShape& inputShape,
    const Shape& outputShape,
    const builder::subgraph::DequantizationOperations& dequantization,
    const std::shared_ptr<Node>& weights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto activations = makeDequantization(input, dequantizationStructure);

    const auto convolutionBackpropData = buildConvBackpropData(activations, weights);
    convolutionBackpropData->set_friendly_name("output");
    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(convolutionBackpropData) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

std::shared_ptr<ov::Model> ConvolutionBackpropDataFunction::getReference(
    const ov::element::Type precision,
    const ov::element::Type netPrecision,
    const PartialShape& inputShape,
    const Shape& outputShape,
    const builder::subgraph::DequantizationOperations& dequantization,
    const std::shared_ptr<Node>& weights,
    const builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    auto dequantizationStructure = dequantization;
    dequantizationStructure.multiply.outPrecision = netPrecision;
    const auto activations = makeDequantization(input, dequantizationStructure);

    auto convolutionBackpropData = std::make_shared<ov::op::TypeRelaxed<ov::opset1::ConvolutionBackpropData>>(
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{dequantizationAfter.empty() ? netPrecision : ov::element::f32},
        ov::op::TemporaryReplaceOutputType(activations, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weights, ov::element::f32).get(),
        Strides{1, 1},
        CoordinateDiff{0, 0},
        CoordinateDiff{0, 0},
        Strides{1, 1});

    auto dequantizationStructureAfter = dequantizationAfter;
    dequantizationStructureAfter.multiply.outPrecision = netPrecision;
    const auto result = makeDequantization(convolutionBackpropData, dequantizationStructureAfter);
    result->set_friendly_name("output");
    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(result) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "ConvolutionBackpropDataTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
