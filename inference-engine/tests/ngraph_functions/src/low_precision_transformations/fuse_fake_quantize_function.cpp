// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    Shape multiplyConstShape(4ul, 1ul);
    multiplyConstShape[1] = inputShape[1];
    std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(
        input,
        ngraph::builder::makeConstant(precision, multiplyConstShape, {}, true));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        multiply, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& expectedSubtractValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    // TODO: way to instantiate TypeRelaxed FakeQuantize
    // TODO: use wrapper later
    auto inputLowNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputLowValues.empty());
    auto inputHighNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.inputHighValues.empty());
    auto outputLowNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputLowValues.empty());
    auto outputHighNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.outputHighValues, fakeQuantizeOnData.outputHighValues.empty());
    // auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
    //    input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fakeQuantizeOnData.quantizationLevel);
    std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(
        input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fakeQuantizeOnData.quantizationLevel);

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, params.precisionsOnActivations[0]);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
