// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/clamp_function.hpp"

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool updatePrecision,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        updatePrecision ? values.lowPrecision : originalFunctionPrecision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        auto constant = std::make_shared<ngraph::opset1::Constant>(
            originalFunctionPrecision,
            Shape({ 1, values.subtractValues.size(), 1, 1 }),
            values.subtractValues);
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            constant);
        parent = subtract;
    }
    if (!values.multiplyValues.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ 1, values.multiplyValues.size(), 1, 1 }), values.multiplyValues));
        parent = multiply;
    }

    const std::shared_ptr<ngraph::Node> clamp = std::make_shared<ngraph::opset1::Clamp>(parent, 0.0, 10.0);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampTransformation");
}

std::shared_ptr<ngraph::Function> ClampFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize,
    const double clampLowConst,
    const double clampHighConst) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    const std::shared_ptr<Node> fq = fakeQuantize.empty() ? nullptr :
        ngraph::builder::makeFakeQuantize(
            input,
            precision,
            fakeQuantize.quantizationLevel,
            fakeQuantize.constantShape,
            fakeQuantize.inputLowValues,
            fakeQuantize.inputHighValues,
            fakeQuantize.outputLowValues,
            fakeQuantize.outputHighValues);

    const std::shared_ptr<ngraph::opset1::Clamp> clamp = std::make_shared<ngraph::opset1::Clamp>(
        fakeQuantize.empty() ? input : fq,
        clampLowConst,
        clampHighConst);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(clamp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampFunction");
}

std::shared_ptr<ngraph::Function> ClampFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool updatePrecisions,
    const ExpectedValues& values) {
    if (values.subtractValues.size() > 1 || values.multiplyValues.size() > 1) {
        return getOriginal(precision, inputShape, updatePrecisions, { values.lowPrecision, values.subtractValues, values.multiplyValues });
    }

    auto input = std::make_shared<ngraph::opset1::Parameter>(values.lowPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> clamp = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Clamp>>(parent, 0.0, 10.0);
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(clamp, precision);
    parent = clamp;

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape({ 1, values.subtractValues.size(), 1, 1 }), values.subtractValues));
        parent = subtract;
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(subtract, precision);
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({ 1, values.multiplyValues.size(), 1, 1 }), values.multiplyValues));
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(multiply, precision);

    if (!updatePrecisions) {
        input = as_type_ptr<ngraph::opset1::Parameter>(replace_node(
            input,
            std::make_shared<ngraph::opset1::Parameter>(
                precision,
                ngraph::Shape(inputShape))));
    }

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "ClampTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
