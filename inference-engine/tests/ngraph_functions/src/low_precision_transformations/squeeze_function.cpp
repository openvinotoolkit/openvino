// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/squeeze_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SqueezeFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const std::vector<float>& axes,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(values.lowPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    // TODO: pass as test case argument
    if (values.lowPrecision != originalFunctionPrecision) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
        parent = convert;
    }

    if (!values.subtract.values.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.subtract.shape, values.subtract.values));
        parent = subtract;
    }

    if (!values.mutliply.values.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.mutliply.shape, values.mutliply.values));
        parent = multiply;
    }

    const std::shared_ptr<ngraph::Node> squeeze = std::make_shared<ngraph::opset1::Squeeze>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));
    squeeze->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(squeeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ngraph::Function> SqueezeFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& axes) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = fakeQuantizeOnData.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
            fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> squeeze = std::make_shared<ngraph::opset1::Squeeze>(
        fakeQuantize == nullptr ? input : fakeQuantize,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(squeeze) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}

std::shared_ptr<ngraph::Function> SqueezeFunction::getReference(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const std::vector<float>& axes,
    const ExpectedValues& values) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(values.activationPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> squeeze = std::make_shared<ngraph::opset1::Squeeze>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(element::i64, Shape{ axes.size() }, axes));
    parent = squeeze;

    // TODO: pass as test case argument
    if (parent->output(0).get_element_type() != originalFunctionPrecision) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
        parent = convert;
    }

    if (!values.subtract.values.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.subtract.shape, values.subtract.values));
        parent = subtract;
    }

    if (!values.mutliply.values.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, values.mutliply.shape, values.mutliply.values));
        parent = multiply;
    }

    if (values.activationPrecision != originalFunctionPrecision) {
        input = as_type_ptr<ngraph::opset1::Parameter>(replace_node(
            input,
            std::make_shared<ngraph::opset1::Parameter>(values.activationPrecision, ngraph::Shape(inputShape))));
    }
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SqueezeTransformation");
}



}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
