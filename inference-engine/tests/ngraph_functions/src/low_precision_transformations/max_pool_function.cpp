// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(values.lowPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, originalFunctionPrecision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.subtractValues.size() }), values.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));
    parent = multiply;

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    maxPool->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

std::shared_ptr<ngraph::Function> MaxPoolFunction::getReference(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ExpectedValues& values) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(values.activationPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    parent = maxPool;

    if (parent->get_output_element_type(0) != originalFunctionPrecision) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::pass::low_precision::DequantizationConvert>(parent, originalFunctionPrecision);
        parent = convert;
    }

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::pass::low_precision::DequantizationSubtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.subtractValues.size() }), values.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::pass::low_precision::DequantizationMultiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(originalFunctionPrecision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));
    multiply->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
