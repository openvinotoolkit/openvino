// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::element::Type lowPrecision,
    const ngraph::builder::subgraph::DequantizationOperations dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(lowPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;
    input->set_friendly_name("input");

    parent = makeDequantization(parent, dequantization);

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    maxPool->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxPool) };
    results[0]->set_friendly_name("result");
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
    const ngraph::element::Type activationPrecision,
    const ngraph::builder::subgraph::DequantizationOperations dequantization) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(activationPrecision, ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    maxPool->set_friendly_name("output");
    parent = maxPool;

    if (parent->get_output_element_type(0) == originalFunctionPrecision) {
        auto newDequantization = dequantization;
        newDequantization.convert = {};
        parent = makeDequantization(parent, newDequantization);
    } else {
        parent = makeDequantization(parent, dequantization);
    }

    maxPool->set_friendly_name("output_original");
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    results[0]->set_friendly_name("result");
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
