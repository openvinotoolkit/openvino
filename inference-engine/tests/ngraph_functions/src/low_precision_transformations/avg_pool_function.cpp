// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/avg_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::element::Type lowPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(lowPrecision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto dequantizationBefore = makeDequantization(input, dequantization);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        dequantizationBefore,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    avgPool->set_friendly_name("avgPool");
    std::shared_ptr<Node> lastLayer = avgPool;

    if (additionalLayer == "maxpool") {
        lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
            lastLayer,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        lastLayer->set_friendly_name("maxPool");
    }

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, originalFunctionPrecision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 }, "fakeQuantize");
    }

    lastLayer->set_friendly_name("output");
    const auto result = std::make_shared<ngraph::opset1::Result>(lastLayer);
    result->set_friendly_name("result");

    return std::make_shared<ngraph::Function>(result, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        fakeQuantize,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(avgPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getReference(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::Shape& inputShape,
    const bool addFQ,
    const std::string additionalLayer,
    const ngraph::element::Type activationPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(activationPrecision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<ngraph::Node> pooling = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        input,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);
    pooling->set_friendly_name("avgPool");
    const auto avgPoolPrecision = addFQ ? originalFunctionPrecision : activationPrecision;
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(pooling, avgPoolPrecision);


    if (additionalLayer == "maxpool") {
        pooling = std::make_shared<ngraph::opset1::MaxPool>(
            pooling,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        pooling->set_friendly_name("maxPool");
    }

    auto dequantizationAfter = dequantization;
    dequantizationAfter.convert = {};
    std::shared_ptr<Node> lastLayer = makeDequantization(pooling, dequantizationAfter);

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, originalFunctionPrecision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 }, "fakeQuantize");
    }

    lastLayer->set_friendly_name("output");
    const auto result = std::make_shared<ngraph::opset1::Result>(lastLayer);
    result->set_friendly_name("result");

    return std::make_shared<ngraph::Function>(result, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
