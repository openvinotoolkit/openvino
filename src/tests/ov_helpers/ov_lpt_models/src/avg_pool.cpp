// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

#include "ov_lpt_models/avg_pool.hpp"
#include "ov_models/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const bool addFQ,
    const std::vector<std::string>& additionalLayers,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    auto deqBeforeStructure = dequantizationBefore;
    deqBeforeStructure.multiply.outPrecision = precision;
    const auto dequantization = makeDequantization(input, deqBeforeStructure);

    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ngraph::opset1::AvgPool>(
        dequantization,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        true,
        op::RoundingType::FLOOR);

    std::shared_ptr<Node> lastLayer = avgPool;
    for (const std::string& additionalLayer : additionalLayers) {
        if (additionalLayer == "maxpool") {
            lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
                lastLayer,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                op::RoundingType::FLOOR);
        } else if (additionalLayer == "softmax") {
            lastLayer = std::make_shared<opset1::Softmax>(lastLayer);
        } else if (additionalLayer == "convolution") {
            lastLayer = makeConvolution(lastLayer, precision, false);
        } else if (additionalLayer == "unsupported_convolution") {
            lastLayer = makeConvolution(lastLayer, precision, true, element::f32);
        }
    }

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ngraph::Function> AvgPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

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
    const ngraph::element::Type precision,
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const bool addFQ,
    const std::vector<std::string>& additionalLayers,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationEnd) {
    auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);

    const auto deqBefore = makeDequantization(input, dequantizationBefore);
    auto outPrecision = precisionAfterOperation;
    const std::shared_ptr<ngraph::Node> avgPool = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::AvgPool>>(
        opset1::AvgPool(
            deqBefore,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            true,
            op::RoundingType::FLOOR),
        outPrecision);

    std::shared_ptr<Node> lastLayer = avgPool;

    auto deqStructure = dequantizationAfter;
    deqStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqStructure);

    for (const std::string& additionalLayer : additionalLayers) {
        if (additionalLayer == "maxpool") {
            lastLayer = std::make_shared<ngraph::opset1::MaxPool>(
                lastLayer,
                Strides{ 1, 1 },
                Shape{ 1, 1 },
                Shape{ 0, 0 },
                Shape{ 2, 2 },
                op::RoundingType::FLOOR);
        } else if (additionalLayer == "softmax") {
            lastLayer = std::make_shared<opset1::Softmax>(lastLayer);
        } else if (additionalLayer == "convolution") {
            lastLayer = makeConvolution(lastLayer, element::f32, dequantizationAfter.empty());
        } else if (additionalLayer == "unsupported_convolution") {
            lastLayer = makeConvolution(lastLayer, precision, true, element::f32);
        }
    }

    deqStructure = dequantizationEnd;
    deqStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqStructure);

    if (addFQ) {
        lastLayer = ngraph::builder::makeFakeQuantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastLayer) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "AvgPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
