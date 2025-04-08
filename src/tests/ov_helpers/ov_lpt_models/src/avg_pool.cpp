// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>

#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

#include "ov_lpt_models/avg_pool.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> AvgPoolFunction::getOriginal(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const bool addFQ,
    const std::vector<std::string>& additionalLayers,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    std::shared_ptr<ov::Node> parent = input;

    auto deqBeforeStructure = dequantizationBefore;
    deqBeforeStructure.multiply.outPrecision = precision;
    const auto dequantization = makeDequantization(input, deqBeforeStructure);

    const std::shared_ptr<ov::Node> avgPool = std::make_shared<ov::opset1::AvgPool>(dequantization,
                                                                                    Strides{1, 1},
                                                                                    Shape{1, 1},
                                                                                    Shape{0, 0},
                                                                                    Shape{2, 2},
                                                                                    true,
                                                                                    ov::op::RoundingType::FLOOR);

    std::shared_ptr<Node> lastLayer = avgPool;
    for (const std::string& additionalLayer : additionalLayers) {
        if (additionalLayer == "maxpool") {
            lastLayer = std::make_shared<ov::opset1::MaxPool>(lastLayer,
                                                              Strides{1, 1},
                                                              Shape{1, 1},
                                                              Shape{0, 0},
                                                              Shape{2, 2},
                                                              ov::op::RoundingType::FLOOR);
        } else if (additionalLayer == "softmax") {
            lastLayer = std::make_shared<ov::opset1::Softmax>(lastLayer);
        } else if (additionalLayer == "convolution") {
            lastLayer = makeConvolution(lastLayer, precision, false);
        } else if (additionalLayer == "unsupported_convolution") {
            lastLayer = makeConvolution(lastLayer, precision, true, ov::element::f32);
        }
    }

    if (addFQ) {
        lastLayer = ov::test::utils::make_fake_quantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastLayer) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ov::Model> AvgPoolFunction::getOriginal(
    const ov::element::Type originalFunctionPrecision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(originalFunctionPrecision, inputShape);

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        input, originalFunctionPrecision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);

    const std::shared_ptr<ov::Node> avgPool = std::make_shared<ov::opset1::AvgPool>(fakeQuantize,
                                                                                    Strides{1, 1},
                                                                                    Shape{1, 1},
                                                                                    Shape{0, 0},
                                                                                    Shape{2, 2},
                                                                                    true,
                                                                                    ov::op::RoundingType::FLOOR);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(avgPool) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "AvgPoolTransformation");
}

std::shared_ptr<ov::Model> AvgPoolFunction::getReference(
    const ov::element::Type precision,
    const ov::element::Type inputPrecision,
    const ov::PartialShape& inputShape,
    const bool addFQ,
    const std::vector<std::string>& additionalLayers,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const ov::builder::subgraph::DequantizationOperations& dequantizationEnd) {
    auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);

    const auto deqBefore = makeDequantization(input, dequantizationBefore);
    auto outPrecision = precisionAfterOperation;
    const std::shared_ptr<ov::Node> avgPool =
        std::make_shared<ov::op::TypeRelaxed<ov::opset1::AvgPool>>(ov::opset1::AvgPool(deqBefore,
                                                                                       Strides{1, 1},
                                                                                       Shape{1, 1},
                                                                                       Shape{0, 0},
                                                                                       Shape{2, 2},
                                                                                       true,
                                                                                       ov::op::RoundingType::FLOOR),
                                                                   outPrecision);

    std::shared_ptr<Node> lastLayer = avgPool;

    auto deqStructure = dequantizationAfter;
    deqStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqStructure);

    for (const std::string& additionalLayer : additionalLayers) {
        if (additionalLayer == "maxpool") {
            lastLayer = std::make_shared<ov::opset1::MaxPool>(lastLayer,
                                                              Strides{1, 1},
                                                              Shape{1, 1},
                                                              Shape{0, 0},
                                                              Shape{2, 2},
                                                              ov::op::RoundingType::FLOOR);
        } else if (additionalLayer == "softmax") {
            lastLayer = std::make_shared<ov::opset1::Softmax>(lastLayer);
        } else if (additionalLayer == "convolution") {
            lastLayer = makeConvolution(lastLayer, ov::element::f32, dequantizationAfter.empty());
        } else if (additionalLayer == "unsupported_convolution") {
            lastLayer = makeConvolution(lastLayer, precision, true, ov::element::f32);
        }
    }

    deqStructure = dequantizationEnd;
    deqStructure.multiply.outPrecision = precision;
    lastLayer = makeDequantization(lastLayer, deqStructure);

    if (addFQ) {
        lastLayer = ov::test::utils::make_fake_quantize(
            lastLayer, precision, 256, {}, { 0 }, { 255 }, { 0 }, { 255 });
    }

    lastLayer->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastLayer) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "AvgPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
