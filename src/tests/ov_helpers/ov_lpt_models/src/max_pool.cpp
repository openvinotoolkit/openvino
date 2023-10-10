// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/max_pool.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type originalFunctionPrecision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(originalFunctionPrecision, inputShape);

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

std::shared_ptr<ngraph::Function> MaxPoolFunction::get(
    const ngraph::PartialShape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ngraph::element::Type precisionAfterOperation,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    parent = makeDequantization(parent, dequantizationBefore);

    const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    parent = maxPool;
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(maxPool, precisionAfterOperation);

    parent = makeDequantization(maxPool, dequantizationAfter);
    maxPool->set_friendly_name("maxPool");

    const std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(parent);

    const std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
        "MaxPoolTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
