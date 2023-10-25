// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/move_dequantization_after.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> MoveDequantizationAfterFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);

        const auto deq = makeDequantization(input, dequantization);
        const auto op = ngraph::opset1::MaxPool(
            deq,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        const auto targetOp = std::make_shared<ov::op::TypeRelaxed<opset1::MaxPool>>(
            op,
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(targetOp) },
            ngraph::ParameterVector{ input },
            "MoveDequantizationAfterFunction");
    }

    std::shared_ptr<ngraph::Function> MoveDequantizationAfterFunction::getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantizationBefore,
        const ngraph::element::Type precisionAfterOperation,
        const ngraph::builder::subgraph::DequantizationOperations dequantizationAfter) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);

        const auto deqBefore = makeDequantization(input, dequantizationBefore);
        const auto op = ngraph::opset1::MaxPool(
            deqBefore,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        const auto targetOp = std::make_shared<ov::op::TypeRelaxed<opset1::MaxPool>>(
            op,
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
        ov::pass::low_precision::NetworkHelper::setOutDataPrecisionForTypeRelaxed(targetOp, precisionAfterOperation);
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        const auto deqAfter = makeDequantization(targetOp, dequantizationAfter);

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(deqAfter) },
            ngraph::ParameterVector{ input },
            "MoveDequantizationAfterFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
