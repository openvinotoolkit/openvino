// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/normalize_dequantization_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "ngraph_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

    std::shared_ptr<ngraph::Function> NormalizeDequantizationFunction::getOriginal(
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
        const auto targetOp = std::make_shared<op::TypeRelaxed<opset1::MaxPool>>(
            op,
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(targetOp) },
            ngraph::ParameterVector{ input },
            "NormalizeDequantizationFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph