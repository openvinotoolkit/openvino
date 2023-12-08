// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/normalize_dequantization.hpp"

#include "ov_models/subgraph_builders.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_ops/type_relaxed.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

    std::shared_ptr<ov::Model> NormalizeDequantizationFunction::getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);

        const auto deq = makeDequantization(input, dequantization);

        const auto op = ov::opset1::MaxPool(
            deq,
            Strides{ 1, 1 },
            Shape{ 1, 1 },
            Shape{ 0, 0 },
            Shape{ 2, 2 },
            op::RoundingType::FLOOR);
        const auto targetOp = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MaxPool>>(
            op,
            std::vector<element::Type>{ element::f32, element::f32 },
            std::vector<element::Type>{});
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        return std::make_shared<ov::Model>(
            ov::ResultVector{ std::make_shared<ov::opset1::Result>(targetOp) },
            ov::ParameterVector{ input },
            "NormalizeDequantizationFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
