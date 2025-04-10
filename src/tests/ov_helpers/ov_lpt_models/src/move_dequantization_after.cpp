// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/move_dequantization_after.hpp"
#include "low_precision/network_helper.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {
    std::shared_ptr<ov::Model> MoveDequantizationAfterFunction::getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization,
        const bool typeRelaxed) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);

        const auto deq = makeDequantization(input, dequantization);
        const auto targetOp = typeRelaxed ? std::make_shared<ov::op::TypeRelaxed<ov::opset1::MaxPool>>(
                                                std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
                                                std::vector<ov::element::Type>{},
                                                deq,
                                                Strides{1, 1},
                                                Shape{1, 1},
                                                Shape{0, 0},
                                                Shape{2, 2},
                                                ov::op::RoundingType::FLOOR)
                                          : std::make_shared<ov::opset1::MaxPool>(deq,
                                                                                  Strides{1, 1},
                                                                                  Shape{1, 1},
                                                                                  Shape{0, 0},
                                                                                  Shape{2, 2},
                                                                                  ov::op::RoundingType::FLOOR);
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        return std::make_shared<ov::Model>(
            ov::ResultVector{ std::make_shared<ov::opset1::Result>(targetOp) },
            ov::ParameterVector{ input },
            "MoveDequantizationAfterFunction");
    }

    std::shared_ptr<ov::Model> MoveDequantizationAfterFunction::getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const ov::builder::subgraph::DequantizationOperations dequantizationAfter,
        const bool typeRelaxed) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);

        const auto deqBefore = makeDequantization(input, dequantizationBefore);
        const auto targetOp = typeRelaxed ? std::make_shared<ov::op::TypeRelaxed<ov::opset1::MaxPool>>(
                                                std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
                                                std::vector<ov::element::Type>{precisionAfterOperation},
                                                deqBefore,
                                                Strides{1, 1},
                                                Shape{1, 1},
                                                Shape{0, 0},
                                                Shape{2, 2},
                                                ov::op::RoundingType::FLOOR)
                                          : std::make_shared<ov::opset1::MaxPool>(deqBefore,
                                                                                  Strides{1, 1},
                                                                                  Shape{1, 1},
                                                                                  Shape{0, 0},
                                                                                  Shape{2, 2},
                                                                                  ov::op::RoundingType::FLOOR);
        auto& rtInfo = targetOp->get_rt_info();
        rtInfo["Variant::std::string"] = "targetOp";

        const auto deqAfter = makeDequantization(targetOp, dequantizationAfter);

        return std::make_shared<ov::Model>(
            ov::ResultVector{ std::make_shared<ov::opset1::Result>(deqAfter) },
            ov::ParameterVector{ input },
            "MoveDequantizationAfterFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
