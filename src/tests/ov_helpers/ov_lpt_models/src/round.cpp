// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"

#include "ov_lpt_models/round.hpp"
#include "ov_lpt_models/common/builders.hpp"


using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {
    std::shared_ptr<ov::Model> RoundWithToleranceFunction::getOriginal(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto deq = makeDequantization(input, dequantization);
        deq->set_friendly_name("output");

        const auto result = std::make_shared<ov::opset1::Result>(deq);
        result->set_friendly_name("result");

        return std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ input },
            "RoundWithToleranceFunction");
    }

    std::shared_ptr<ov::Model> RoundWithToleranceFunction::getReference(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const ov::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ov::op::v0::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto deq = makeDequantization(input, dequantization);
        deq->set_friendly_name("output");

        const auto result = std::make_shared<ov::opset1::Result>(deq);
        result->set_friendly_name("result");

        return std::make_shared<ov::Model>(
            ov::ResultVector{ result },
            ov::ParameterVector{ input },
            "RoundWithToleranceFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
