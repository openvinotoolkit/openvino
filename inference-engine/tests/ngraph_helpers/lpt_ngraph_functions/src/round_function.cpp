// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset1.hpp>

#include "lpt_ngraph_functions/round_function.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

#include "ngraph_functions/subgraph_builders.hpp"

using namespace ngraph::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> RoundWithToleranceFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto deq = makeDequantization(input, dequantization);
        deq->set_friendly_name("output");

        const auto result = std::make_shared<ngraph::opset1::Result>(deq);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "RoundWithToleranceFunction");
    }

    std::shared_ptr<ngraph::Function> RoundWithToleranceFunction::getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::builder::subgraph::DequantizationOperations dequantization) {
        const auto input = std::make_shared<ngraph::op::v0::Parameter>(precision, inputShape);
        input->set_friendly_name("input");

        const auto deq = makeDequantization(input, dequantization);
        deq->set_friendly_name("output");

        const auto result = std::make_shared<ngraph::opset1::Result>(deq);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{ result },
            ngraph::ParameterVector{ input },
            "RoundWithToleranceFunction");
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
