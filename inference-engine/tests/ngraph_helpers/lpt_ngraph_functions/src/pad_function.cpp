// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>


#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/pad_function.hpp"
#include "low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> PadFunction::get(
    const Shape& inputShape,
    const element::Type precisionBeforeDequantization,
    const builder::subgraph::DequantizationOperations& dequantizationBefore,
    const std::vector<uint64_t>& padsBegin,
    const std::vector<uint64_t>& padsEnd,
    const op::PadMode mode,
    const float padValue,
    const element::Type precisionAfterOperation,
    const builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<opset1::Parameter>(precisionBeforeDequantization, inputShape);
    const auto deqBefore = makeDequantization(input, dequantizationBefore);

    const auto padsBeginConst = opset1::Constant::create(element::u64, Shape{ padsBegin.size() }, padsBegin);
    const auto padsEndConst = opset1::Constant::create(element::u64, Shape{ padsEnd.size() }, padsEnd);
    const auto padsValueConst = opset1::Constant::create(deqBefore->get_output_element_type(0), Shape{}, { padValue });

    const auto pad = std::make_shared<ngraph::op::TypeRelaxed<opset1::Pad>>(
        std::vector<element::Type>{ precisionAfterOperation, element::u64, element::u64, precisionAfterOperation},
        std::vector<element::Type>{ precisionAfterOperation },
        op::TemporaryReplaceOutputType(deqBefore, precisionAfterOperation).get(),
        op::TemporaryReplaceOutputType(padsBeginConst, element::u64).get(),
        op::TemporaryReplaceOutputType(padsEndConst, element::u64).get(),
        op::TemporaryReplaceOutputType(padsValueConst, precisionAfterOperation).get(),
        mode);

    const auto deqAfter = makeDequantization(pad, dequantizationAfter);
    deqAfter->set_friendly_name("Pad");

    auto function = std::make_shared<ngraph::Function>(
        ResultVector{ std::make_shared<ngraph::opset1::Result>(deqAfter) },
        ngraph::ParameterVector{ input }, "PadFunction");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
