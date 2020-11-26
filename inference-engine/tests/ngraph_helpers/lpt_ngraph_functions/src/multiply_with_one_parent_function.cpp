// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/multiply_with_one_parent_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyWithOneParentFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
            input, precision, fqOnData.quantizationLevel, fqOnData.constantShape,
        fqOnData.inputLowValues, fqOnData.inputHighValues, fqOnData.outputLowValues, fqOnData.outputHighValues);

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(fakeQuantize->output(0), fakeQuantize->output(0));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MultiplyWithOneParentFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
