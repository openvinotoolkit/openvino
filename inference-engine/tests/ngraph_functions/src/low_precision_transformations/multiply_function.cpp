// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/multiply_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MultiplyFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape1,
    const ngraph::Shape& inputShape2,
    const FakeQuantizeOnData& fq1,
    const FakeQuantizeOnData& fq2) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape1));
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape2));
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1, input2 }, "MultiplyTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
