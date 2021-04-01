// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include <ngraph/opsets/opset1.hpp>
#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/fake_quantize_on_weights_and_unsupported_child_function.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_weights.hpp"
#include "low_precision/network_helper.hpp"
#include "ngraph_functions/builders.hpp"


namespace ngraph {
namespace builder {
namespace subgraph {
std::shared_ptr<ngraph::Function> FakeQuantizeOnWeightsAndUnsupportedChildFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const std::shared_ptr<ngraph::opset1::Constant> weights,
    const ngraph::builder::subgraph::FakeQuantizeOnWeights fqOnWeights) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("Input");
    weights->set_friendly_name("Weights");

    std::shared_ptr<ngraph::Node> weightsParent = weights;
    if (!fqOnWeights.empty()) {
        const auto fakeQuantizeOnWeights = makeFakeQuantize(weights, inputPrecision, fqOnWeights);
        fakeQuantizeOnWeights->set_friendly_name("FakeQuantize");
        weightsParent = fakeQuantizeOnWeights;
    }

    auto unsupportedOperation = std::make_shared<ngraph::opset1::ConvolutionBackpropData>(
        input, weightsParent, ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 }, ngraph::CoordinateDiff{ 0, 0 }, ngraph::Strides{ 1, 1 });
    unsupportedOperation->set_friendly_name("UnsupportedOperation");

    const auto result = std::make_shared<ngraph::opset1::Result>(unsupportedOperation);
    result->set_friendly_name("Result");

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ResultVector{ result },
        ngraph::ParameterVector{ input },
        "FakeQuantizeOnWeightsWithUnsupportedOperations");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
