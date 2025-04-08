// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/opsets/opset1.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/fake_quantize_on_weights_and_unsupported_child.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "low_precision/network_helper.hpp"


namespace ov {
namespace builder {
namespace subgraph {
std::shared_ptr<ov::Model> FakeQuantizeOnWeightsAndUnsupportedChildFunction::get(
    const ov::Shape& inputShape,
    const ov::element::Type inputPrecision,
    const std::shared_ptr<ov::opset1::Constant> weights,
    const ov::builder::subgraph::FakeQuantizeOnWeights fqOnWeights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(inputPrecision, inputShape);
    input->set_friendly_name("Input");
    weights->set_friendly_name("Weights");

    std::shared_ptr<ov::Node> weightsParent = weights;
    if (!fqOnWeights.empty()) {
        const auto fakeQuantizeOnWeights = makeFakeQuantize(weights, inputPrecision, fqOnWeights);
        fakeQuantizeOnWeights->set_friendly_name("FakeQuantize");
        weightsParent = fakeQuantizeOnWeights;
    }

    auto unsupportedOperation = std::make_shared<ov::opset1::ConvolutionBackpropData>(
        input, weightsParent, ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 }, ov::CoordinateDiff{ 0, 0 }, ov::Strides{ 1, 1 });
    unsupportedOperation->set_friendly_name("UnsupportedOperation");

    const auto result = std::make_shared<ov::opset1::Result>(unsupportedOperation);
    result->set_friendly_name("Result");

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ResultVector{ result },
        ov::ParameterVector{ input },
        "FakeQuantizeOnWeightsWithUnsupportedOperations");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
