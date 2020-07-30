// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fuse_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const std::shared_ptr<Node> lastDequantization = makeDequantization(input, dequantization);
    const std::shared_ptr<Node> fakeQuantize = precisionAfterDequantization == precisionFqOnData ?
            makeFakeQuantize(lastDequantization, precisionFqOnData, fqOnData) :
            makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnData);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::Shape& inputShape,
    const std::vector<Branch>& branches,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    if (branches.size() != 2ul) {
        THROW_IE_EXCEPTION << "unsupported branches count";
    }

    if (branches[0].dequantization.multiply.outPrecision != branches[1].dequantization.multiply.outPrecision) {
        THROW_IE_EXCEPTION << "branch precisions are not equal";
    }

    ngraph::ParameterVector inputs;
    std::vector<std::shared_ptr<Node>> lastDequantizations;
    for (const Branch& branch : branches) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(branch.precisionBeforeDequantization, ngraph::Shape(inputShape));
        inputs.push_back(input);

        const std::shared_ptr<Node> lastDequantization = makeDequantization(input, branch.dequantization);
        lastDequantizations.push_back(lastDequantization);
    }

    std::shared_ptr<ngraph::opset1::Multiply> multiply = std::make_shared<ngraph::opset1::Multiply>(lastDequantizations[0], lastDequantizations[1]);

    const std::shared_ptr<Node> fakeQuantize = branches[0].dequantization.multiply.outPrecision == precisionFqOnData ?
        makeFakeQuantize(multiply, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(multiply, precisionFqOnData, fqOnData);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, inputs, "FuseFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
