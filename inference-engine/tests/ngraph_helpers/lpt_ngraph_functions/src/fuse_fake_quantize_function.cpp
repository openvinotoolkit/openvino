// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/fuse_fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeAdd,
    const Add& add,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        add.empty() ? precisionBeforeDequantization : precisionBeforeAdd,
        ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    std::shared_ptr<Node> parent = input;
    if (!add.empty()) {
        parent = makeElementwise<ngraph::opset1::Add>(parent, add);
    }

    const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

    const std::shared_ptr<Node> fakeQuantize = precisionAfterDequantization == precisionFqOnData ?
        makeFakeQuantize(lastDequantization, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnData);
    fakeQuantize->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

    std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::getReference(
            const ngraph::Shape& inputShape,
            const ngraph::element::Type precisionBeforeAdd,
            const Add& add,
            const ngraph::element::Type precisionBeforeDequantization,
            const DequantizationOperations& dequantization,
            const ngraph::element::Type precisionAfterDequantization,
            const ngraph::element::Type precisionFqOnData,
            const FakeQuantizeOnData& fqOnData) {
        const auto input = std::make_shared<ngraph::opset1::Parameter>(
                add.empty() ? precisionBeforeDequantization : precisionBeforeAdd,
                ngraph::Shape(inputShape));
        input->set_friendly_name("input");

        std::shared_ptr<Node> parent = input;
        if (!add.empty()) {
            parent = makeElementwise<ngraph::opset1::Add>(parent, add);
        }

        const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

        std::shared_ptr<Node> lastNode;

        if (fqOnData.outputLowValues == std::vector<float>{0.f} &&
                fqOnData.outputHighValues == std::vector<float>{2.55f}) {
            auto fqOnDataCopy = fqOnData;
            fqOnDataCopy.outputHighValues = {255.f};
            fqOnDataCopy.outputPrecision = ngraph::element::u8;
            lastNode = makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnDataCopy);
            lastNode = makeDequantization(lastNode, { {element::f32}, {}, {{0.01f}, precisionFqOnData} });

        } else {
            throw std::runtime_error("Unknown parameter on output intervals!");
        }
        lastNode->set_friendly_name("output");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastNode) };
        return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FuseFakeQuantizeFunction");
    }

std::shared_ptr<ngraph::Function> FuseFakeQuantizeFunction::get(
    const ngraph::Shape& inputShape,
    const std::vector<Branch>& branches,
    const ngraph::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    if (branches.size() != 2ul) {
        throw std::runtime_error("unsupported branches count");
    }

    if (branches[0].dequantization.multiply.outPrecision != branches[1].dequantization.multiply.outPrecision) {
        throw std::runtime_error("branch precisions are not equal");
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
    fakeQuantize->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, inputs, "FuseFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
