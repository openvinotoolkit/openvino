// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"
#include <low_precision/relu.hpp>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/fake_quantize_on_data.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> MoveFakeQuantize::get(
    const ngraph::element::Type inputPrecision,
    const ngraph::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData1,
    const DequantizationOperations::Convert& convert1,
    const DequantizationOperations& dequantization1,
    const FakeQuantizeOnDataWithConstant& fqOnData2,
    const DequantizationOperations::Convert& convert2,
    const DequantizationOperations& dequantization2,
    const std::string& operation,
    const FakeQuantizeOnDataWithConstant& fqOnData3,
    const DequantizationOperations::Convert& convert3,
    const DequantizationOperations& dequantization3,
    const std::vector<ov::Any>& concatAttributes,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const std::int64_t& axis) {

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input1->set_friendly_name("input1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input2->set_friendly_name("input2");
    std::shared_ptr<Node> parent1 = input1, parent2 = input2;
    if (!fqOnData1.empty()) {
        if (operation == "relu") {
            auto relu1 = std::make_shared<ngraph::opset1::Relu>(input1->output(0));
            parent1 = makeFakeQuantize(relu1, inputPrecision, fqOnData1);
        } else {
            parent1 = makeFakeQuantize(input1, inputPrecision, fqOnData1);
        }
        parent1->set_friendly_name("concat_fq1");
        if (!convert1.empty()) {
            parent1 = std::make_shared<opset1::Convert>(parent1, convert1.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent1 = makeDequantization(parent1, dequantization1);
        }
    }
    if (!fqOnData2.empty()) {
        if (operation == "relu") {
            auto relu2 = std::make_shared<ngraph::opset1::Relu>(input2->output(0));
            parent2 = makeFakeQuantize(relu2, inputPrecision, fqOnData2);
        } else {
            parent2 = makeFakeQuantize(input1, inputPrecision, fqOnData2);
        }
        parent2->set_friendly_name("concat_fq2");
        if (!convert2.empty()) {
            parent1 = std::make_shared<opset1::Convert>(parent2, convert2.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent2 = makeDequantization(parent2, dequantization2);
        }
    }
    const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ parent1, parent2 }, axis);
    concat->set_friendly_name("concat");
    std::shared_ptr<ngraph::Node> parent = concat;
    if (!dequantizationAfter.empty()) {
        const auto lastDequantization = makeDequantization(concat, dequantizationAfter);
        lastDequantization->set_friendly_name("multiply");
        parent = lastDequantization;
    }
    addAttributes({ parent }, concatAttributes);
    if (!fqOnData3.empty()) {
        std::shared_ptr<Node> fq;
        if (operation == "relu") {
            auto relu = std::make_shared<ngraph::opset1::Relu>(concat->output(0));
            fq = makeFakeQuantize(relu, inputPrecision, fqOnData3);
        } else {
            fq = makeFakeQuantize(concat, inputPrecision, fqOnData3);
        }
        fq->set_friendly_name("fakeQuantizeAfter");
        parent = fq;
    }
    parent->set_friendly_name("output");
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ input1, input2 },
        "MoveFakeQuantize");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
