// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/move_fake_quantize_function.hpp"

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
    const FakeQuantizeOnDataWithConstant& fqOnData3,
    const DequantizationOperations::Convert& convert3,
    const DequantizationOperations& dequantization3,
    const ngraph::element::Type precisionAfterOperation,
    const DequantizationOperations& dequantizationAfter,
    const std::int64_t& axis) {

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input1->set_friendly_name("input1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(inputPrecision, inputShape);
    input2->set_friendly_name("input2");

    if (fqOnData3.empty()) {
        std::shared_ptr<Node> parent1 = makeFakeQuantizeTypeRelaxed(input1, inputPrecision, fqOnData1);
        if (!convert1.empty()) {
            parent1 = std::make_shared<opset1::Convert>(parent1, convert1.outPrecision);
        }
        if (!dequantization1.empty()) {
            parent1 = makeDequantization(parent1, dequantization1);
        }

        std::shared_ptr<Node> parent2 = makeFakeQuantizeTypeRelaxed(input2, inputPrecision, fqOnData2);
        if (!convert2.empty()) {
            parent2 = std::make_shared<opset1::Convert>(parent2, convert2.outPrecision);
        }
        if (!dequantization2.empty()) {
            parent2 = makeDequantization(parent2, dequantization2);
        }

        const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ parent1, parent2 }, axis);

        auto& rtInfo = concat->get_rt_info();
        rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

        const auto lastDequantization = makeDequantization(concat, dequantizationAfter);
        lastDequantization->set_friendly_name("output");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
        std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input1, input2 },
            "MoveFakeQuantize");
        return function;
    }
    else {
        const std::shared_ptr<ngraph::opset1::Concat> concat = std::make_shared<ngraph::opset1::Concat>(ngraph::OutputVector{ input1, input2 }, axis);

        auto& rtInfo = concat->get_rt_info();
        rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("concat");

        std::shared_ptr<Node> fq = makeFakeQuantizeTypeRelaxed(concat, inputPrecision, fqOnData3);

        const auto lastDequantization = makeDequantization(fq, dequantizationAfter);
        lastDequantization->set_friendly_name("output");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(lastDequantization) };
        std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input1, input2 },
            "MoveFakeQuantize");
        return function;
    }
}

std::shared_ptr<Node> MoveFakeQuantize::makeMaxPool(const Output<Node>& parent, const std::vector<size_t>& kernel) {
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
    const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;
    const auto pooling = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);
    return pooling;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
