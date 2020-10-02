// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fuse_convert_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> FuseConvertFunction::get(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const bool constInput) {
    std::shared_ptr<Node> parent;
    std::shared_ptr<op::Parameter> input;
    if (constInput) {
        parent = std::make_shared<opset1::Constant>(inputPrecision, inputShape, std::vector<float>{ 128.f });
    } else {
        input = std::make_shared<ngraph::opset1::Parameter>(
            inputPrecision,
            ngraph::Shape(inputShape));
        parent = input;
    }

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(parent, dequantization);
    dequantizationOp->set_friendly_name("output");

    auto parameters = constInput ?
        ngraph::ParameterVector{}:
        ngraph::ParameterVector{ input };

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, parameters, "FuseConvertFunction");
}

std::shared_ptr<ngraph::Function> FuseConvertFunction::getWithFQ(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type inputPrecision,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const bool constInput) {
    std::shared_ptr<Node> parent;
    std::shared_ptr<op::Parameter> input1;
    if (constInput) {
        parent = std::make_shared<opset1::Constant>(inputPrecision, inputShape, std::vector<float>{ 128.f });
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
                inputPrecision,
                ngraph::Shape(inputShape));
        parent = input1;
    }

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(parent, dequantization);

    std::shared_ptr<op::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(
            inputPrecision,
            ngraph::Shape(inputShape));

    const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
        input2, inputPrecision, 256ul, { 1ul },
        { 0.f }, { 255.f }, { 0.f }, { 255.f });

    // just some non-transparent layer
    const auto power = std::make_shared<opset1::Power>(
        fakeQuantizeOnActivations,
        std::make_shared<opset1::Constant>(element::f32, Shape{}, std::vector<float>{2.f}));

    const auto add = std::make_shared<opset1::Add>(
        dequantizationOp,
        power);

    add->set_friendly_name("output");

    auto parameters = constInput ?
                      ngraph::ParameterVector{ input2 }:
                      ngraph::ParameterVector{ input1, input2 };

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    return std::make_shared<ngraph::Function>(results, parameters, "FuseConvertFunction");
}


}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
