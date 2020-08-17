// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/add_function.hpp"
#include "transformations/low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> AddFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const int constInput,
    const std::vector<float>& constValues) {
    std::shared_ptr<ngraph::Node> input1;
    if (constInput == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }

    const auto dequantizationOp1 = makeDequantization(input1, dequantization1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2, ngraph::Shape(inputShape));
    }

    const auto dequantizationOp2 = makeDequantization(input2, dequantization2);

    const auto add = std::make_shared<ngraph::opset1::Add>(dequantizationOp1, dequantizationOp2);

    add->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(add) };
    ngraph::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        THROW_IE_EXCEPTION << "Unexpected constant input index";
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

std::shared_ptr<ngraph::Function> AddFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const bool broadcast,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ngraph::element::Type& precision1,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization1,
    const ngraph::element::Type& precision2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization2,
    const ngraph::builder::subgraph::DequantizationOperations& dequantizationAfter,
    const int constInput,
    const std::vector<float>& constValues) {
    std::shared_ptr<ngraph::Node> input1;
    if (constInput == 0) {
        input1 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input1 = std::make_shared<ngraph::opset1::Parameter>(
            precision1,
            broadcast ? ngraph::Shape({ inputShape[0], inputShape[1], 1, 1 }) : ngraph::Shape(inputShape));
    }

    const auto dequantizationOp1 = makeDequantization(input1, dequantization1);

    std::shared_ptr<ngraph::Node> input2;
    if (constInput == 1) {
        input2 = std::make_shared<ngraph::opset1::Constant>(
            precision,
            inputShape,
            constValues);
    } else {
        input2 = std::make_shared<ngraph::opset1::Parameter>(
            precision2, ngraph::Shape(inputShape));
    }

    const auto dequantizationOp2 = makeDequantization(input2, dequantization2);

    auto addOriginal = ngraph::opset1::Add(
        ngraph::op::TemporaryReplaceOutputType(dequantizationOp1, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantizationOp2, element::f32).get());

    auto add = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Add>>(
        addOriginal,
        std::vector<element::Type>{ element::f32, element::f32 },
        std::vector<element::Type>{});

    const auto dequantizationOpAfter = makeDequantization(add, dequantizationAfter);

    dequantizationOpAfter->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOpAfter) };
    ngraph::ParameterVector parameters;
    if (constInput == -1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1), as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 0) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input2) };
    } else if (constInput == 1) {
        parameters = { as_type_ptr<ngraph::opset1::Parameter>(input1) };
    } else {
        THROW_IE_EXCEPTION << "Unexpected constant input index";
    }
    return std::make_shared<ngraph::Function>(results, parameters, "AddTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
