// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/multiply_partial_function.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include <ov_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"

#include "lpt_ngraph_functions/common/builders.hpp"
#include "lpt_ngraph_functions/common/dequantization_operations.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {

namespace multiply_partial_function {
struct BranchNodes {
    std::shared_ptr<Node> input;
    std::shared_ptr<Node> dequantization;
};

BranchNodes getBranch(const MultiplyPartialBranch& branch) {
    const std::shared_ptr<Node> parent = branch.constant.empty() ?
        std::make_shared<ngraph::opset1::Parameter>(branch.precisionBeforeDequantization, branch.inputShape) :
        std::dynamic_pointer_cast<Node>(std::make_shared<ngraph::opset1::Constant>(
            branch.constant.outPrecision,
            branch.constant.shape,
            branch.constant.values));

    const auto dequantization = makeDequantization(parent, branch.dequantization);
    return {parent, dequantization};
}
} // namespace multiply_partial_function

std::shared_ptr<ngraph::Function> MultiplyPartialFunction::get(
    const element::Type precision,
    const MultiplyPartialValues& actualValues) {
    auto branch1Structure = actualValues.branch1;
    branch1Structure.precisionBeforeDequantization = precision;
    branch1Structure.dequantization.multiply.outPrecision = precision;
    auto branch2Structure = actualValues.branch2;
    branch2Structure.precisionBeforeDequantization = precision;
    branch2Structure.dequantization.multiply.outPrecision = precision;

    const auto branchNodes1 = multiply_partial_function::getBranch(actualValues.branch1);
    const auto branchNodes2 = multiply_partial_function::getBranch(actualValues.branch2);

    auto multiplyOriginal = opset1::Multiply(
        ov::op::TemporaryReplaceOutputType(branchNodes1.dequantization, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(branchNodes2.dequantization, element::f32).get());

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ov::op::TypeRelaxed<ngraph::opset1::Multiply>>(
        multiplyOriginal,
        std::vector<element::Type>{element::f32, element::f32},
        std::vector<element::Type>{precision});
    auto& rtInfo = multiply->get_rt_info();
    rtInfo["Variant::std::string"] = "multiply";
    multiply->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };

    ngraph::ParameterVector inputs;
    if (is_type<opset1::Parameter>(branchNodes1.input)) {
        inputs.push_back(std::dynamic_pointer_cast<opset1::Parameter>(branchNodes1.input));
    }
    if (is_type<opset1::Parameter>(branchNodes2.input)) {
        inputs.push_back(std::dynamic_pointer_cast<opset1::Parameter>(branchNodes2.input));
    }

    return std::make_shared<ngraph::Function>(results, inputs, "MultiplyTransformation");
}

std::shared_ptr<ngraph::Function> MultiplyPartialFunction::get(
    const ngraph::element::Type precision,
    const ngraph::PartialShape& inputShape,
    const bool broadcast1,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fq1,
    const bool broadcast2,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fq2,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqAfter,
    const bool secondInputIsConstant) {
    auto inputShape1 = inputShape;
    if (broadcast1) {
        inputShape1[2] = 1;
        inputShape1[3] = 1;
    }

    ngraph::PartialShape inputShape2;
    if (secondInputIsConstant) {
        inputShape2 = {};
    } else {
        inputShape2 = inputShape;
        if (broadcast2) {
            inputShape2[2] = 1;
            inputShape2[3] = 1;
        }
    }

    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape1);
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);
    if (fakeQuantize1 != nullptr) {
        fakeQuantize1->set_friendly_name("fakeQuantize1");
    }

    const std::shared_ptr<ngraph::Node> input2 = secondInputIsConstant ?
        makeConstant(element::f32, Shape{}, std::vector<float>{0.5f}, false) :
        std::make_shared<ngraph::opset1::Parameter>(precision, inputShape2);
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ngraph::builder::makeFakeQuantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);
    if (fakeQuantize2 != nullptr) {
        fakeQuantize2->set_friendly_name("fakeQuantize2");
    }

    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(
        fq1.empty() ? input1 : fakeQuantize1,
        fq2.empty() ? input2 : fakeQuantize2);
    multiply->set_friendly_name("multiply");

    auto const fakeQuantizeAfter = fqAfter.empty() ?
        nullptr :
        makeFakeQuantize(multiply, precision, fqAfter);
    if (fakeQuantizeAfter != nullptr) {
        fakeQuantizeAfter->set_friendly_name("fakeQuantizeAfter");
    }

    const std::shared_ptr<Node> result = fakeQuantizeAfter == nullptr ? std::dynamic_pointer_cast<Node>(multiply) : fakeQuantizeAfter;
    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(result) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        secondInputIsConstant ?
            ngraph::ParameterVector{ input1 } :
            ngraph::ParameterVector{ input1, ngraph::as_type_ptr<ngraph::opset1::Parameter>(input2) },
        "MultiplyTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
