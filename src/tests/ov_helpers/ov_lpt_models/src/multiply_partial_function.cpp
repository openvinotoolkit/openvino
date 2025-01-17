// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/multiply_partial_function.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "openvino/op/constant.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

namespace multiply_partial_function {
struct BranchNodes {
    std::shared_ptr<Node> input;
    std::shared_ptr<Node> dequantization;
};

BranchNodes getBranch(const MultiplyPartialBranch& branch) {
    const std::shared_ptr<Node> parent = branch.constant.empty() ?
        std::make_shared<ov::opset1::Parameter>(branch.precisionBeforeDequantization, branch.inputShape) :
        std::dynamic_pointer_cast<Node>(std::make_shared<ov::opset1::Constant>(
            branch.constant.outPrecision,
            branch.constant.shape,
            branch.constant.values));

    const auto dequantization = makeDequantization(parent, branch.dequantization);
    return {parent, dequantization};
}
} // namespace multiply_partial_function

std::shared_ptr<ov::Model> MultiplyPartialFunction::get(const ov::element::Type precision,
                                                        const MultiplyPartialValues& actualValues) {
    auto branch1Structure = actualValues.branch1;
    branch1Structure.precisionBeforeDequantization = precision;
    branch1Structure.dequantization.multiply.outPrecision = precision;
    auto branch2Structure = actualValues.branch2;
    branch2Structure.precisionBeforeDequantization = precision;
    branch2Structure.dequantization.multiply.outPrecision = precision;

    const auto branchNodes1 = multiply_partial_function::getBranch(actualValues.branch1);
    const auto branchNodes2 = multiply_partial_function::getBranch(actualValues.branch2);

    auto multiplyOriginal =
        ov::opset1::Multiply(ov::op::TemporaryReplaceOutputType(branchNodes1.dequantization, ov::element::f32).get(),
                             ov::op::TemporaryReplaceOutputType(branchNodes2.dequantization, ov::element::f32).get());

    const std::shared_ptr<ov::Node> multiply = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        multiplyOriginal,
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{precision});
    auto& rtInfo = multiply->get_rt_info();
    rtInfo["Variant::std::string"] = "multiply";
    multiply->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(multiply) };

    ov::ParameterVector inputs;
    if (ov::is_type<ov::opset1::Parameter>(branchNodes1.input)) {
        inputs.push_back(ov::as_type_ptr<ov::opset1::Parameter>(branchNodes1.input));
    }
    if (ov::is_type<ov::opset1::Parameter>(branchNodes2.input)) {
        inputs.push_back(ov::as_type_ptr<ov::opset1::Parameter>(branchNodes2.input));
    }

    return std::make_shared<ov::Model>(results, inputs, "MultiplyTransformation");
}

std::shared_ptr<ov::Model> MultiplyPartialFunction::get(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const bool broadcast1,
    const ov::builder::subgraph::FakeQuantizeOnData& fq1,
    const bool broadcast2,
    const ov::builder::subgraph::FakeQuantizeOnData& fq2,
    const ov::builder::subgraph::FakeQuantizeOnData& fqAfter,
    const bool secondInputIsConstant) {
    auto inputShape1 = inputShape;
    if (broadcast1) {
        inputShape1[2] = 1;
        inputShape1[3] = 1;
    }

    ov::PartialShape inputShape2;
    if (secondInputIsConstant) {
        inputShape2 = {};
    } else {
        inputShape2 = inputShape;
        if (broadcast2) {
            inputShape2[2] = 1;
            inputShape2[3] = 1;
        }
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    const auto fakeQuantize1 = fq1.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input1, precision, fq1.quantizationLevel, fq1.constantShape,
            fq1.inputLowValues, fq1.inputHighValues, fq1.outputLowValues, fq1.outputHighValues);
    if (fakeQuantize1 != nullptr) {
        fakeQuantize1->set_friendly_name("fakeQuantize1");
    }

    const std::shared_ptr<ov::Node> input2 =
        secondInputIsConstant
            ? static_cast<std::shared_ptr<ov::Node>>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.5f}))
            : static_cast<std::shared_ptr<ov::Node>>(std::make_shared<ov::opset1::Parameter>(precision, inputShape2));
    const auto fakeQuantize2 = fq2.empty() ?
        nullptr :
        ov::test::utils::make_fake_quantize(
            input2, precision, fq2.quantizationLevel, fq2.constantShape,
            fq2.inputLowValues, fq2.inputHighValues, fq2.outputLowValues, fq2.outputHighValues);
    if (fakeQuantize2 != nullptr) {
        fakeQuantize2->set_friendly_name("fakeQuantize2");
    }

    const auto multiply = std::make_shared<ov::opset1::Multiply>(
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
    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(result) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        secondInputIsConstant ?
            ov::ParameterVector{ input1 } :
            ov::ParameterVector{ input1, ov::as_type_ptr<ov::opset1::Parameter>(input2) },
        "MultiplyTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
