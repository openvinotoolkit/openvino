// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/multiply.hpp"

#include <memory>

#include "openvino/opsets/opset1.hpp"
#include <ov_ops/type_relaxed.hpp>
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

using namespace ov::pass::low_precision;

namespace ov {
namespace builder {
namespace subgraph {

namespace multiply_function {
struct BranchNodes {
    std::shared_ptr<Node> input;
    std::shared_ptr<Node> dequantization;
};

BranchNodes makeBranch(const MultiplyBranch& branch) {
    std::shared_ptr<Node> parent = branch.constant.empty() ?
        std::make_shared<ov::opset1::Parameter>(branch.input_precision, branch.inputShape) :
        std::dynamic_pointer_cast<Node>(std::make_shared<ov::opset1::Constant>(
            branch.constant.outPrecision,
            branch.constant.shape,
            branch.constant.values));

    if (!branch.fake_quantize.empty()) {
        if ((parent->get_output_element_type(0) != ov::element::f32) &&
            (parent->get_output_element_type(0) != ov::element::f16)) {
            throw std::runtime_error("unexpected precision before FakeQuantize");
        }
        parent = makeFakeQuantize(parent, parent->get_output_element_type(0), branch.fake_quantize);
    }

    const auto dequantization = makeDequantization(parent, branch.dequantization);

    return {parent, dequantization};
}
} // namespace multiply_function

std::shared_ptr<ov::Model> MultiplyFunction::get(const ov::element::Type model_precision,
                                                 const MultiplyValues& actualValues) {
    const auto branchNodes1 = multiply_function::makeBranch(actualValues.branch1);
    const auto branchNodes2 = multiply_function::makeBranch(actualValues.branch2);

    // branchNodes1.dequantization & branchNodes2.dequantization can have different input types
    std::shared_ptr<ov::Node> parent = std::make_shared<ov::op::TypeRelaxed<ov::opset1::Multiply>>(
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{actualValues.after_dequantization.empty() ? model_precision : ov::element::f32},
        ov::op::TemporaryReplaceOutputType(branchNodes1.dequantization, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(branchNodes2.dequantization, ov::element::f32).get());

    auto& rtInfo = parent->get_rt_info();
    rtInfo["Variant::std::string"] = "multiply";

    parent = makeDequantization(parent, actualValues.after_dequantization);
    parent->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };

    ov::ParameterVector inputs;
    if (ov::is_type<ov::opset1::Parameter>(branchNodes1.input)) {
        inputs.push_back(ov::as_type_ptr<ov::opset1::Parameter>(branchNodes1.input));
    }
    if (ov::is_type<ov::opset1::Parameter>(branchNodes2.input)) {
        inputs.push_back(ov::as_type_ptr<ov::opset1::Parameter>(branchNodes2.input));
    }

    return std::make_shared<ov::Model>(results, inputs, "MultiplyTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
