// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::vector<std::shared_ptr<ngraph::op::Parameter>> MatMulFunction::getInputs(const std::vector<std::shared_ptr<ngraph::Node>>& nodes) {
    std::vector<std::shared_ptr<ngraph::op::Parameter>> inputs;

    for (std::shared_ptr<ngraph::Node> node : nodes) {
        std::queue<std::shared_ptr<ngraph::Node>> q;
        q.push({ node });
        while (!q.empty()) {
            auto currentNode = q.front();
            q.pop();

            const size_t size = currentNode->inputs().size();
            if (size == 0) {
                std::shared_ptr<ngraph::op::Parameter> input = ngraph::as_type_ptr<ngraph::op::Parameter>(currentNode);
                if (input != nullptr) {
                    input->set_friendly_name("input" + std::to_string(inputs.size() + 1));
                    inputs.push_back(input);
                }
            }

            for (int i = 0; i < size; ++i) {
                auto parent = currentNode->get_input_node_shared_ptr(i);
                q.push(parent);
            }
        }
    }

    return inputs;
}

std::pair<std::shared_ptr<ngraph::opset1::Parameter>, std::shared_ptr<ngraph::opset1::Multiply>> getOriginalBranch(
    const ngraph::element::Type precision,
    const MatMulFunctionBranch& branch,
    const std::string& inputName) {
    std::shared_ptr<ngraph::opset1::Parameter> parameter = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Parameter>>(
        ngraph::element::f32,
        branch.shape);
    parameter->set_friendly_name(inputName);

    std::shared_ptr<ngraph::Node> parent = parameter;

    std::shared_ptr<ngraph::opset1::Convert> convert1 = branch.convert1.empty() ?
        nullptr :
        std::make_shared<ngraph::opset1::Convert>(parent, branch.convert1.precision);
    parent = convert1 == nullptr ? parent : convert1;

    std::shared_ptr<ngraph::opset1::Convert> convert2 = branch.convert2.empty() ?
        nullptr :
        std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert2 == nullptr ? parent : convert2;

    std::shared_ptr<ngraph::opset1::Multiply> multiply = branch.multiplyConst.empty() ?
        nullptr :
        std::make_shared<ngraph::opset1::Multiply>(
            parent,
            // ngraph::builder::makeConstant(precision, std::vector<size_t>{1, 16, 1, 1}, {}, true));
            ngraph::builder::makeConstant(precision, branch.multiplyConst.shape, branch.multiplyConst.values, branch.multiplyConst.values.empty()));
    parent = multiply == nullptr ? parent : multiply;

    return make_pair(parameter, multiply);
}

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::MatMulFunctionBranches& branches) {
    auto branch1 = getOriginalBranch(ngPrecision, branches.first, "input1");
    auto branch2 = getOriginalBranch(ngPrecision, branches.second, "input2");

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(branch1.second, branch2.second, false, false);
    matMul->set_friendly_name("matMul");

    // std::shared_ptr<ngraph::opset1::Result> result;
    // if (nodes.size() > 2) {
    //    const auto add = std::make_shared<ngraph::opset1::Add>(matMul, nodes[2]);
    //    add->set_friendly_name("add");
    //    result = std::make_shared<ngraph::opset1::Result>(add);
    // } else {
    //    result = std::make_shared<ngraph::opset1::Result>(matMul);
    // }

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(matMul);

    // std::vector<std::shared_ptr<ngraph::op::Parameter>> inputs = getInputs(nodes);
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { branch1.first, branch2.first },
        "MatMulTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getReference(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape,
    const ngraph::builder::subgraph::MatMulFunctionBranches& branches) {
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
