// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
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

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape,
    const std::vector<std::shared_ptr<ngraph::Node>>& nodes) {
    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(
        nodes[0],
        nodes[1],
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result;
    if (nodes.size() > 2) {
        const auto add = std::make_shared<ngraph::opset1::Add>(matMul, nodes[2]);
        add->set_friendly_name("add");
        result = std::make_shared<ngraph::opset1::Result>(add);
    } else {
        result = std::make_shared<ngraph::opset1::Result>(matMul);
    }

    std::vector<std::shared_ptr<ngraph::op::Parameter>> inputs = getInputs(nodes);
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, inputs, "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getReference(
    const ngraph::element::Type ngPrecision,
    const ngraph::Shape& inputShape,
    const std::vector<std::shared_ptr<ngraph::Node>>& nodes) {
    return nullptr;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
