// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/if.hpp"

#include "core/graph.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/if.hpp"

namespace ov {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector if_op(const Node& node) {
    const auto& ng_inputs = node.get_ng_inputs();
    OPENVINO_ASSERT(ng_inputs.size() == 1, "If operator takes only one input");

    const auto& subgraphs = node.get_subgraphs();
    OPENVINO_ASSERT(subgraphs.count("then_branch") == 1, "Missing 'then_branch' attribute");
    auto then_subgraph = subgraphs.at("then_branch");
    const auto& then_params = then_subgraph->get_ng_parameters();
    auto then_branch =
        std::make_shared<ov::Model>(then_subgraph->get_ng_outputs(), then_params, then_subgraph->get_name());
    OPENVINO_ASSERT(subgraphs.count("else_branch") == 1, "Missing 'else_branch' attribute");
    auto else_subgraph = subgraphs.at("else_branch");
    const auto& else_params = else_subgraph->get_ng_parameters();
    auto else_branch =
        std::make_shared<ov::Model>(else_subgraph->get_ng_outputs(), else_params, else_subgraph->get_name());

    auto if_node = std::make_shared<ov::op::v8::If>(ng_inputs.at(0));
    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);

    const auto then_branch_inputs_from_parent = then_subgraph->get_inputs_from_parent();
    OPENVINO_ASSERT(then_branch_inputs_from_parent.size() == then_params.size(),
                    "Number of inputs to 'then_branch' is invalid. Expected " +
                        std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                        std::to_string(then_params.size()));
    auto then_param = then_params.cbegin();
    for (const auto& from_parent : then_branch_inputs_from_parent) {
        if_node->set_input(from_parent, *then_param, nullptr);
        then_param++;
    }
    const auto else_branch_inputs_from_parent = else_subgraph->get_inputs_from_parent();
    OPENVINO_ASSERT(else_branch_inputs_from_parent.size() == else_params.size(),
                    "Number of inputs to 'else_branch' is invalid. Expected " +
                        std::to_string(else_branch_inputs_from_parent.size()) + ", actual " +
                        std::to_string(else_params.size()));
    auto else_param = else_params.cbegin();
    for (const auto& from_parent : else_branch_inputs_from_parent) {
        if_node->set_input(from_parent, nullptr, *else_param);
        else_param++;
    }
    OPENVINO_ASSERT(then_branch->get_results().size() == else_branch->get_results().size(),
                    "'then' and 'else' branches have to have the same number of outputs");
    auto else_result = else_branch->get_results().cbegin();
    for (const auto& then_result : then_branch->get_results()) {
        if_node->set_output(then_result, *else_result);
        else_result++;
    }
    if_node->validate_and_infer_types();

    return if_node->outputs();
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ov
