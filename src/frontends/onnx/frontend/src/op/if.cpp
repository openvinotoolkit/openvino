// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include "core/graph.hpp"
#include "core/operator_set.hpp"
#include "openvino/core/model.hpp"
#include "openvino/frontend/exception.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector if_op(const ov::frontend::onnx::Node& node) {
    const auto& ng_inputs = node.get_ov_inputs();
    FRONT_END_GENERAL_CHECK(ng_inputs.size() == 1, "If operator takes only one input");

    const auto& subgraphs = node.get_subgraphs();
    FRONT_END_GENERAL_CHECK(subgraphs.count("then_branch") == 1, "Missing 'then_branch' attribute");
    auto then_subgraph = subgraphs.at("then_branch");
    const auto& then_params = then_subgraph->get_ng_parameters();
    auto then_branch =
        std::make_shared<ov::Model>(then_subgraph->get_ov_outputs(), then_params, then_subgraph->get_name());
    FRONT_END_GENERAL_CHECK(subgraphs.count("else_branch") == 1, "Missing 'else_branch' attribute");
    auto else_subgraph = subgraphs.at("else_branch");
    const auto& else_params = else_subgraph->get_ng_parameters();
    auto else_branch =
        std::make_shared<ov::Model>(else_subgraph->get_ov_outputs(), else_params, else_subgraph->get_name());

    auto if_node = std::make_shared<v8::If>(ng_inputs.at(0));
    if_node->set_then_body(then_branch);
    if_node->set_else_body(else_branch);

    const auto then_branch_inputs_from_parent = then_subgraph->get_inputs_from_parent();
    FRONT_END_GENERAL_CHECK(then_branch_inputs_from_parent.size() == then_params.size(),
                            "Number of inputs to 'then_branch' is invalid. Expected " +
                                std::to_string(then_branch_inputs_from_parent.size()) + ", actual " +
                                std::to_string(then_params.size()));
    auto then_param = then_params.cbegin();
    for (const auto& from_parent : then_branch_inputs_from_parent) {
        if_node->set_input(from_parent, *then_param, nullptr);
        then_param++;
    }
    const auto else_branch_inputs_from_parent = else_subgraph->get_inputs_from_parent();
    FRONT_END_GENERAL_CHECK(else_branch_inputs_from_parent.size() == else_params.size(),
                            "Number of inputs to 'else_branch' is invalid. Expected " +
                                std::to_string(else_branch_inputs_from_parent.size()) + ", actual " +
                                std::to_string(else_params.size()));
    auto else_param = else_params.cbegin();
    for (const auto& from_parent : else_branch_inputs_from_parent) {
        if_node->set_input(from_parent, nullptr, *else_param);
        else_param++;
    }
    FRONT_END_GENERAL_CHECK(then_branch->get_results().size() == else_branch->get_results().size(),
                            "'then' and 'else' branches have to have the same number of outputs");
    auto else_result = else_branch->get_results().cbegin();
    for (const auto& then_result : then_branch->get_results()) {
        if_node->set_output(then_result, *else_result);
        else_result++;
    }
    if_node->validate_and_infer_types();

    return if_node->outputs();
}
ONNX_OP("If", OPSET_SINCE(1), ai_onnx::opset_1::if_op);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
