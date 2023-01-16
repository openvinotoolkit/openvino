// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "input_model.hpp"
#include "common_op_table.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_if_op(const NodeContext& node) {
    default_op_checks(node, 1, {"If", "StatelessIf"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    FRONT_END_GENERAL_CHECK(translate_session, "[TensorFlow Frontend] Internal error: Translate session is nullptr.");

    // retrieve body ov::Model for then and else branches
    auto then_branch_type = node.get_attribute<std::string>("then_branch");
    auto else_branch_type = node.get_attribute<std::string>("else_branch");
    auto then_branch_body = translate_session->get_body_ov_model(then_branch_type);
    FRONT_END_GENERAL_CHECK(
        then_branch_body,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find branch body graph with name " +
            then_branch_type);
    auto else_branch_body = translate_session->get_body_ov_model(else_branch_type);
    FRONT_END_GENERAL_CHECK(
        else_branch_body,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find branch body graph with name " +
            else_branch_type);

    // get condition output
    auto cond = node.get_input(0);
    size_t input_size_t = node.get_input_size() - 1;
    auto then_params = then_branch_body->get_parameters();
    auto else_params = else_branch_body->get_parameters();
    FRONT_END_GENERAL_CHECK(input_size_t == then_params.size(),
                            "[TensorFlow Frontend] Internal error or incorrect input model: number of inputs to If and "
                            "number of inputs in then branch do not match.");
    FRONT_END_GENERAL_CHECK(input_size_t == else_params.size(),
                            "[TensorFlow Frontend] Internal error or incorrect input model: number of inputs to If and "
                            "number of inputs in else branch do not match.");

    // create If operation and set body graphs
    auto if_op = std::make_shared<If>(cond);
    if_op->set_then_body(then_branch_body);
    if_op->set_else_body(else_branch_body);

    // set inputs
    int input_size = static_cast<int>(input_size_t);
    for (int ind = 0; ind < input_size; ++ind) {
        auto curr_input = node.get_input(ind + 1);
        auto then_param = then_params[ind];
        auto else_param = else_params[ind];
        if_op->set_input(curr_input, then_param, else_param);
    }

    // set outputs
    auto then_results = then_branch_body->get_results();
    auto else_results = else_branch_body->get_results();
    FRONT_END_GENERAL_CHECK(then_results.size() == else_results.size(),
                            "[TensorFlow Frontend] Internal error or incorrect input model: number of result nodes in "
                            "then and else branches do not match.");
    int output_size = static_cast<int>(then_results.size());
    for (int ind = 0; ind < output_size; ++ind) {
        if_op->set_output(then_results[ind], else_results[ind]);
    }

    auto ov_outputs = if_op->outputs();

    // set output tensor names
    for (size_t idx = 0; idx < ov_outputs.size(); ++idx) {
        ov_outputs[idx].get_tensor().set_names({node_name + ":" + std::to_string(idx)});
    }

    return ov_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
