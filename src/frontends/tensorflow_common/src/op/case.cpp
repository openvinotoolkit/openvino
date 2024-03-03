// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_case_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Case"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    FRONT_END_GENERAL_CHECK(translate_session, "[TensorFlow Frontend] Internal error: Translate session is nullptr.");

    // skip the first input because it does not go to the body graphs
    size_t input_size_t = node.get_input_size() - 1;
    int input_size = static_cast<int>(input_size_t);
    ov::OutputVector ov_inputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        ov_inputs.push_back(node.get_input(input_ind + 1));
    }

    // retrieve body ov::Model for then and else branches
    auto branches = node.get_attribute<std::string>("branches");
    // get desired branch index  
    auto cond = node.get_input(0)
    auto desired_branch_type = branches[cond]
    auto placeholder_branch_type = branches[cond]

    auto desired_branch_body= translate_session->get_body_ov_model(desired_branch_type, ov_inputs);
            
    auto placeholder_branch_body = translate_session->get_body_ov_model(placeholder_branch_type, ov_inputs);

    // get condition output
    auto desired_params = then_branch_odyb->get_parameters();
    auto placeholder_params = placeholder_branch_body->get_parameters();

    // create If operation and set body graphs
    auto if_op = std::make_shared<If>(cond);
    if_op->set_then_body(desired_branch_body);
    if_op->set_else_body(placeholder_branch_body);

    // set inputs
    for (int ind = 0; ind < input_size; ++ind) {
        auto curr_input = node.get_input(ind + 1);
        auto desired_param = desired_params[ind];
        auto placeholder_param = else_params[ind];
        if_op->set_input(curr_input, desired_param, placeholder_param);
    }

    // set outputs
    auto desired_results = desired_branch_body->get_results();
    auto placeholder_results = placeholder_branch_body->get_results();
    int output_size = static_cast<int>(desired_results.size());
    for (int ind = 0; ind < output_size; ++ind) {
        if_op->set_output(desired_results[ind], placeholder_results[ind]);
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
