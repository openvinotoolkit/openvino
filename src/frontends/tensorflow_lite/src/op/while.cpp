// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "op_translation_utils.hpp"
#include "openvino/opsets/opset11.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset11;

namespace ov {
namespace frontend {
namespace tensorflow_lite {
namespace op {

OutputVector while_op(const ov::frontend::tensorflow_lite::NodeContext& node) {
    int32_t cond_idx = node.get_attribute<int32_t>("cond_subgraph_index");
    int32_t body_idx = node.get_attribute<int32_t>("body_subgraph_index");

    auto condition = node.get_subgraph(cond_idx);
    auto body = node.get_subgraph(body_idx);

    FRONT_END_GENERAL_CHECK(condition, "Incorrect model: condition graph was not read properly");
    FRONT_END_GENERAL_CHECK(body, "Incorrect model: body graph was not read properly");

    // insert condition before loop
    ov::OutputVector ov_inputs = node.get_inputs();
    auto condition_prior_loop = condition->clone();
    auto condition_parameters = condition_prior_loop->get_parameters();
    for (size_t param_ind = 0; param_ind < condition_parameters.size(); ++param_ind)
        condition_parameters[param_ind]->output(0).replace(ov_inputs[param_ind]);
    ov::OutputVector ov_outputs;
    for (const auto& result_node : condition_prior_loop->get_results())
        ov_outputs.push_back(result_node->input_value(0));

    TENSORFLOW_OP_VALIDATION(node,
                             ov_outputs.size() == 1,
                             "[TensorFlow Lite Frontend] Internal error or inconsistent model: condition body must "
                             "contain one Result node.");

    auto exec_cond = ov_outputs[0];
    auto trip_count = make_shared<Constant>(element::i32, Shape{}, -1);
    auto loop = make_shared<Loop>(trip_count, exec_cond);

    // prepare body model to be set for the Loop node
    // note that condition should be computed on the updated input
    // because this is while(cond) {} construction,
    // that is why condition graph is stitched to the body results
    auto body_params = body->get_parameters();
    auto body_results = body->get_results();
    auto cond_results = condition->get_results();
    condition_parameters = condition->get_parameters();
    auto cond_params_size = condition_parameters.size();
    TENSORFLOW_OP_VALIDATION(node,
                             body_params.size() == node.get_input_size(),
                             "[TensorFlow Lite Frontend] Internal error or inconsistent model: body graph "
                             " must have the same number of Parameter nodes as a number of inputs to While.");
    TENSORFLOW_OP_VALIDATION(node,
                             body_results.size() == node.get_input_size(),
                             "[TensorFlow Lite Frontend] Internal error or inconsistent model: body graphs "
                             " must have the same number of Result nodes as a number of inputs to While.");
    TENSORFLOW_OP_VALIDATION(node,
                             condition_parameters.size() == node.get_input_size(),
                             "[TensorFlow Lite Frontend] Internal error or inconsistent model: condition graph "
                             " must have the same number of Parameter nodes as a number of inputs to While.");
    for (size_t param_ind = 0; param_ind < cond_params_size; ++param_ind) {
        condition_parameters[param_ind]->output(0).replace(body_results[param_ind]->input_value(0));
    }

    // update body model with the new result that corresponds to execution condition
    TENSORFLOW_OP_VALIDATION(node,
                             cond_results.size() == 1 && cond_results[0],
                             "[TensorFlow Lite Frontend] Internal error or inconsistent model: condition body must "
                             "contain one Result node.");
    auto body_condition_output_idx = static_cast<int64_t>(body_results.size());
    body->add_results(cond_results);

    // set data for the Loop node
    loop->set_function(body);

    for (int input_ind = 0; input_ind < static_cast<int>(node.get_input_size()); ++input_ind) {
        loop->set_merged_input(body_params[input_ind],
                               node.get_input(input_ind),
                               body_results[input_ind]->input_value(0));
    }
    loop->set_special_body_ports({-1, body_condition_output_idx});

    // set external outputs for Loop node
    // do not get execution condition outside the Loop node
    for (size_t output_ind = 0; output_ind < node.get_input_size(); ++output_ind) {
        loop->get_iter_value(body_results[output_ind]);
    }
    loop->validate_and_infer_types();
    loop->set_friendly_name(node.get_name());
    return loop->outputs();
}

}  // namespace op
}  // namespace tensorflow_lite
}  // namespace frontend
}  // namespace ov
