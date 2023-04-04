// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_while_op(const NodeContext& node) {
    default_op_checks(node, 1, {"While", "StatelessWhile"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    auto input_size_t = node.get_input_size();
    auto input_size = static_cast<int>(input_size_t);

    TENSORFLOW_OP_VALIDATION(node,
                             translate_session,
                             "[TensorFlow Frontend] Internal error: Translate session is nullptr.");
    // retrieve condition and body graphs
    auto cond_type = node.get_attribute<std::string>("cond");
    auto body_type = node.get_attribute<std::string>("body");
    auto cond_model = translate_session->get_body_ov_model(cond_type);
    TENSORFLOW_OP_VALIDATION(
        node,
        cond_model,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find body graph with name " + cond_type);
    auto body_model = translate_session->get_body_ov_model(body_type);
    TENSORFLOW_OP_VALIDATION(
        node,
        body_model,
        "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find body graph with name " + body_type);

    // inject condition body graph prior to Loop node
    // to check condition before to start iterations
    ov::OutputVector ov_inputs;
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        ov_inputs.push_back(node.get_input(input_ind));
    }

    auto cond_params = cond_model->get_parameters();
    // type setting for body graph parameters is needed for TensorList support since DT_VARIANT type is present
    // also for more accurate execution_condition variable shape deducing we need shape inference for condition graph
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        cond_params[input_ind]->set_element_type(node.get_input(input_ind).get_element_type());
        cond_params[input_ind]->set_partial_shape(node.get_input(input_ind).get_partial_shape());
    }
    cond_model->validate_nodes_and_infer_types();

    auto cond_prior = cond_model->clone();
    ov::OutputVector ov_outputs;
    translate_session->inject_body_model(cond_prior, node.get_name() + "/cond", ov_inputs, ov_outputs);
    TENSORFLOW_OP_VALIDATION(
        node,
        ov_outputs.size() == 1,
        "[TensorFlow Frontend] Internal error or inconsistent model: condition body must contain one Result node.");
    auto exec_cond = ov_outputs[0];
    auto trip_count = make_shared<Constant>(element::i32, Shape{}, -1);
    auto loop = make_shared<Loop>(trip_count, exec_cond);

    // prepare body model to be set for the Loop node
    // note that condition should be computed on the updated input
    // because this is while(cond) {} construction,
    // that is why condition graph is stitched to the body results
    auto body_params = body_model->get_parameters();
    auto body_results = body_model->get_results();
    auto cond_results = cond_model->get_results();
    auto cond_params_size = cond_params.size();
    TENSORFLOW_OP_VALIDATION(node,
                             body_params.size() == input_size_t,
                             "[TensorFlow Frontend] Internal error or inconsistent model: body graph "
                             " must have the same number of Parameter nodes as a number of inputs to While.");
    TENSORFLOW_OP_VALIDATION(node,
                             body_results.size() == input_size_t,
                             "[TensorFlow Frontend] Internal error or inconsistent model: body graphs "
                             " must have the same number of Result nodes as a number of inputs to While.");
    TENSORFLOW_OP_VALIDATION(node,
                             cond_params.size() == input_size_t,
                             "[TensorFlow Frontend] Internal error or inconsistent model: condition graph "
                             " must have the same number of Parameter nodes as a number of inputs to While.");
    for (size_t param_ind = 0; param_ind < cond_params_size; ++param_ind) {
        cond_params[param_ind]->output(0).replace(body_results[param_ind]->input_value(0));
    }

    // update body model with the new result that corresponds to execution condition
    TENSORFLOW_OP_VALIDATION(
        node,
        cond_results.size() == 1 && cond_results[0],
        "[TensorFlow Frontend] Internal error or inconsistent model: condition body must contain one Result node.");
    auto body_condition_output_idx = static_cast<int64_t>(body_results.size());
    body_model->add_results(cond_results);

    // type setting for body graph parameters is needed for TensorList support since DT_VARIANT type is present
    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        body_params[input_ind]->set_element_type(node.get_input(input_ind).get_element_type());
    }

    // set data for the Loop node
    loop->set_function(body_model);

    for (int input_ind = 0; input_ind < input_size; ++input_ind) {
        loop->set_merged_input(body_params[input_ind],
                               node.get_input(input_ind),
                               body_results[input_ind]->input_value(0));
    }
    loop->set_special_body_ports({-1, body_condition_output_idx});

    // set external outputs for Loop node
    // do not get execution condition outside of the Loop node
    for (size_t output_ind = 0; output_ind < input_size_t; ++output_ind) {
        loop->get_iter_value(body_results[output_ind]);
    }
    loop->validate_and_infer_types();

    set_node_name(node.get_name(), loop);
    return loop->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
