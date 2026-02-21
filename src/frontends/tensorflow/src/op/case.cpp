// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/if.hpp"

#include "common_op_table.hpp"
#include "input_model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/result.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_case_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Case"});
    auto node_name = node.get_name();
    auto translate_session = node.get_translate_session();
    FRONT_END_GENERAL_CHECK(translate_session, "[TensorFlow Frontend] Internal error: Translate session is nullptr.");

    auto branch_index = node.get_input(0);
    size_t num_data_inputs = node.get_input_size() - 1;
    ov::OutputVector ov_inputs;
    for (size_t input_ind = 0; input_ind < num_data_inputs; ++input_ind) {
        ov_inputs.push_back(node.get_input(input_ind + 1));
    }

    auto branch_functions = node.get_attribute<std::vector<std::string>>("branches");
    size_t num_branches = branch_functions.size();
    FRONT_END_GENERAL_CHECK(num_branches >= 2,
                            "[TensorFlow Frontend] Case operation must have at least two branches.");

    std::vector<std::shared_ptr<ov::Model>> branch_bodies;
    for (size_t i = 0; i < num_branches; ++i) {
        auto branch_body = translate_session->get_body_ov_model(branch_functions[i], ov_inputs);
        FRONT_END_GENERAL_CHECK(
            branch_body,
            "[TensorFlow Frontend] Internal error or incorrect input model. Cannot find branch body graph with name " +
                branch_functions[i]);
        branch_bodies.push_back(branch_body);
    }

    size_t num_outputs = branch_bodies[0]->get_results().size();
    for (size_t i = 1; i < num_branches; ++i) {
        FRONT_END_GENERAL_CHECK(
            branch_bodies[i]->get_results().size() == num_outputs,
            "[TensorFlow Frontend] All branches in Case operation must have the same number of outputs.");
    }

    auto zero_const = v0::Constant::create(element::i32, Shape{}, {0});
    auto cond = std::make_shared<v1::Equal>(branch_index, zero_const);
    auto if_op = std::make_shared<v8::If>(cond);
    if_op->set_then_body(branch_bodies[0]);

    std::shared_ptr<ov::Model> current_else_model = branch_bodies[1]->clone();

    for (int i = 1; i < static_cast<int>(num_branches) - 1; ++i) {
        ov::ParameterVector wrapper_params;
        auto branch_ind_param = std::make_shared<v0::Parameter>(element::i32, Shape{});
        wrapper_params.push_back(branch_ind_param);

        for (size_t ind = 0; ind < num_data_inputs; ++ind) {
            auto param = std::make_shared<v0::Parameter>(
                ov_inputs[ind].get_element_type(),
                ov_inputs[ind].get_partial_shape());
            wrapper_params.push_back(param);
        }

        auto ind_const = v0::Constant::create(element::i32, Shape{}, {i});
        auto inner_cond = std::make_shared<v1::Equal>(branch_ind_param, ind_const);

        auto inner_if = std::make_shared<v8::If>(inner_cond);
        inner_if->set_then_body(branch_bodies[i]->clone());
        inner_if->set_else_body(current_else_model);

        auto then_body = inner_if->get_then_body();
        auto else_body = inner_if->get_else_body();
        auto then_params = then_body->get_parameters();
        auto else_params = else_body->get_parameters();

        bool else_has_branch_ind = (else_params.size() == num_data_inputs + 1);
        for (size_t ind = 0; ind < num_data_inputs; ++ind) {
            size_t else_ind = else_has_branch_ind ? ind + 1 : ind;
            inner_if->set_input(wrapper_params[ind + 1], then_params[ind], else_params[else_ind]);
        }

        if (else_has_branch_ind) {
            auto dummy_param = std::make_shared<v0::Parameter>(element::i32, Shape{});
            then_body->add_parameters({dummy_param});
            inner_if->set_input(branch_ind_param, dummy_param, else_params[0]);
        }

        auto then_results = then_body->get_results();
        auto else_results = else_body->get_results();
        for (size_t ind = 0; ind < num_outputs; ++ind) {
            inner_if->set_output(then_results[ind], else_results[ind]);
        }

        ov::ResultVector wrapper_results;
        for (size_t ind = 0; ind < num_outputs; ++ind) {
            wrapper_results.push_back(std::make_shared<v0::Result>(inner_if->output(ind)));
        }

        current_else_model = std::make_shared<ov::Model>(wrapper_results, wrapper_params);
    }

    if_op->set_else_body(current_else_model);

    auto then_params = branch_bodies[0]->get_parameters();
    auto else_params = current_else_model->get_parameters();
    bool else_has_branch_ind = (else_params.size() == num_data_inputs + 1);

    for (size_t ind = 0; ind < num_data_inputs; ++ind) {
        size_t else_ind = else_has_branch_ind ? ind + 1 : ind;
        if_op->set_input(ov_inputs[ind], then_params[ind], else_params[else_ind]);
    }

    if (else_has_branch_ind) {
        auto dummy_param = std::make_shared<v0::Parameter>(element::i32, Shape{});
        if_op->get_then_body()->add_parameters({dummy_param});
        if_op->set_input(branch_index, dummy_param, else_params[0]);
    }

    auto then_results = if_op->get_then_body()->get_results();
    auto else_results = current_else_model->get_results();
    for (size_t ind = 0; ind < num_outputs; ++ind) {
        if_op->set_output(then_results[ind], else_results[ind]);
    }

    auto ov_outputs = if_op->outputs();
    for (size_t ind = 0; ind < ov_outputs.size(); ++ind) {
        ov_outputs[ind].get_tensor().set_names({node_name + ":" + std::to_string(ind)});
    }

    return ov_outputs;
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
