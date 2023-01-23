// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/push_constant_to_subgraph.hpp"

#include <openvino/core/validation_util.hpp>
#include <openvino/op/util/multi_subgraph_base.hpp>

#include "itt.hpp"

bool ov::pass::PushConstantToSubgraph::run_on_model(const std::shared_ptr<Model>& model) {
    RUN_ON_FUNCTION_SCOPE(PushConstantToSubgraph);

    bool result = false;
    for (const auto& op : model->get_ordered_ops()) {
        const auto multi_sub_graph_op = as_type_ptr<op::util::MultiSubGraphOp>(op);
        if (!multi_sub_graph_op) {
            continue;
        }

        std::unordered_map<size_t, std::shared_ptr<op::v0::Constant>> constants;
        int remove_inputs_mask = 0;

        for (int body_idx = 0; body_idx < static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
             body_idx++) {
            const auto& body = multi_sub_graph_op->get_function(body_idx);
            auto& body_params = body->get_parameters();
            auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
            for (auto desc_it = descriptions.begin(); desc_it < descriptions.end();) {
                const auto& desc = *desc_it;
                const auto input_index = desc->m_input_index;
                if (std::dynamic_pointer_cast<op::util::MultiSubGraphOp::InvariantInputDescription>(desc) == nullptr) {
                    remove_inputs_mask &= ~(1 << input_index);
                    desc_it++;
                    continue;
                }
                std::shared_ptr<op::v0::Constant> constant;
                auto it = constants.find(input_index);
                if (it == constants.end()) {
                    constant = constantfold_subgraph(multi_sub_graph_op->input_value(input_index));
                    if (!constant) {
                        desc_it++;
                        continue;
                    }
                    constants.insert({input_index, constant});
                } else {
                    constant = it->second;
                }
                const auto body_parameter_index = desc->m_body_parameter_index;
                auto& body_param = body_params[body_parameter_index];
                body_param->output(0).replace(constant);
                body->remove_parameter(body_param);
                desc_it = descriptions.erase(desc_it);
                for (auto& desc : descriptions) {
                    if (desc->m_body_parameter_index > body_parameter_index) {
                        desc->m_body_parameter_index--;
                    }
                }
                remove_inputs_mask |= 1 << input_index;
                result = true;
            }
        }

        if (remove_inputs_mask > 0) {
            auto inputs = multi_sub_graph_op->input_values();
            for (size_t i = multi_sub_graph_op->get_input_size(); i > 0; i--) {
                const auto input_index = i - 1;
                if ((remove_inputs_mask & (1 << input_index)) != 0) {
                    inputs.erase(inputs.begin() + input_index);

                    for (int body_idx = 0;
                         body_idx < static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
                         body_idx++) {
                        auto& descriptions = multi_sub_graph_op->get_input_descriptions(body_idx);
                        for (auto& desc : descriptions) {
                            if (desc->m_input_index > input_index) {
                                desc->m_input_index--;
                            }
                        }
                    }
                }
            }
            multi_sub_graph_op->set_arguments(inputs);
        }

        for (int body_idx = 0; body_idx < static_cast<int>(multi_sub_graph_op->get_internal_subgraphs_size());
             body_idx++) {
            bool model_changed = run_on_model(multi_sub_graph_op->get_function(body_idx));
            result = result || model_changed;
        }
    }

    return result;
}
