// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_multi_subgraph_op_dangling_params.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/loop.hpp"
#include "openvino/op/tensor_iterator.hpp"
#include "openvino/op/util/multi_subgraph_base.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op::util;

bool ov::pass::RemoveMultiSubGraphOpDanglingParamsResults::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(RemoveMultiSubGraphOpDanglingParamsResults);
    bool is_changed = false;
    auto ops = m->get_ordered_ops();
    // Going in reverse order
    for (auto it = ops.rbegin(); it != ops.rend(); ++it) {
        auto multi_subgraph_op = std::dynamic_pointer_cast<MultiSubGraphOp>(*it);
        if (!multi_subgraph_op)
            continue;
        auto if_op = ov::as_type_ptr<ov::op::v8::If>(multi_subgraph_op);
        auto loop_op = ov::as_type_ptr<ov::op::v5::Loop>(multi_subgraph_op);
        auto ti_op = ov::as_type_ptr<ov::op::v0::TensorIterator>(multi_subgraph_op);
        // Only If, Loop and TensorIterator are supported
        if (!if_op && !loop_op && !ti_op)
            continue;

        // Shouldn't remove special output
        int64_t special_out_port = -1;
        if (loop_op) {
            special_out_port = loop_op->get_special_body_ports().body_condition_output_idx;
        }

        const auto subgraphs_size = multi_subgraph_op->get_internal_subgraphs_size();
        // Starting from outputs
        std::set<size_t> outputs_to_remove;
        for (size_t out_idx = 0; out_idx < multi_subgraph_op->get_output_size(); ++out_idx) {
            if (multi_subgraph_op->output(out_idx).get_target_inputs().empty()) {
                outputs_to_remove.insert(out_idx);
            }
        }
        std::vector<MultiSubGraphOp::MultiSubgraphOutputDescriptionVector> new_op_out_desc;
        for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
            auto body = multi_subgraph_op->get_function(static_cast<int>(body_idx));
            // recursive call of this transformation on each body
            run_on_model(body);
            // need to pay attention to merged inputs, shouldn't remove them
            MultiSubGraphOp::MultiSubgraphInputDescriptionVector merged_input_descs;
            for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
                for (const auto& desc : multi_subgraph_op->get_input_descriptions(static_cast<int>(body_idx))) {
                    if (const auto& merged_input_desc =
                            ov::as_type_ptr<MultiSubGraphOp::MergedInputDescription>(desc)) {
                        merged_input_descs.push_back(desc);
                    }
                }
            }
            const auto& out_desc = multi_subgraph_op->get_output_descriptions(static_cast<int>(body_idx));
            MultiSubGraphOp::MultiSubgraphOutputDescriptionVector new_out_desc;
            std::set<size_t> results_idxs_to_remove;
            for (const auto& odesc : out_desc) {
                bool to_remove = outputs_to_remove.find(odesc->m_output_index) != outputs_to_remove.end();
                if (!to_remove) {
                    new_out_desc.push_back(odesc);
                } else if (static_cast<int64_t>(odesc->m_body_value_index) == special_out_port) {
                    // If this is special out port, we will remove output description and output, but do not remove
                    // Result
                    to_remove = false;
                }
                if (to_remove) {
                    for (const auto& desc : merged_input_descs) {
                        const auto& mdesc = ov::as_type_ptr<MultiSubGraphOp::MergedInputDescription>(desc);
                        if (mdesc && mdesc->m_body_value_index == odesc->m_body_value_index) {
                            // Cannot remove Result which is part of merged input
                            to_remove = false;
                        }
                    }
                }
                if (to_remove) {
                    results_idxs_to_remove.insert(odesc->m_body_value_index);
                }
            }
            new_op_out_desc.push_back(new_out_desc);
            auto results = body->get_results();
            // go in reverse order to first delete last result
            for (auto it = results_idxs_to_remove.rbegin(); it != results_idxs_to_remove.rend(); ++it) {
                body->remove_result(results.at(*it));
                is_changed = true;
                // We need to go over output descriptors and modify them to reflect deleted result
                for (auto& desc : new_out_desc) {
                    if (desc->m_body_value_index > *it) {
                        desc->m_body_value_index--;
                    }
                }
                for (auto& desc : merged_input_descs) {
                    const auto& mdesc = ov::as_type_ptr<MultiSubGraphOp::MergedInputDescription>(desc);
                    if (mdesc && mdesc->m_body_value_index > *it) {
                        mdesc->m_body_value_index--;
                    }
                }
                if (special_out_port != -1) {
                    if (special_out_port > static_cast<int64_t>(*it)) {
                        special_out_port--;
                    }
                }
            }
            if (special_out_port != -1) {
                loop_op->set_special_body_ports(
                    {loop_op->get_special_body_ports().current_iteration_input_idx, special_out_port});
            }
        }
        // Remove inputs
        bool pass_required = false;
        std::set<uint64_t> required_inputs_indices;
        auto op_inputs = multi_subgraph_op->input_values();
        std::vector<std::vector<size_t>> to_remove_descriptors_indexes;
        to_remove_descriptors_indexes.resize(subgraphs_size);
        for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
            auto& body_func = multi_subgraph_op->get_function(static_cast<int>(body_idx));
            auto& body_params = body_func->get_parameters();
            auto& body_in_descriptors = multi_subgraph_op->get_input_descriptions(static_cast<int>(body_idx));
            // collect all descriptors which should be removed and required inputs
            for (size_t i = 0; i < body_in_descriptors.size(); ++i) {
                auto& body_param = body_params[body_in_descriptors[i]->m_body_parameter_index];
                if (body_param->get_output_target_inputs(0).size() == 0) {
                    to_remove_descriptors_indexes[body_idx].push_back(i);
                    pass_required = true;
                } else {
                    // collecting required inputs is needed to detect cases where the input
                    // is not needed in a one body, but the other one uses it (for example If case)
                    required_inputs_indices.insert(body_in_descriptors[i]->m_input_index);
                }
            }
        }
        if (pass_required) {
            is_changed = true;
            using DescType = op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector;
            auto update_body_param_desc = [](DescType& descriptors, uint64_t removed_body_idx) {
                for (auto& desc : descriptors) {
                    if (desc->m_body_parameter_index > removed_body_idx) {
                        desc->m_body_parameter_index--;
                    }
                }
            };
            auto update_op_inputs_desc = [&subgraphs_size](const std::shared_ptr<op::util::MultiSubGraphOp>& op,
                                                           std::set<uint64_t>& required_inputs_indices,
                                                           uint64_t removed_loop_idx) {
                std::set<uint64_t> new_required_inputs_indices;
                for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
                    auto& descriptors = op->get_input_descriptions(static_cast<int>(body_idx));
                    for (auto& desc : descriptors) {
                        if (desc->m_input_index > removed_loop_idx) {
                            desc->m_input_index--;
                        }
                    }
                }
                for (auto input_index : required_inputs_indices) {
                    if (input_index > removed_loop_idx) {
                        new_required_inputs_indices.insert(input_index - 1);
                    } else {
                        new_required_inputs_indices.insert(input_index);
                    }
                }
                required_inputs_indices = new_required_inputs_indices;
            };
            // Remove dangling body params and input and update input descriptors
            for (size_t body_idx = 0; body_idx < subgraphs_size; ++body_idx) {
                auto& body_in_descriptors = multi_subgraph_op->get_input_descriptions(static_cast<int>(body_idx));
                auto& body_func = multi_subgraph_op->get_function(static_cast<int>(body_idx));
                auto& body_params = body_func->get_parameters();
                op::util::MultiSubGraphOp::MultiSubgraphInputDescriptionVector updated_body_in_descriptors;
                for (size_t desc_idx = 0; desc_idx < body_in_descriptors.size(); ++desc_idx) {
                    if (std::count(std::begin(to_remove_descriptors_indexes[body_idx]),
                                   std::end(to_remove_descriptors_indexes[body_idx]),
                                   desc_idx) > 0) {
                        if (body_in_descriptors[desc_idx]->m_body_parameter_index < body_params.size()) {
                            auto& body_param = body_params[body_in_descriptors[desc_idx]->m_body_parameter_index];
                            body_func->remove_parameter(body_param);
                            // Move all body indexes which are after these indicated by to_remove_descriptors_indexes
                            update_body_param_desc(body_in_descriptors,
                                                   body_in_descriptors[desc_idx]->m_body_parameter_index);
                        }
                        // remove dangling input of MultiSubGraphOp which was not removed earlier
                        auto current_input_idx = body_in_descriptors[desc_idx]->m_input_index;
                        // the same input tensor can go to different input ports
                        if (current_input_idx < op_inputs.size() &&
                            std::count(std::begin(required_inputs_indices),
                                       std::end(required_inputs_indices),
                                       current_input_idx) == 0 &&
                            std::count(std::begin(op_inputs), std::end(op_inputs), op_inputs[current_input_idx]) > 0) {
                            op_inputs.erase(std::next(op_inputs.begin(), current_input_idx));
                            // Move all input indexes (in all bodies) which are after these indicated by
                            // to_remove_descriptors_indexes and are not used in any body
                            update_op_inputs_desc(multi_subgraph_op, required_inputs_indices, current_input_idx);
                        }
                    } else {
                        updated_body_in_descriptors.emplace_back(body_in_descriptors[desc_idx]);
                    }
                }
                multi_subgraph_op->set_input_descriptions(static_cast<int>(body_idx), updated_body_in_descriptors);
            }
            multi_subgraph_op->set_arguments(op_inputs);
        }
        if (!outputs_to_remove.empty()) {
            // we need to reconstruct operation with new number of outputs, we cannot reduce number of outputs of
            // existing op
            std::shared_ptr<MultiSubGraphOp> new_op;
            if (if_op) {
                new_op = std::make_shared<ov::op::v8::If>();
            } else if (loop_op) {
                auto new_loop_op = std::make_shared<ov::op::v5::Loop>();
                new_loop_op->set_special_body_ports(loop_op->get_special_body_ports());
                new_op = new_loop_op;
            } else if (ti_op) {
                new_op = std::make_shared<ov::op::v0::TensorIterator>();
            }
            new_op->set_arguments(multi_subgraph_op->input_values());
            new_op->set_friendly_name(multi_subgraph_op->get_friendly_name());
            copy_runtime_info(multi_subgraph_op, new_op);
            for (int body_idx = 0; static_cast<size_t>(body_idx) < subgraphs_size; ++body_idx) {
                new_op->set_function(body_idx, multi_subgraph_op->get_function(body_idx));
                new_op->set_input_descriptions(body_idx, multi_subgraph_op->get_input_descriptions(body_idx));
                new_op->set_output_descriptions(body_idx, new_op_out_desc.at(body_idx));
            }
            size_t removed_outs_counter = 0;
            new_op->set_output_size(multi_subgraph_op->get_output_size() - outputs_to_remove.size());
            for (size_t out_idx = 0; out_idx < multi_subgraph_op->get_output_size(); ++out_idx) {
                if (outputs_to_remove.find(out_idx) != outputs_to_remove.end()) {
                    // Need to go through all output descriptors to reflect deleted output
                    for (int body_idx = 0; static_cast<size_t>(body_idx) < subgraphs_size; ++body_idx) {
                        for (auto& odesc : new_op->get_output_descriptions(body_idx)) {
                            if (odesc->m_output_index > out_idx - removed_outs_counter) {
                                odesc->m_output_index--;
                            }
                        }
                    }
                    ++removed_outs_counter;
                } else {
                    // replace output with new one
                    multi_subgraph_op->output(out_idx).replace(new_op->output(out_idx - removed_outs_counter));
                }
            }
            new_op->validate_and_infer_types();
        }
    }
    return is_changed;
}
