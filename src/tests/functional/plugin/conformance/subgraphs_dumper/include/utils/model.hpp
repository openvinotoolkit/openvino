// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils/node.hpp"

namespace ov {
namespace util {

std::string get_model_type(const std::shared_ptr<ov::Model>& model);

std::map<std::string, ov::conformance::InputInfo>
get_input_info_by_model(const std::shared_ptr<ov::Model>& model);

std::map<std::string, ov::conformance::InputInfo>
align_input_info(const std::shared_ptr<ov::Model>& model,
                 const std::shared_ptr<ov::Model>& model_ref,
                 const std::map<std::string, ov::conformance::InputInfo> &in_info,
                 const std::map<std::string, ov::conformance::InputInfo> &in_info_ref,
                 const std::unordered_map<std::string, std::string> &matched_op);

// get set nodes of subgraph after start_node
void
get_subgraph_set_node(std::unordered_set<std::shared_ptr<ov::Node>>& nodes_to_check,
                      const std::shared_ptr<ov::Node>& node);

bool is_same_paired_op_cnt(const std::shared_ptr<ov::Model> &fist_model,
                           const std::shared_ptr<ov::Model> &second_model);

bool build_control_dependency(std::shared_ptr<ov::Model> &model);

inline std::pair<std::shared_ptr<ov::Model>, std::map<std::string, ov::conformance::InputInfo>>
generate_model(ov::NodeVector& nodes,
               bool is_copy_constants = true,
               bool is_save_only_borders = false) {
    // map to recover graph using cloned nodes and original connections
    // { original_node_name, cloned_node }
    std::unordered_map<std::string, std::shared_ptr<ov::Node>> cloned_node_map;
    // map to fill output nodes in models:
    // { original_node_names, out_port_idx_without_orig_node_to_check }
    std::unordered_map<std::string, std::unordered_set<size_t>> model_output_nodes;
    std::map<std::string, ov::conformance::InputInfo> model_input_info;
    ov::ParameterVector model_parameters;
    ov::SinkVector model_sinks;
    {
        // prepare map { original_op_name, cloned_node }
        size_t functional_node_cnt = 0;
        for (const auto& node : nodes) {
            auto orig_node_name = node->get_friendly_name();
            cloned_node_map.insert({ orig_node_name,
                                     clone_node(node, is_copy_constants, false, orig_node_name) });

            // create temporary vector to fill node output indexes
            std::vector<size_t> out_ports(node->outputs().size());
            std::iota(out_ports.begin(), out_ports.end(), 0);
            // fill by all nodes with output ports
            model_output_nodes.insert({orig_node_name, std::unordered_set<size_t>(out_ports.begin(), out_ports.end())});
            if (!ov::op::util::is_output(node) &&
                !ov::op::util::is_constant(node) &&
                !ov::op::util::is_parameter(node)) {
                ++functional_node_cnt;
            }
        }

        if (functional_node_cnt < 2) {
            throw std::runtime_error("Incorrect node number to create model!");
        }
        // replace new inputs by taken from graph if possible and
        // find input and output nodes in future subgraph
        for (const auto& node : nodes) {
            // variable to store updated input index
            int filled_input_idx = -1;
            auto cloned_node = cloned_node_map[node->get_friendly_name()];
            auto node_input_info = get_input_info_by_node(cloned_node);
            for (size_t in_idx = 0; in_idx < node->inputs().size(); ++in_idx) {
                auto orig_in_node = node->get_input_node_ptr(in_idx)->shared_from_this();
                for (size_t out_idx = 0; out_idx < orig_in_node->outputs().size(); ++out_idx) {
                    for (const auto& orig_node_to_check : orig_in_node->output(out_idx).get_target_inputs()) {
                        if (orig_node_to_check.get_node()->shared_from_this() == node) {
                            auto orig_in_node_name = orig_in_node->get_friendly_name();
                            auto cloned_in_node = cloned_node->get_input_node_shared_ptr(in_idx);
                            // if op input node is in subgraph replace parameters
                            // in cloned node by other nodes from the map
                            if (cloned_node_map.count(orig_in_node_name)) {
                                auto orig_in_node = cloned_node_map[orig_in_node_name];
                                auto cloned_in_node_name = cloned_in_node->get_friendly_name();
                                size_t cloned_in_node_out_idx = ov::op::util::is_parameter(cloned_in_node) ||
                                                                ov::op::util::is_constant(cloned_in_node) ? 0 : out_idx;
                                // cloned_in_node is parameter or constant, it could have only one input
                                ov::replace_output_update_name(cloned_in_node->output(cloned_in_node_out_idx), orig_in_node->output(out_idx));
                                if (ov::op::util::is_parameter(orig_in_node)) {
                                    auto param = ov::as_type_ptr<ov::op::v0::Parameter>(orig_in_node);
                                    model_parameters.push_back(param);
                                    node_input_info.insert({ orig_in_node->get_friendly_name(),
                                                             node_input_info[cloned_in_node_name]});
                                } else if (ov::op::util::is_constant(orig_in_node)) {
                                    auto op_to_replace = ov::as_type_ptr<ov::op::v0::Constant>(orig_in_node);
                                    auto param = convert_const_to_param(op_to_replace);
                                    if (param != nullptr) {
                                        model_parameters.push_back(param);
                                    }
                                    node_input_info.insert({ orig_in_node->get_friendly_name(),
                                                             node_input_info[cloned_in_node_name]});
                                } else if (ov::op::util::is_sink(cloned_node)) {
                                    model_sinks.push_back(ov::as_type_ptr<ov::op::Sink>(cloned_node->shared_from_this()));
                                }
                                filled_input_idx++;
                                // clean up replaced node data
                                node_input_info.erase(cloned_in_node_name);
                                model_output_nodes[orig_in_node_name].erase(out_idx);
                                if (model_output_nodes[orig_in_node_name].empty()) {
                                    model_output_nodes.erase(orig_in_node_name);
                                }
                            } else if (ov::op::util::is_parameter(cloned_in_node)) {
                                auto param = ov::as_type_ptr<ov::op::v0::Parameter>(cloned_in_node);
                                model_parameters.push_back(param);
                            } else if (ov::op::util::is_constant(cloned_in_node)) {
                                auto op_to_replace = ov::as_type_ptr<ov::op::v0::Constant>(cloned_in_node);
                                auto param = convert_const_to_param(op_to_replace);
                                if (param != nullptr) {
                                    model_parameters.push_back(param);
                                }
                            }
                            break;
                        }
                    }
                    if (filled_input_idx == in_idx) {
                        break;
                    }
                }
            }
            if (!node_input_info.empty()) {
                model_input_info.insert(node_input_info.begin(), node_input_info.end());
            }
        }
    }
    ov::ResultVector model_results;
    for (const auto& out_node_name : model_output_nodes) {
        auto out_node = cloned_node_map[out_node_name.first];
        if (ov::op::util::is_output(out_node)) {
            model_results.push_back(ov::as_type_ptr<ov::op::v0::Result>(out_node));
        } else {
            for (const auto& out_port_id : out_node_name.second) {
                model_results.push_back(std::make_shared<ov::op::v0::Result>(out_node->output(out_port_id)));
            }
        }
    }

    auto model = std::make_shared<ov::Model>(model_results, model_sinks, model_parameters);

    if (!build_control_dependency(model)) {
        throw std::runtime_error("Incorrect ReadValue/Assign amout, correct model could not be created!");
    }

    // prepare unique model name based on operations from model
    std::string string_to_hash;
    for (const auto& op : model->get_ordered_ops()) {
        bool is_erase_node = !is_save_only_borders;
        std::ostringstream result;
        result << op->get_type_info();
        for (size_t i = 0; i < op->inputs().size(); ++i) {
            const auto& in = op->input(i);
            if (!is_node_to_skip(op->get_input_node_shared_ptr(i))) {
                is_erase_node |= true;
            }
            result << in.get_element_type();
            result << in.get_partial_shape().rank();
            result << in.get_partial_shape().is_static();
        }
        for (const auto& out : op->outputs()) {
            for (const auto& target_input : out.get_target_inputs()) {
                if (!is_node_to_skip(target_input.get_node()->shared_from_this())) {
                    is_erase_node |= true;
                    break;
                }
            }
            result << out.get_element_type();
            result << out.get_partial_shape().rank();
            result << out.get_partial_shape().is_static();
        }
        string_to_hash += result.str();
        if (is_erase_node) {
            cloned_node_map.erase(op->get_friendly_name());
        }
    }
    for (const auto& in : model_input_info) {
        string_to_hash += (in.second.is_const ? "1" : "0");
    }
    auto h1 = std::hash<std::string>{}(string_to_hash);
    model->set_friendly_name(std::to_string(h1));
    for (auto it = nodes.begin(); it != nodes.end();) {
        if (cloned_node_map.count((*it)->get_friendly_name())) {
            it = nodes.erase(it);
        } else {
            ++it;
        }
    }

    return { model, model_input_info };
}

}  // namespace util
}  // namespace ov
