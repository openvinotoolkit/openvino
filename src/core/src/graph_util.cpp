// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/graph_util.hpp"

#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/descriptor_tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "openvino/util/env_util.hpp"
#include "openvino/util/file_util.hpp"
#include "transformations/common_optimizations/compress_float_constants.hpp"
#include "transformations/common_optimizations/fused_names_cleanup.hpp"

namespace {

void clone_ov_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>>& node_map) {
    // for each node in topological order
    for (const auto& node : nodes) {
        if (!node_map.count(node.get())) {
            // get (already) cloned arguments and clone the node
            ov::OutputVector cloned_args;
            for (const auto& input : node->inputs()) {
                ov::Output<ov::Node> output = input.get_source_output();
                cloned_args.push_back(output.for_node(node_map.at(output.get_node())));
            }
            std::vector<std::shared_ptr<ov::Node>> cloned_dependencies;
            for (const auto& dependency : node->get_control_dependencies()) {
                std::shared_ptr<ov::Node>& dependent = node_map.at(dependency.get());
                if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                    cloned_dependencies.end()) {
                    cloned_dependencies.push_back(dependent);
                }
            }
            auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
            // There is a friendly name for this node so copy it
            cloned_node->set_friendly_name(node->get_friendly_name());
            cloned_node->get_rt_info() = node->get_rt_info();

            for (const auto& output : node->outputs()) {
                cloned_node->output(output.get_index()).get_tensor().clone_from(output.get_tensor());
            }

            for (const auto& input : node->inputs()) {
                cloned_node->input(input.get_index()).get_rt_info() = input.get_rt_info();
            }

            node_map[node.get()] = std::move(cloned_node);
        }
    }
}

}  // namespace

namespace ov {

void traverse_nodes(const std::shared_ptr<const Model>& p, const std::function<void(const std::shared_ptr<Node>&)>& f) {
    traverse_nodes(p.get(), f);
}

void traverse_nodes(const Model* p, const std::function<void(const std::shared_ptr<Node>&)>& f) {
    NodeVector nodes;

    for (const auto& r : p->get_results()) {
        nodes.push_back(r);
    }
    for (auto s : p->get_sinks()) {
        nodes.emplace_back(s);
    }

    for (const auto& param : p->get_parameters()) {
        nodes.push_back(param);
    }

    traverse_nodes(nodes, f);
}

void traverse_nodes(const NodeVector& subgraph_results,
                    const std::function<void(const std::shared_ptr<Node>&)>& f,
                    const NodeVector& subgraph_params) {
    std::unordered_set<Node*> instances_seen;
    std::stack<Node*, std::vector<Node*>> stack;
    for (auto& node_ptr : subgraph_params) {
        instances_seen.insert(node_ptr.get());
    }
    for (auto& node_ptr : subgraph_results) {
        stack.push(node_ptr.get());
    }

    while (!stack.empty()) {
        Node* n = stack.top();
        stack.pop();
        if (instances_seen.insert(n).second) {
            f(n->shared_from_this());
            for (size_t i = 0; i < n->inputs().size(); i++) {
                stack.push(n->get_input_node_ptr(i));
            }

            for (auto& cdep : n->get_control_dependencies()) {
                stack.push(cdep.get());
            }
        }
    }
}

void replace_node(const std::shared_ptr<Node>& target,
                  const std::shared_ptr<Node>& replacement,
                  const std::vector<int64_t>& output_order) {
    if (ov::op::util::is_output(target)) {
        OPENVINO_THROW("Result nodes cannot be replaced.");
    }

    OPENVINO_ASSERT(target->get_output_size() == output_order.size(),
                    "Target output size: ",
                    target->get_output_size(),
                    " must be equal output_order size: ",
                    output_order.size());

    // Fix input/output descriptors
    OPENVINO_ASSERT(target->get_output_size() == replacement->get_output_size());

    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_output_size(); i++) {
        target->output(i).replace(replacement->output(output_order[i]));
    }

    replacement->add_node_control_dependents(target);
    replacement->add_node_control_dependencies(target);
    target->clear_control_dependents();
}

void replace_node(const std::shared_ptr<Node>& target, const OutputVector& replacement_values) {
    if (ov::op::util::is_output(target)) {
        OPENVINO_THROW("Result nodes cannot be replaced.");
    }

    OPENVINO_ASSERT(target->get_output_size() == replacement_values.size());

    std::unordered_set<std::shared_ptr<Node>> replacement_nodes;
    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_output_size(); i++) {
        auto& replacement_value = replacement_values.at(i);
        auto replacement_node = replacement_value.get_node_shared_ptr();
        if (replacement_nodes.find(replacement_node) == replacement_nodes.end()) {
            replacement_node->add_node_control_dependents(target);
            replacement_node->add_node_control_dependencies(target);
            replacement_nodes.insert(replacement_node);
        }
        target->output(i).replace(replacement_values.at(i));
    }
    target->clear_control_dependents();
}

void replace_node(const std::shared_ptr<Node>& target, const std::shared_ptr<Node>& replacement) {
    auto default_output_order = std::vector<int64_t>(target->get_output_size());
    std::iota(default_output_order.begin(), default_output_order.end(), 0);
    replace_node(target, replacement, default_output_order);
}

void replace_nodes(const std::shared_ptr<Model>& f,
                   const std::unordered_map<std::shared_ptr<ov::op::v0::Parameter>,
                                            std::shared_ptr<ov::op::v0::Parameter>>& parameter_replacement_map,
                   const std::unordered_map<std::shared_ptr<Node>, std::shared_ptr<Node>>& body_replacement_map) {
    auto& params = f->get_parameters();

    for (size_t i = 0; i < params.size(); i++) {
        if (parameter_replacement_map.count(params[i]) != 0 && parameter_replacement_map.at(params[i]) != params[i]) {
            f->replace_parameter(i, parameter_replacement_map.at(params[i]));
        }
    }

    for (auto& kv : body_replacement_map) {
        auto& k = kv.first;
        auto& v = kv.second;

        if (k != v) {
            f->replace_node(k, v);
        }
    }
}

std::shared_ptr<Model> clone_ov_model(const Model& func, std::unordered_map<Node*, std::shared_ptr<Node>>& node_map) {
    // clone model operations
    clone_ov_nodes(func.get_ordered_ops(), node_map);

    // clone variables
    auto variables = func.get_variables();
    ov::op::util::VariableVector cloned_vars;
    std::map<std::string, std::shared_ptr<ov::op::util::Variable>> var_map;
    for (const auto& var : variables) {
        auto cloned_var = std::make_shared<ov::op::util::Variable>(var->get_info());
        cloned_vars.push_back(cloned_var);
        var_map[cloned_var->get_info().variable_id] = cloned_var;
    }
    if (!variables.empty()) {
        for (const auto& op : node_map) {
            if (auto read_val = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op.second)) {
                read_val->set_variable(var_map.at(read_val->get_variable_id()));
            } else if (auto assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op.second)) {
                assign->set_variable(var_map.at(assign->get_variable_id()));
            }
        }
    }

    // get cloned model results and sinks and parameters
    ResultVector cloned_results;
    for (std::shared_ptr<Node> node : func.get_results()) {
        auto result = ov::as_type_ptr<op::v0::Result>(node_map.at(node.get()));
        if (!result) {
            OPENVINO_THROW("Results should be of type ov::op::v0::Result");
        }
        cloned_results.push_back(result);
    }
    SinkVector cloned_sinks;
    for (const auto& node : func.get_sinks()) {
        cloned_sinks.push_back(std::static_pointer_cast<op::Sink>(node_map.at(node.get())));
    }

    std::vector<std::shared_ptr<op::v0::Parameter>> cloned_params;
    for (const auto& param : func.get_parameters()) {
        cloned_params.push_back(ov::as_type_ptr<op::v0::Parameter>(node_map.at(param.get())));
    }

    // create and return cloned model
    auto result =
        std::make_shared<ov::Model>(cloned_results, cloned_sinks, cloned_params, cloned_vars, func.get_friendly_name());
    result->get_rt_info() = func.get_rt_info();
    result->m_shared_object = func.m_shared_object;
    return result;
}

bool compare_constants(const std::shared_ptr<Node>& n1, const std::shared_ptr<Node>& n2) {
    if (!(op::util::is_constant(n1) && op::util::is_constant(n2))) {
        return false;
    }

    if (std::static_pointer_cast<op::v0::Constant>(n1)->get_value_strings() !=
        std::static_pointer_cast<op::v0::Constant>(n2)->get_value_strings()) {
        return false;
    }

    return true;
}

bool replace_output_update_name(Output<Node> output, const Output<Node>& replacement) {
    // output port consumers can be reconnected to replacement port only when:
    // 1. output has no Result consumers (so we do not propagate node name)
    // 2. output has Result consumers and single output port and replacement doesn't have Results consumers
    //    and has exactly one output port
    // In all other cases output name will be lost or changed, so we don't perform the replacement

    auto has_result_consumers = [](const Output<Node>& port) {
        const auto& consumers = port.get_target_inputs();
        return std::any_of(consumers.cbegin(), consumers.cend(), [](const Input<Node>& consumer) {
            return ov::is_type<op::v0::Result>(consumer.get_node());
        });
    };

    if (has_result_consumers(output)) {
        if (output.get_node()->get_output_size() != 1 || replacement.get_node()->get_output_size() != 1 ||
            is_type<ov::op::v0::Parameter>(replacement.get_node()) || has_result_consumers(replacement)) {
            return false;
        }
        replacement.get_node()->set_friendly_name(output.get_node()->get_friendly_name());
    }

    output.replace(replacement);

    copy_runtime_info({replacement.get_node_shared_ptr(), output.get_node_shared_ptr()},
                      replacement.get_node_shared_ptr());
    return true;
}

bool replace_node_update_name(const std::shared_ptr<Node>& target, const std::shared_ptr<Node>& replacement) {
    for (auto& output : target->output(0).get_target_inputs()) {
        if (replacement->get_input_size() > 0 &&
            ov::as_type<op::v0::Parameter>(replacement->input_value(0).get_node()) &&
            ov::as_type<op::v0::Result>(output.get_node())) {
            return false;
        }
    }
    replace_node(target, replacement);
    replacement->set_friendly_name(target->get_friendly_name());
    copy_runtime_info(target, replacement);
    return true;
}

void serialize(const std::shared_ptr<const ov::Model>& m,
               const std::string& xml_path,
               const std::string& bin_path,
               ov::pass::Serialize::Version version) {
    ov::pass::Manager manager("Serialize");
    manager.register_pass<ov::pass::Serialize>(xml_path, bin_path, version);
    manager.run_passes(std::const_pointer_cast<ov::Model>(m));
}

void save_model(const std::shared_ptr<const ov::Model>& m, const std::string& output_model, bool compress_to_fp16) {
    auto cloned = m->clone();
    if (compress_to_fp16) {
        // TODO: Implement on-the-fly compression in pass::Serialize, Ticket: 145380
        bool postponed = true;
        ov::pass::compress_model_to_f16(cloned, postponed);
    }

    ov::pass::Manager manager("SaveModel");
    manager.register_pass<ov::pass::FusedNamesCleanup>();
    manager.register_pass<ov::pass::Serialize>(output_model, "");
    manager.run_passes(std::move(cloned));
}

#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT)
void save_model(const std::shared_ptr<const ov::Model>& m, const std::wstring& output_model, bool compress_to_fp16) {
    save_model(m, ov::util::wstring_to_string(output_model), compress_to_fp16);
}
#endif

bool is_used(Node* node);
bool is_used(Node* node) {
    std::unordered_set<Node*> instances_seen;
    std::stack<Node*, std::vector<Node*>> stack;
    stack.push(node);

    while (stack.size() > 0) {
        Node* n = stack.top();
        if (instances_seen.count(n) == 0) {
            if (ov::op::util::is_output(n)) {
                return true;
            }
            instances_seen.insert(n);
        }
        stack.pop();
        for (const auto& arg : n->get_users()) {
            if (instances_seen.count(arg.get()) == 0) {
                stack.push(arg.get());
            }
        }
    }
    return false;
}
}  // namespace ov
