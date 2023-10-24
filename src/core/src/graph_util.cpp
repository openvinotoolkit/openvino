// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/graph_util.hpp"

#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "openvino/core/descriptor/tensor.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/visualize_tree.hpp"
#include "transformations/common_optimizations/compress_float_constants.hpp"
#include "transformations/common_optimizations/fused_names_cleanup.hpp"
#include "transformations/common_optimizations/mark_precision_sensitive_shapeof_subgraphs.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

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

            node_map[node.get()] = cloned_node;
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
            OPENVINO_THROW("Results should be of type op::Result");
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

OPENVINO_SUPPRESS_DEPRECATED_START
std::shared_ptr<ov::Model> clone_model(const ov::Model& func) {
    std::unordered_map<ov::Node*, std::shared_ptr<ov::Node>> nm;
    return clone_model(func, nm);
}

std::shared_ptr<ov::Model> clone_model(const ov::Model& func,
                                       std::unordered_map<Node*, std::shared_ptr<Node>>& node_map) {
    return ov::clone_ov_model(func, node_map);
}
OPENVINO_SUPPRESS_DEPRECATED_END

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

    bool preserve_legacy_output_name = false;
    if (has_result_consumers(output)) {
        preserve_legacy_output_name = true;
        if (output.get_node()->get_output_size() != 1 || replacement.get_node()->get_output_size() != 1 ||
            is_type<ov::op::v0::Parameter>(replacement.get_node()) || has_result_consumers(replacement)) {
            return false;
        }
    }

    OPENVINO_SUPPRESS_DEPRECATED_START
    if (preserve_legacy_output_name) {
        replacement.get_node()->set_friendly_name(output.get_node()->get_friendly_name());
        // Update output tensor name
        const auto& output_tensor_name = ov::descriptor::get_ov_tensor_legacy_name(output.get_tensor());
        if (!output_tensor_name.empty()) {
            ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), output_tensor_name);
        } else {
            ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), output.get_node()->get_friendly_name());
        }
    }

    // Save replacement tensor name before replacement as they will be overridden by the output tensor name
    const auto tensor_name = ov::descriptor::get_ov_tensor_legacy_name(replacement.get_tensor());

    output.replace(replacement);

    // Restore back original replacement tensor name
    ov::descriptor::set_ov_tensor_legacy_name(replacement.get_tensor(), tensor_name);
    OPENVINO_SUPPRESS_DEPRECATED_END

    copy_runtime_info({replacement.get_node_shared_ptr(), output.get_node_shared_ptr()},
                      replacement.get_node_shared_ptr());
    return true;
}

bool replace_node_update_name(const std::shared_ptr<Node>& target, const std::shared_ptr<Node>& replacement) {
    for (auto& output : target->output(0).get_target_inputs()) {
        if (ov::as_type<op::v0::Parameter>(replacement->input_value(0).get_node()) &&
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
    ov::pass::Manager manager;
    // TODO: if rt_info is set in python api as a string ['disable_fp16_compression_0'] = '',
    //  we need to convert value to a class in order to have rt_info in the IR. The code below will convert
    // ['disable_fp16_compression_0'] = '' into => rt_info['disable_fp16_compression_0'] = DisableFP16Compression{}
    for (auto& node : m->get_ops())
        if (fp16_compression_is_disabled(node))
            disable_fp16_compression(node);
    manager.register_pass<ov::pass::Serialize>(xml_path, bin_path, version);
    manager.run_passes(std::const_pointer_cast<ov::Model>(m));
}

void save_model(const std::shared_ptr<const ov::Model>& m, const std::string& output_model, bool compress_to_fp16) {
    ov::pass::Manager manager;
    if (compress_to_fp16) {
        manager.register_pass<ov::pass::MarkPrecisionSensitiveConstants>();
        manager.register_pass<ov::pass::CompressFloatConstants>(/*postponed=*/true);
    }
    manager.register_pass<ov::pass::FusedNamesCleanup>();
    manager.register_pass<ov::pass::Serialize>(output_model, "");
    auto cloned = m->clone();  // TODO: Implement on-the-fly compression in pass::Serialize
    manager.run_passes(cloned);
}

}  // namespace ov

OPENVINO_SUPPRESS_DEPRECATED_START

namespace ngraph {
ov::NodeVector find_common_args(std::shared_ptr<Node> node1, std::shared_ptr<Node> node2) {
    std::unordered_set<std::shared_ptr<Node>> node1_args;

    auto compute_node1_args = [&node1_args](const std::shared_ptr<Node>& node) {
        node1_args.insert(node);
    };

    traverse_nodes({std::move(node1)}, compute_node1_args, NodeVector{});

    std::unordered_set<std::shared_ptr<Node>> node2_args;

    auto compute_node2_args = [&node2_args](const std::shared_ptr<Node>& node) {
        node2_args.insert(node);
    };

    traverse_nodes({std::move(node2)}, compute_node2_args, NodeVector{});

    NodeVector common_args;
    for (const auto& e : node1_args) {
        if (node2_args.count(e) > 0) {
            common_args.push_back(e);
        }
    }

    return common_args;
}

// Check if all paths from X to a result go through Y
bool is_post_dominated(Node* X, Node* Y) {
    std::unordered_set<Node*> visited;
    std::stack<Node*, std::vector<Node*>> stack;
    stack.push(X);

    while (stack.size() > 0) {
        ov::Node* curr = stack.top();
        visited.insert(curr);
        if (ov::op::util::is_output(curr)) {
            return false;
        }
        stack.pop();
        if (curr != Y) {
            for (const auto& next : curr->get_users()) {
                if (visited.count(next.get()) == 0) {
                    stack.push(next.get());
                }
            }
        }
    }
    return true;
}

std::vector<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                   NodeMap& node_map) {
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    for (const auto& node : sorted_nodes) {
        if (node_map.count(node.get()) == 0) {
            // get (already) cloned arguments and clone the node
            OutputVector cloned_args;
            for (auto input : node->inputs()) {
                Output<Node> output = input.get_source_output();
                cloned_args.push_back(output.for_node(node_map.at(output.get_node())));
            }
            std::vector<std::shared_ptr<Node>> cloned_dependencies;
            for (auto& dependency : node->get_control_dependencies()) {
                std::shared_ptr<Node>& dependent = node_map.at(dependency.get());
                if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                    cloned_dependencies.end()) {
                    cloned_dependencies.push_back(dependent);
                }
            }
            auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
            // There is a friendly name for this node so copy it
            cloned_node->set_friendly_name(node->get_friendly_name());
            auto rt_info = node->get_rt_info();
            cloned_node->get_rt_info() = rt_info;

            for (auto output : node->outputs()) {
                const auto& output_rt_info = output.get_rt_info();
                auto new_output = output.for_node(cloned_node);
                new_output.get_rt_info() = output_rt_info;
            }

            for (auto input : node->inputs()) {
                const auto& output_rt_info = input.get_rt_info();
                auto new_input = cloned_node->input(input.get_index());
                new_input.get_rt_info() = output_rt_info;
            }

            node_map[node.get()] = cloned_node;
        }
    }

    // create and return vector of cloned nodes
    // order matches input vector (not necessarily topological)
    std::vector<std::shared_ptr<ov::Node>> cloned_nodes;
    for (const auto& node : nodes) {
        cloned_nodes.push_back(node_map.at(node.get()));
    }
    return cloned_nodes;
}

std::list<std::shared_ptr<ov::Node>> clone_nodes(const std::vector<std::shared_ptr<ov::Node>>& nodes,
                                                 RawNodeOutputMap& output_map) {
    // for each node in topological order
    auto sorted_nodes = topological_sort(nodes);
    std::list<std::shared_ptr<Node>> cloned_nodes;
    for (const auto& node : sorted_nodes) {
        auto node_outputs = node->outputs();
        for (const auto& value : node_outputs) {
            if (output_map.count(value) == 0) {
                // We need this node cloned
                // get (already) cloned arguments and clone the node
                OutputVector cloned_args;
                for (const auto& value : node->input_values()) {
                    cloned_args.push_back(output_map.at(value));
                }
                NodeVector cloned_dependencies;
                for (auto& dependency : node->get_control_dependencies()) {
                    for (const auto& dependency_value : dependency->outputs()) {
                        std::shared_ptr<Node> dependent = output_map.at(dependency_value).get_node_shared_ptr();
                        if (find(cloned_dependencies.begin(), cloned_dependencies.end(), dependent) ==
                            cloned_dependencies.end()) {
                            cloned_dependencies.push_back(dependent);
                        }
                    }
                }
                auto cloned_node = node->copy_with_new_inputs(cloned_args, cloned_dependencies);
                cloned_nodes.push_back(cloned_node);
                // There is a friendly name for this node so copy it
                cloned_node->set_friendly_name(node->get_friendly_name());
                auto rt_info = node->get_rt_info();
                cloned_node->get_rt_info() = rt_info;
                for (const auto& cloned_value : cloned_node->outputs()) {
                    auto original_value = node_outputs.at(cloned_value.get_index());
                    if (output_map.count(original_value) == 0) {
                        output_map[original_value] = cloned_value;
                    }
                }
                break;
            }
        }
    }
    return cloned_nodes;
}

bool is_equal_to_const_value(const std::string& const_value, const Output<Node>& reduce_constant) {
    if (auto rc = ov::as_type_ptr<ov::op::v0::Constant>(reduce_constant.get_node_shared_ptr())) {
        return (rc->get_all_data_elements_bitwise_identical() && rc->convert_value_to_string(0) == const_value);
    } else {
        return false;
    }
}

// Insert result and parameter node between src_node and dst_node by splitting the graph
//
// Before:                        |  After:
// (Device:0)         (Device:1)  |  (Device:0)         (Device:0)  (Device:1)         (Device:1)
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+  +-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |  |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     |  |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |  |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ res |  | par +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |  |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     |  |     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     |  |     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+  +-----+               +-----+
std::pair<std::shared_ptr<ov::op::v0::Result>, std::shared_ptr<ov::op::v0::Parameter>> insert_result_parameter_split(
    const std::shared_ptr<Node>& src_node,
    const std::shared_ptr<Node>& dst_node) {
    if (src_node->get_output_size() != 1) {
        OPENVINO_THROW("Multiple output per op not supported in graph partition yet.");
    }

    // Make parameter node
    std::shared_ptr<op::Parameter> par_node =
        std::make_shared<op::Parameter>(src_node->get_output_element_type(0), src_node->get_output_shape(0));

    // Fix input / output among src, dst and par
    std::vector<Input<Node>> dst_inputs = get_inputs_from(*src_node, *dst_node);
    OPENVINO_ASSERT(dst_inputs.size() == 1,
                    "insert_result_parameter_split encountered more than "
                    "one input between the source and destination nodes");
    auto& dst_input = dst_inputs[0];

    std::vector<Output<Node>> src_outputs = get_outputs_to(*src_node, *dst_node);
    OPENVINO_ASSERT(src_outputs.size() == 1,
                    "insert_result_parameter_split encountered more than "
                    "one output between the source and destination nodes");
    auto& src_output = src_outputs[0];

    // Remove [0]
    src_output.remove_target_input(dst_input);

    // Remove [0] (again), add [8], remove [1], add [9]
    dst_input.replace_source_output(par_node->output(0));

    // Add res node
    // Add [4], [5], [6], [7]
    std::shared_ptr<op::Result> res_node = std::make_shared<op::Result>(src_node);

    return make_pair(res_node, par_node);
}

// Insert unary node between two nodes like S->D => S->N->D
// Before:                        |  After:
// +-----+---+       +---+-----+  |  +-----+---+       +---+-----+---+       +---+-----+
// |     |   |       |   |     |  |  |     |   |       |   |     |   |       |   |     |
// |     | o +--[0]--> i |     |  |  |     | o +--[4]--> i |     | o +--[8]--> i |     |
// |     |   <--[1]--+   |     |  |  |     |   <--[5]--+   |     |   <--[9]--+   |     |
// | src +---+       +---+ dst |  |  | src +---+       +---+ new +---+       +---+ dst |
// |     |               |     |  |  |     |               |     |               |     |
// |     +------[2]------>     |  |  |     +------[6]------>     +------[10]----->     |
// |     <------[3]------+     |  |  |     <------[7]------+     <------[11]-----+     |
// +-----+               +-----+  |  +-----+               +-----+               +-----+
//                                |
// +-----+---+       +---+-----+  |
// |     |   |       |   |     |  |
// |     | o +--[4]--> i |     |  |
// |     |   <--[5]--+   |     |  |
// | src +---+       +---+ new |  |
// |     |               |     |  |
// |     +------[6]------>     |  |
// |     <------[7]------+     |  |
// +-----+               +-----+  |
//
// This cannot be achieved by ngraph::replace_node().
// With replace_node(), we could do:
// [     S           S      ]
// [    / \          |      ]
// [   /   \   =>    N      ]
// [  /     \       / \     ]
// [ D0     D1    D0   D1   ]
//
// But we want:
// [     S            S     ]
// [    / \          / \    ]
// [   /   \   =>   N0  N1  ]
// [  /     \      /     \  ]
// [ D0     D1    D0     D1 ]
//
// Typically new_node is connected to src_node already. The reason we don't create `new_node`
// inside the function and return it (similar to ngraph::insert_result_parameter_split) is that
// we'll have to templatize its function to call new_node's constructor.
void insert_new_node_between(const std::shared_ptr<Node>& src_node,
                             const std::shared_ptr<Node>& dst_node,
                             const std::shared_ptr<Node>& new_node) {
    // Fix input / output
    std::vector<Input<Node>> dst_inputs = get_inputs_from(*src_node, *dst_node);
    OPENVINO_ASSERT(dst_inputs.size() == 1,
                    "insert_new_node_between encountered more than one "
                    "input between the source and destination nodes");
    auto& dst_input = dst_inputs[0];

    std::vector<Output<Node>> src_outputs = get_outputs_to(*src_node, *dst_node);
    OPENVINO_ASSERT(src_outputs.size() == 1,
                    "insert_new_node_between encountered more than one "
                    "output between the source and destination nodes");
    auto& src_output = src_outputs[0];

    src_output.remove_target_input(dst_input);             // Remove [0]
    dst_input.replace_source_output(new_node->output(0));  // Remove [0] (again), add [8], remove [1], add [9]
}

std::shared_ptr<ov::Node> make_zero(const element::Type& element_type, const Shape& shape) {
    auto zero = ov::op::v0::Constant::create(element_type, Shape{}, {0.0});
    if (shape.size() > 0) {
        return std::make_shared<ov::op::v1::Broadcast>(
            zero,
            op::v0::Constant::create(element::u64, Shape{shape.size()}, shape));
    }
    return zero;
}

std::shared_ptr<ov::Node> make_constant_from_string(std::string val,
                                                    const element::Type& element_type,
                                                    const Shape& shape) {
    auto cvals = std::vector<std::string>(shape_size(shape), val);
    return std::make_shared<op::v0::Constant>(element_type, shape, cvals);
}

bool is_zero(const Output<Node>& reduce_constant) {
    auto result_bool = is_equal_to_const_value("0", reduce_constant);
    return result_bool;
}

bool is_one(const Output<Node>& reduce_constant) {
    auto result_bool = is_equal_to_const_value("1", reduce_constant);
    return result_bool;
}

ov::NodeVector get_subgraph_outputs(const NodeVector& nodes,
                                    const NodeVector& exclusions,
                                    bool ignore_unused,
                                    bool ignore_output_duplicates) {
    std::set<std::shared_ptr<Node>> exclusions_set(exclusions.begin(), exclusions.end());
    std::set<std::shared_ptr<Node>> nodes_set(nodes.begin(), nodes.end());

    NodeVector outputs;

    for (const auto& n : nodes) {
        if (exclusions_set.count(n) != 0) {
            continue;
        }

        for (const auto& u : n->get_users()) {
            bool add_output = nodes_set.count(u) == 0 && (!ignore_unused || is_used(u.get()));
            // check if output is already captured
            add_output &= (ignore_output_duplicates || std::find(outputs.begin(), outputs.end(), n) == outputs.end());
            if (add_output) {
                outputs.push_back(n);
            }
        }
    }
    return outputs;
}

ov::NodeVector extract_subgraph(const NodeVector& results, const NodeVector& args) {
    NodeVector subgraph;
    traverse_nodes(
        results,
        [&](const std::shared_ptr<Node>& n) {
            subgraph.push_back(n);
        },
        args);
    return subgraph;
}

bool is_used(Node* node) {
    std::unordered_set<Node*> instances_seen;
    std::stack<Node*, std::vector<Node*>> stack;
    stack.push(node);

    while (stack.size() > 0) {
        ov::Node* n = stack.top();
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

size_t get_user_count(Node* node) {
    size_t count = 0;
    for (const auto& node_user : node->get_users()) {
        count += is_used(node_user.get());
    }
    return count;
}

bool is_strided(const Strides& strides) {
    return std::any_of(strides.begin(), strides.end(), [](size_t stride) {
        return stride != 1;
    });
}

bool is_valid_rank(const std::shared_ptr<Node>& node, std::vector<size_t> valid_ranks) {
    auto node_rank = node->get_shape().size();
    for (auto rank : valid_ranks) {
        if (rank == node_rank) {
            return true;
        }
    }
    return false;
}

void plot_graph(std::shared_ptr<Function> f,
                const std::string& filename,
                std::function<void(const Node& node, std::vector<std::string>& attributes)> attributes) {
    ov::pass::Manager pass_manager;
    pass_manager.register_pass<ov::pass::VisualizeTree>(filename, attributes);
    pass_manager.run_passes(std::move(f));
}

std::vector<ov::Input<ov::Node>> get_inputs_from(Node& src, Node& dst) {
    std::vector<Input<Node>> result;

    for (auto& input : dst.inputs()) {
        if (input.get_source_output().get_node() == &src) {
            result.push_back(input);
        }
    }

    return result;
}

std::vector<ov::Output<ov::Node>> get_outputs_to(Node& src, Node& dst) {
    std::vector<Output<Node>> result;

    for (auto& output : src.outputs()) {
        bool targets_dst = false;

        for (auto& input : output.get_target_inputs()) {
            if (input.get_node() == &dst) {
                targets_dst = true;
                break;
            }
        }

        if (targets_dst) {
            result.push_back(output);
        }
    }

    return result;
}

static bool check_for_cycles_bkwd(const std::shared_ptr<ov::Node>& node,
                                  std::deque<std::shared_ptr<ov::Node>>& path,
                                  std::unordered_set<std::shared_ptr<ov::Node>>& path_set,
                                  ov::NodeVector& cycle_nodes) {
    path.push_back(node);
    path_set.insert(node);
    for (size_t i = 0; i < node->inputs().size(); i++) {
        auto arg = node->get_input_node_shared_ptr(i);
        if (path_set.find(arg) != path_set.end()) {
            for (const auto& it : path) {
                cycle_nodes.push_back(it);
            }
            // last node
            cycle_nodes.push_back(arg);
            return true;
        }
        if (check_for_cycles_bkwd(arg, path, path_set, cycle_nodes)) {
            return true;
        }
    }
    path_set.erase(path.back());
    path.pop_back();
    return false;
}

static bool check_for_cycles_fwd(const std::shared_ptr<ov::Node>& node,
                                 std::deque<std::shared_ptr<ov::Node>>& path,
                                 std::unordered_set<std::shared_ptr<ov::Node>>& path_set,
                                 ov::NodeVector& cycle_nodes) {
    path.push_back(node);
    path_set.insert(node);
    for (auto& arg : node->get_users()) {
        if (path_set.find(arg) != path_set.end()) {
            for (const auto& it : path) {
                cycle_nodes.push_back(it);
            }
            // last node
            cycle_nodes.push_back(arg);
            return true;
        }
        if (check_for_cycles_fwd(arg, path, path_set, cycle_nodes)) {
            return true;
        }
    }
    path_set.erase(path.back());
    path.pop_back();
    return false;
}

bool check_for_cycles(const ov::Model* func, ov::NodeVector& cycle_nodes, bool& is_bkwd_cycle) {
    for (const auto& res : func->get_results()) {
        std::deque<std::shared_ptr<Node>> path;
        // mirror of path stack for faster cycle check
        std::unordered_set<std::shared_ptr<Node>> path_set;
        if (check_for_cycles_bkwd(res, path, path_set, cycle_nodes)) {
            is_bkwd_cycle = true;
            return true;
        }
    }

    for (const auto& res : func->get_sinks()) {
        std::deque<std::shared_ptr<Node>> path;
        // mirror of path stack for faster cycle check
        std::unordered_set<std::shared_ptr<Node>> path_set;
        if (check_for_cycles_bkwd(res, path, path_set, cycle_nodes)) {
            is_bkwd_cycle = true;
            return true;
        }
    }

    for (const auto& param : func->get_parameters()) {
        std::deque<std::shared_ptr<Node>> path;
        // mirror of path stack for faster cycle check
        std::unordered_set<std::shared_ptr<Node>> path_set;
        if (check_for_cycles_fwd(param, path, path_set, cycle_nodes)) {
            is_bkwd_cycle = false;
            return true;
        }
    }
    // no cycles
    return false;
}

}  // namespace ngraph
