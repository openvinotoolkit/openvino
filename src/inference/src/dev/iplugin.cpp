// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include <openvino/core/graph_util.hpp>

#include "openvino/op/broadcast.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fused_names_cleanup.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace {

std::unordered_set<std::string> get_removed_nodes(const std::shared_ptr<const ov::Model>& original_model,
                                                  const std::shared_ptr<const ov::Model>& transformed_model) {
    std::unordered_set<std::string> result = {};
    std::unordered_set<std::string> transformed_node_names = {};

    for (auto&& node : transformed_model->get_ops()) {
        transformed_node_names.emplace(node->get_friendly_name());
        for (auto&& fused_layer_name : ov::getFusedNamesVector(node)) {
            transformed_node_names.emplace(fused_layer_name);
        }
    }

    for (auto&& original_node : original_model->get_ops()) {
        if (!transformed_node_names.count(original_node->get_friendly_name()))
            result.emplace(original_node->get_friendly_name());
    }

    return result;
}

std::unordered_set<std::string> get_broadcast_fused_nodes(const std::shared_ptr<const ov::Model>& transformed_model) {
    std::unordered_set<std::string> result = {};

    for (auto&& node : transformed_model->get_ops()) {
        if (ov::is_type<ov::op::util::BroadcastBase>(node)) {
            for (auto&& fused_layer_name : ov::getFusedNamesVector(node)) {
                result.emplace(fused_layer_name);
                result.erase(node->get_friendly_name());
            }
        }
    }

    return result;
}

}  // namespace

ov::IPlugin::IPlugin() : m_executor_manager(ov::threading::executor_manager()) {}

void ov::IPlugin::set_version(const ov::Version& version) {
    m_version = version;
}

const ov::Version& ov::IPlugin::get_version() const {
    return m_version;
}

void ov::IPlugin::set_device_name(const std::string& name) {
    m_plugin_name = name;
}

const std::string& ov::IPlugin::get_device_name() const {
    return m_plugin_name;
}

void ov::IPlugin::set_core(const std::weak_ptr<ov::ICore>& core) {
    OPENVINO_ASSERT(!core.expired());
    m_core = core;
    auto locked_core = m_core.lock();
}

std::shared_ptr<ov::ICore> ov::IPlugin::get_core() const {
    return m_core.lock();
}

const std::shared_ptr<ov::threading::ExecutorManager>& ov::IPlugin::get_executor_manager() const {
    return m_executor_manager;
}

std::shared_ptr<ov::ICompiledModel> ov::IPlugin::compile_model(const std::string& model_path,
                                                               const ov::AnyMap& properties) const {
    auto core = get_core();
    OPENVINO_ASSERT(core);
    auto model = core->read_model(model_path, std::string());
    return compile_model(model, properties);
}

std::unordered_set<std::string> ov::get_supported_nodes(
    const std::shared_ptr<const ov::Model>& model,
    std::function<void(std::shared_ptr<ov::Model>&)> transform,
    std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported,
    uint64_t memory_size_in_bytes) {
    // Collect original operation names
    bool memory_control = memory_size_in_bytes > 0;
    std::unordered_set<std::string> original_ops;
    for (auto&& node : model->get_ops()) {
        original_ops.emplace(node->get_friendly_name());
    }

    auto transformed_model = model->clone();
    // Cleanup fused names if there are present in original model
    ov::pass::Manager m;
    m.register_pass<ov::pass::FusedNamesCleanup>();
    m.run_passes(transformed_model);

    transform(transformed_model);
    auto ops = transformed_model->get_ordered_ops();

    using NameSet = std::unordered_set<std::string>;
    using NodePtr = std::shared_ptr<ov::Node>;

    NameSet supported;
    NameSet unsupported;

    auto get_names_set = [](const NodePtr& op) -> NameSet {
        auto fused_names = ov::getFusedNamesVector(op);
        NameSet names(fused_names.begin(), fused_names.end());
        names.insert(op->get_friendly_name());
        return names;
    };

    // Collect all operation names even there are no such names in original model
    for (auto&& op : ops) {
        auto names = get_names_set(op);
        if (is_node_supported(op)) {
            supported.insert(names.begin(), names.end());
        } else {
            unsupported.insert(names.begin(), names.end());
        }
    }

    // If operation was fused into several operations where one is supported
    // but another one is not supported remove it from supported
    for (auto&& name : unsupported) {
        supported.erase(name);
    }

    auto get_output_node = [](const ov::Output<ov::Node>& output) -> NodePtr {
        return output.get_node_shared_ptr();
    };

    auto get_input_node = [&get_output_node](const ov::Input<ov::Node>& input) -> NodePtr {
        return get_output_node(input.get_source_output());
    };

    auto has_all_consumers_unsupported = [&](const NameSet& supported, const NodePtr& node) -> bool {
        bool has_consumers = false;
        for (auto&& output : node->outputs()) {
            for (auto&& input : output.get_target_inputs()) {
                has_consumers = true;
                if (supported.count(input.get_node()->get_friendly_name())) {
                    return false;
                }
            }
        }
        return has_consumers;
    };

    auto has_users_supported = [&](const NameSet& supported, const NodePtr& node) -> bool {
        auto users = node->get_users();
        for (auto& user : users) {
            if (supported.count(user->get_friendly_name())) {
                return true;
            }
        }
        return false;
    };

    auto has_unsupported_source =
        [&get_input_node](const NameSet& supported, const NodePtr& op, bool const_only = false) -> bool {
        for (auto& input : op->inputs()) {
            const auto& node = get_input_node(input);
            if (const_only && !ov::op::util::is_constant(node))
                continue;
            if (!supported.count(node->get_friendly_name())) {
                return true;
            }
        }
        return false;
    };

    auto remove_op_from_supported = [&](const NodePtr& node) {
        auto names = get_names_set(node);
        for (auto& name : get_names_set(node)) {
            supported.erase(name);
        }
    };

    auto check_pairs = [](std::map<std::string, int> pair_checker) {
        return std::all_of(pair_checker.begin(), pair_checker.end(), [](const std::pair<std::string, int>& val) {
            return val.second == 2;
        });
    };

    // Check the ops to make sure Assign and ReadValue operations in pairs on the network
    std::map<std::string, int> pair_checker;
    for (auto&& op : ops) {
        if (supported.count(op->get_friendly_name())) {
            if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
                if (pair_checker.count(assign->get_variable_id()) == 0) {
                    pair_checker[assign->get_variable_id()] = 1;
                } else {
                    pair_checker[assign->get_variable_id()]++;
                }
            }
        }
    }

    if (!check_pairs(pair_checker)) {
        for (auto& op : ops) {
            if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
                if (pair_checker[assign->get_variable_id()] == 1) {
                    remove_op_from_supported(op);
                }
            }
        }
    }

    bool start_split = false;
    unsigned long total_size = 0;
    std::map<std::string, int> pair_checker_temp;
    // Walk over transformed model for special handing of Parameters/Constants/Results
    for (auto&& op : ops) {
        // Mark Constants and all fused names as unsupported if they are have no
        // supported consumers/sources
        if (ov::op::util::is_constant(op)) {
            if (has_all_consumers_unsupported(supported, op)) {
                remove_op_from_supported(op);
                continue;
            }
        }
        if (memory_control && supported.count(op->get_friendly_name())) {
            if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
                if (pair_checker_temp.count(assign->get_variable_id()) == 0) {
                    pair_checker_temp[assign->get_variable_id()] = 1;
                } else {
                    pair_checker_temp[assign->get_variable_id()]++;
                }
            }
            if (ov::op::util::is_constant(op) && !start_split) {
                const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
                total_size += const_byte_size;
                if (total_size * 1.2 >= memory_size_in_bytes) {
                    if (!start_split) {
                        start_split = check_pairs(pair_checker_temp);
                    }
                }
            }
            if (start_split) {
                if (!ov::op::util::is_constant(op)) {
                    if (!has_unsupported_source(supported, op, false)) {
                        continue;
                    }
                    remove_op_from_supported(op);
                    for (auto& input : op->inputs()) {
                        const auto& node = get_input_node(input);
                        if (ov::op::util::is_constant(node)) {
                            remove_op_from_supported(node);
                        }
                    }
                } else {
                    remove_op_from_supported(op);
                }
            }
        }
    }

    // Get removed nodes and nodes fused in broadcast
    NameSet removed_nodes = get_removed_nodes(model, transformed_model);
    NameSet broadcast_fused_nodes = get_broadcast_fused_nodes(transformed_model);
    // Filter ShapeOfs & Broadcast, Broadcast can't be split with it's output ops
    for (auto& op : model->get_ordered_ops()) {
        const auto& name = op->get_friendly_name();
        if ((ov::is_type<ov::op::util::ShapeOfBase>(op) || ov::is_type<ov::op::util::BroadcastBase>(op)) &&
            (supported.count(name) || removed_nodes.count(name))) {
            if (has_all_consumers_unsupported(supported, op) && has_all_consumers_unsupported(removed_nodes, op)) {
                remove_op_from_supported(op);
                removed_nodes.erase(name);
            }
        }
    }

    if (memory_control) {
        for (auto& op : model->get_ordered_ops()) {
            const auto& name = op->get_friendly_name();
            // Some ops in other layers will be fused with Broadcast op in LLM, so need filter the fused ops in other layers
            if (broadcast_fused_nodes.count(name)) {
                if (has_all_consumers_unsupported(supported, op) && has_all_consumers_unsupported(removed_nodes, op)) {
                    remove_op_from_supported(op);
                    broadcast_fused_nodes.erase(name);
                }
            }
        }
    } else {
        // If memory control is off
        // mark all removed nodes as supported
        supported.insert(removed_nodes.begin(), removed_nodes.end());
    }
    
    // In case some ops not in orederd
    if (memory_control) {
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& op : model->get_ordered_ops()) {
                if (!supported.count(op->get_friendly_name()) && has_users_supported(supported, op) &&
                    !unsupported.count(op->get_friendly_name())) {
                    supported.insert(op->get_friendly_name());
                    changed = true;
                }
            }
        }
    }
    // Finally get intersection of all supported operation names
    // and operation names from original model
    NameSet res;
    for (auto&& name : supported) {
        if (original_ops.count(name)) {
            res.insert(name);
        }
    }

    // Remove parameters (or parameter + convert) which has no supported consumers
    // and results (or result + convert) which has no supported source node
    for (auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Convert>(op)) {
            if (ov::op::util::is_parameter(get_input_node(op->input(0))) && has_all_consumers_unsupported(res, op)) {
                res.erase(op->get_friendly_name());
            }
        } else {
            auto outputs = op->outputs();
            auto all_consumers_are_results =
                std::all_of(outputs.begin(), outputs.end(), [&](const ov::Output<ov::Node>& output) -> bool {
                    return ov::op::util::is_output(get_output_node(output));
                });
            if (all_consumers_are_results && has_unsupported_source(res, op, true)) {
                res.erase(op->get_friendly_name());
            }
        }
    }

    for (auto& param : model->get_parameters()) {
        if (has_all_consumers_unsupported(res, param)) {
            res.erase(param->get_friendly_name());
        }
    }

    for (auto& result : model->get_results()) {
        if (has_unsupported_source(res, result)) {
            res.erase(result->get_friendly_name());
        }
    }

    return res;
}
