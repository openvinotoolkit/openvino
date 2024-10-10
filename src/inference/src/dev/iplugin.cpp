// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include "openvino/op/convert.hpp"
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

bool is_graph_input_node(const std::shared_ptr<ov::Node>& node) {
    return ov::op::util::is_parameter(node) || ov::op::util::is_constant(node);
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
    float query_model_ratio) {
    using NameSet = std::unordered_set<std::string>;
    using NodePtr = std::shared_ptr<ov::Node>;
    NameSet res;
    if (query_model_ratio <= 0) {
        return res;
    }
    bool query_by_memory_control = query_model_ratio < 1;
    // Collect original operation names
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
    const auto& ops = transformed_model->get_ordered_ops();

    NameSet supported;
    NameSet unsupported;
    NameSet removed_nodes = get_removed_nodes(model, transformed_model);

    auto get_names_set = [](const NodePtr& op) -> NameSet {
        auto fused_names = ov::getFusedNamesVector(op);
        NameSet names(fused_names.begin(), fused_names.end());
        names.insert(op->get_friendly_name());
        return names;
    };

    // Collect all operation names even there are no such names in original model
    std::map<std::string, std::shared_ptr<ov::Node>> transformed_model_op_map;
    std::map<std::string, std::string> fused_model_op_map;
    for (const auto& op : ops) {
        auto names = get_names_set(op);
        for (auto& name : names) {
            if (name != op->get_friendly_name())
                fused_model_op_map[name] = op->get_friendly_name();
        }
        if (is_node_supported(op)) {
            supported.insert(names.begin(), names.end());
        } else {
            unsupported.insert(names.begin(), names.end());
        }
        transformed_model_op_map[op->get_friendly_name()] = op;
    }

    // If operation was fused into several operations where one is supported
    // but another one is not supported remove it from supported
    for (auto&& name : unsupported) {
        supported.erase(name);
    }

    auto copy_set = [](NameSet& source, NameSet& dest) {
        dest.clear();
        copy(source.begin(), source.end(), inserter(dest, dest.end()));
    };

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

    auto has_users_unsupported = [&](const NameSet& supported, const NodePtr& node) -> bool {
        auto users = node->get_users();
        for (auto& user : users) {
            if (!supported.count(user->get_friendly_name()) && !ov::is_type<ov::op::v0::Result>(user)) {
                return true;
            }
        }
        return false;
    };

    auto has_unsupported_source = [&get_input_node](const NameSet& supported,
                                                    const NodePtr& op,
                                                    bool const_only = false,
                                                    bool ignore_input = false) -> bool {
        for (auto& input : op->inputs()) {
            const auto& node = get_input_node(input);
            if (const_only && !ov::op::util::is_constant(node))
                continue;
            if (ignore_input && is_graph_input_node(node))
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

    auto insert_op_to_supported = [&](const NodePtr& node) {
        if (is_node_supported(node)) {
            auto names = get_names_set(node);
            for (auto& name : get_names_set(node)) {
                supported.insert(name);
            }
        }
    };

    auto check_pairs = [](std::map<std::string, int> pair_checker) {
        return std::all_of(pair_checker.begin(), pair_checker.end(), [](const std::pair<std::string, int>& val) {
            return val.second == 2;
        });
    };

    auto check_variables = [&](std::map<std::string, int>& pair_checker, const NameSet& supported) {
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
        return check_pairs(pair_checker);
    };

    // Check the ops to make sure Assign and ReadValue operations in pairs on the network
    std::map<std::string, int> pair_checker;
    if (!check_variables(pair_checker, supported)) {
        for (auto& op : ops) {
            if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
                if (pair_checker[assign->get_variable_id()] == 1) {
                    remove_op_from_supported(op);
                }
            }
        }
    }

    for (auto&& op : ops) {
        // Mark Constants and all fused names as unsupported if they are have no
        // supported consumers/sources
        if (ov::op::util::is_constant(op)) {
            if (has_all_consumers_unsupported(supported, op)) {
                remove_op_from_supported(op);
                continue;
            }
        }
    }

    size_t total_ops_size = 0;
    for (auto&& op : ops) {
        if (ov::op::util::is_constant(op)) {
            const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
            total_ops_size += const_byte_size;
        }
    }
    // If there is no constant or supported nodes in the model, mark query_by_memory_control as false
    if (total_ops_size == 0 || supported.size() == 0) {
        query_by_memory_control = false;
    }
    // mark all removed nodes as supported
    supported.insert(removed_nodes.begin(), removed_nodes.end());
    if (query_by_memory_control) {
        NameSet temp_supported;
        NameSet temp_unsupported;
        NameSet temp_supported_1;
        NameSet temp_unsupported_1;
        bool cancel_split = false;
        std::set<std::string> split_node_set;
        int64_t last_total_len = 0;
        int search_times = 0;
        size_t last_total_size = 0;
        double min_query_size = query_model_ratio * total_ops_size * 0.95;
        double max_query_size = query_model_ratio * total_ops_size * 1.05;
        copy_set(supported, temp_supported);
        copy_set(unsupported, temp_unsupported);
        // Search the smallest transmission node within the user's requested ratio range of 0.95-1.05 times
        do {
            std::map<std::string, int> temp_pair_checker;
            bool ready_split = false;
            bool start_split = false;
            bool has_min_graph = false;
            size_t total_size = 0;
            search_times++;
            copy_set(temp_supported, supported);
            copy_set(temp_unsupported, unsupported);
            // Walk over transformed model for special handing of Parameters/Constants/Results
            for (auto&& op : ops) {
                if (supported.count(op->get_friendly_name()) && !cancel_split) {
                    if (const auto& assign = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
                        if (temp_pair_checker.count(assign->get_variable_id()) == 0) {
                            temp_pair_checker[assign->get_variable_id()] = 1;
                        } else {
                            temp_pair_checker[assign->get_variable_id()]++;
                        }
                    }
                    if (ov::op::util::is_constant(op) && !ready_split) {
                        const auto const_byte_size = op->get_element_type().size() * shape_size(op->get_shape());
                        total_size += const_byte_size;
                        // If the total size is 1.05 times larger than the user's requirement:
                        // - If has_min_graph = false, it means there is no nodes meets requirement, so need cancel
                        //   split and break
                        // - If the search_times > 1, it means this is not the first search in do-while, so cancel
                        //   split and break
                        if (total_size <= max_query_size) {
                            has_min_graph = true;
                        } else if (!has_min_graph || search_times > 1) {
                            cancel_split = true;
                            break;
                        }
                        // Ready to split if total size meets user's requirement and Assign-ReadValue operations in
                        // pairs on the network
                        if (total_size >= min_query_size) {
                            if (!ready_split && split_node_set.find(op->get_friendly_name()) == split_node_set.end()) {
                                ready_split = check_pairs(temp_pair_checker);
                                if (ready_split) {
                                    split_node_set.insert(op->get_friendly_name());
                                    // Judge if the current constant op should be removed from supported
                                    if (total_size < max_query_size)
                                        continue;
                                }
                            }
                        }
                    }
                    // Start splitting when ready and the ops is constant
                    if (ready_split) {
                        if (ov::op::util::is_constant(op)) {
                            supported.erase(op->get_friendly_name());
                            start_split = true;
                        } else if (start_split) {
                            supported.erase(op->get_friendly_name());
                            for (auto& input : op->inputs()) {
                                const auto& node = get_input_node(input);
                                if (ov::op::util::is_constant(node)) {
                                    supported.erase(node->get_friendly_name());
                                }
                            }
                        }
                    }
                }
            }
            // Add the ops to supported that removed by transformations and it has supported users
            // For example:
            //
            //       A (to be marked as supported)
            //       |
            //       B (already in supported)
            //
            for (auto& op : model->get_ordered_ops()) {
                const auto& name = op->get_friendly_name();
                if (!supported.count(name)) {
                    if (!has_all_consumers_unsupported(supported, op)) {
                        if (transformed_model_op_map.find(name) != transformed_model_op_map.end()) {
                            insert_op_to_supported(transformed_model_op_map[name]);
                        } else {
                            insert_op_to_supported(op);
                        }
                    }
                }
            }
            // For example, A op need to be removed from supported:
            //              B (unsupported)
            //              |
            //              A (to be marked as unsupported)
            //
            //
            for (auto& op : model->get_ordered_ops()) {
                const auto& name = op->get_friendly_name();
                if (has_unsupported_source(supported, op, false, true) && supported.count(name)) {
                    supported.erase(name);
                }
            }
            // For example, A op need to be removed from supported:
            //
            //              A (constant, to be marked as unsupported)
            //     _________|_________
            //    |                   |
            //    B (unsupported)     C (unsupported)
            //
            for (auto& op : model->get_ordered_ops()) {
                const auto& name = op->get_friendly_name();
                if ((ov::op::util::is_constant(op) && has_all_consumers_unsupported(supported, op)) &&
                    supported.count(name)) {
                    supported.erase(name);
                }
            }
            // For example, A op need to be removed from supported:
            //              A (removed nodes, to be marked as unsupported)
            //              |
            //              B (unsupported)
            //
            bool update_supported = true;
            while (update_supported) {
                update_supported = false;
                for (auto& op : model->get_ordered_ops()) {
                    const auto& name = op->get_friendly_name();
                    if (removed_nodes.count(name) && supported.count(name)) {
                        if (has_all_consumers_unsupported(supported, op)) {
                            supported.erase(name);
                            removed_nodes.erase(name);
                            update_supported = true;
                        }
                    }
                }
            }
            // For example, A op need to be removed from supported:
            //              A (fused on B, to be marked as unsupported)
            //              |
            //              B (unsupported)
            //
            //            A ShapeOf (to be marked as unsupported)
            //              |
            //              B (unsupported)
            //
            update_supported = true;
            while (update_supported) {
                update_supported = false;
                for (auto& op : model->get_ordered_ops()) {
                    const auto& name = op->get_friendly_name();
                    bool is_shapeof = ov::is_type<op::util::ShapeOfBase>(op);
                    if (((fused_model_op_map.find(name) != fused_model_op_map.end()) || is_shapeof) &&
                        supported.count(name)) {
                        if ((!supported.count(fused_model_op_map[name]) || is_shapeof) &&
                            has_all_consumers_unsupported(supported, op)) {
                            supported.erase(name);
                            update_supported = true;
                        }
                    }
                }
            }
            // Calculate the data size that needs to be transmitted after the current model is split
            std::map<std::string, int> temp_pair_checker_2;
            if (check_variables(temp_pair_checker_2, supported)) {
                int64_t total_len = 0;
                for (auto& op : model->get_ordered_ops()) {
                    if (supported.count(op->get_friendly_name()) && !ov::op::util::is_constant(op) &&
                        !ov::op::util::is_parameter(op)) {
                        if (has_users_unsupported(supported, op)) {
                            int64_t op_size = 1;
                            for (size_t shape_id = 0; shape_id < op->get_output_partial_shape(0).size(); shape_id++) {
                                if (!op->get_output_partial_shape(0)[shape_id].is_dynamic()) {
                                    int64_t len = op->get_output_partial_shape(0)[shape_id].get_length();
                                    if (len >= 1)
                                        op_size *= len;
                                }
                            }
                            total_len += op_size;
                        }
                    }
                }
                if ((total_len < last_total_len || last_total_len == 0) && !cancel_split) {
                    last_total_len = total_len;
                    copy_set(supported, temp_supported_1);
                    copy_set(unsupported, temp_unsupported_1);
                }
            }
            // Cancel split when total size is unchanged in loop
            if (total_size != last_total_size) {
                last_total_size = total_size;
            } else {
                cancel_split = true;
            }
        } while (!cancel_split);
        copy_set(temp_supported_1, supported);
        copy_set(temp_unsupported_1, unsupported);
    }
    // Finally get intersection of all supported operation names
    // and operation names from original model
    for (auto&& name : supported) {
        if (original_ops.count(name)) {
            res.insert(name);
        }
    }
    // Remove parameters (or parameter/constant + convert) which has no supported consumers
    // and results (or result + convert) which has no supported source node
    for (auto& op : model->get_ordered_ops()) {
        if (ov::is_type<ov::op::v0::Convert>(op)) {
            if (is_graph_input_node(get_input_node(op->input(0))) && has_all_consumers_unsupported(res, op)) {
                res.erase(op->get_friendly_name());
                res.erase(get_input_node(op->input(0))->get_friendly_name());
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
