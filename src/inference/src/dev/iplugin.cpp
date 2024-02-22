// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/iplugin.hpp"

#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/fused_names_cleanup.hpp"
#include "transformations/rt_info/fused_names_attribute.hpp"

namespace {

std::unordered_set<std::string> get_removed_nodes(const std::shared_ptr<const ov::Model>& originalFunction,
                                                  const std::shared_ptr<const ov::Model>& transformedFunction) {
    std::unordered_set<std::string> result = {};
    std::unordered_set<std::string> transformedNodeNames = {};

    for (auto&& node : transformedFunction->get_ops()) {
        transformedNodeNames.emplace(node->get_friendly_name());
        for (auto&& fusedLayerName : ov::getFusedNamesVector(node))
            transformedNodeNames.emplace(fusedLayerName);
    }

    for (auto&& originalNode : originalFunction->get_ops()) {
        if (!transformedNodeNames.count(originalNode->get_friendly_name()))
            result.emplace(originalNode->get_friendly_name());
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
    std::function<bool(const std::shared_ptr<ov::Node>)> is_node_supported) {
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
    auto ops = transformed_model->get_ordered_ops();

    // Mark removed nodes as supported
    std::unordered_set<std::string> supported = get_removed_nodes(model, transformed_model);
    std::unordered_set<std::string> unsupported;

    auto get_names_set = [](const std::shared_ptr<ov::Node>& op) -> std::unordered_set<std::string> {
        auto fused_names = ov::getFusedNamesVector(op);
        std::unordered_set<std::string> names(fused_names.begin(), fused_names.end());
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

    auto has_all_consumers_unsupported = [&supported](const std::shared_ptr<ov::Node>& node) {
        for (auto&& input : node->output(0).get_target_inputs()) {
            if (supported.count(input.get_node()->get_friendly_name())) {
                return false;
            }
        }
        return (node->output(0).get_target_inputs().size() != 0);
    };

    auto has_unsupported_source = [&supported](const std::shared_ptr<ov::Node>& node) {
        return !supported.count(node->input_values().begin()->get_node()->get_friendly_name());
    };

    auto remove_op_from_supported = [&](const std::shared_ptr<ov::Node>& node) {
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

    // Walk over transformed model for special handing of Parameters/Constants/Results
    for (auto&& op : ops) {
        // Mark Constants and all fused names as unsupported if they are have no
        // supported consumers/sources
        if (ov::op::util::is_constant(op)) {
            if (has_all_consumers_unsupported(op)) {
                remove_op_from_supported(op);
            }
        }
    }

    // Finally get intersection of all supported operation names
    // and operation names from original model
    std::unordered_set<std::string> res;
    for (auto&& name : supported) {
        if (original_ops.count(name)) {
            res.insert(name);
        }
    }

    // Remove parameters which has no supported consumers
    for (auto& param : model->get_parameters()) {
        if (has_all_consumers_unsupported(param)) {
            res.erase(param->get_friendly_name());
        }
    }

    // Remove results which has no supported source node
    for (auto& result : model->get_results()) {
        if (has_unsupported_source(result)) {
            res.erase(result->get_friendly_name());
        }
    }

    return res;
}
