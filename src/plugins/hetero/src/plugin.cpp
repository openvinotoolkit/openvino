// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin.hpp"

#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "compiled_model.hpp"
#include "itt.hpp"
#include "op/device_subgraph.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/shared_buffer.hpp"
#include "openvino/util/common_util.hpp"
#include "properties.hpp"
#include "remote_context.hpp"

ov::hetero::Plugin::Plugin() {
    set_device_name("HETERO");
}

std::pair<ov::hetero::SubgraphsMappingInfo, std::vector<ov::hetero::SubmodelInfo>> ov::hetero::Plugin::split_graph(
    const std::shared_ptr<ov::Model>& model,
    Configuration config) const {
    std::vector<ov::hetero::SubmodelInfo> submodels;
    ov::SupportedOpsMap query_model_result;
    SubgraphsMappingInfo mapping_info;
    const std::string model_name = model->get_friendly_name();
    bool user_set_affinities = false;
    // Get user defined affinity
    for (const auto& node : model->get_ordered_ops()) {
        const auto& rt_info = node->get_rt_info();
        const auto it = rt_info.find("affinity");
        if (it != rt_info.end()) {
            OPENVINO_ASSERT(it->second.is<std::string>(), "Unexpected type of \"affinity\" attribute");
            query_model_result.emplace(node->get_friendly_name(), it->second.as<std::string>());
            user_set_affinities = true;
        }
    }

    if (user_set_affinities) {
        // All affinities must be defined by user
        ov::hetero::SubgraphsVector ordered_subgraphs;
        std::tie(ordered_subgraphs, mapping_info) =
            get_model_subgraphs(model, query_model_result, true, m_cfg.dump_dot_files());

        submodels.resize(ordered_subgraphs.size());
        for (size_t i = 0; i < ordered_subgraphs.size(); ++i) {
            const auto& subgraph = ordered_subgraphs[i];
            submodels[i].first = subgraph._affinity;
            submodels[i].second = std::make_shared<ov::Model>(subgraph._results,
                                                              subgraph._sinks,
                                                              subgraph._parameters,
                                                              model_name + "_" + std::to_string(i));
        }

        return {mapping_info, submodels};
    }

    // Restore properties in order to pass "device priorities" together
    // with devices properties
    auto full_properties = config.get_hetero_properties();
    for (const auto& [device, props] : config.get_device_properties()) {
        full_properties[device] = props;
    }

    auto cloned_model = model->clone();
    std::tie(query_model_result, mapping_info) = query_model_update(cloned_model, full_properties, true);

    ov::hetero::op::DeviceSubgraphVector ordered_subgraphs;
    for (const auto& op : cloned_model->get_ordered_ops()) {
        if (const auto& subgraph = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
            ordered_subgraphs.push_back(subgraph);
        } else {
            OPENVINO_ASSERT(ov::op::util::is_output(op) || ov::op::util::is_parameter(op) || ov::op::util::is_sink(op),
                            "Unexpected node type found in model after query_model_update()");
        }
    }

    submodels.resize(ordered_subgraphs.size());
    for (size_t i = 0; i < ordered_subgraphs.size(); ++i) {
        submodels[i].first = ordered_subgraphs[i]->get_affinity();
        submodels[i].second = ordered_subgraphs[i]->get_function();
    }

    return {mapping_info, submodels};
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::compile_model");

    auto config = Configuration{properties, m_cfg};
    auto cloned_model = model->clone();
    SubgraphsMappingInfo mapping_info;
    std::vector<ov::hetero::SubmodelInfo> submodels;
    std::tie(mapping_info, submodels) = split_graph(cloned_model, config);
    ov::hetero::RemoteContext::Ptr remote_context;
    try {
        std::map<std::string, ov::SoPtr<ov::IRemoteContext>> contexts_map;
        for (const auto& [device_name, _] : submodels) {
            contexts_map.insert({device_name, get_core()->get_default_context(device_name)});
        }
        remote_context = std::make_shared<ov::hetero::RemoteContext>(std::move(contexts_map));
    } catch (const ov::Exception&) {
    }
    return std::make_shared<CompiledModel>(cloned_model,
                                           submodels,
                                           mapping_info,
                                           shared_from_this(),
                                           remote_context,
                                           config);
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(
    const std::shared_ptr<const ov::Model>& model,
    const ov::AnyMap& properties,
    const ov::SoPtr<ov::IRemoteContext>& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                                     const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}
std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(std::istream& model,
                                                                     const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::import_model");

    // check ov::loaded_from_cache property and erase it due to not needed any more.
    auto _properties = properties;
    const auto& it = _properties.find(ov::loaded_from_cache.name());
    bool loaded_from_cache = false;
    if (it != _properties.end()) {
        loaded_from_cache = it->second.as<bool>();
        _properties.erase(it);
    }

    auto config = Configuration{_properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(model, shared_from_this(), config, loaded_from_cache);
    return compiled_model;
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(const ov::Tensor& model,
                                                                     const ov::AnyMap& properties) const {
    ov::SharedStreamBuffer buffer{reinterpret_cast<char*>(model.data()), model.get_byte_size()};
    std::istream stream{&buffer};
    return import_model(stream, properties);
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::import_model(const ov::Tensor& model,
                                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                                     const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::hetero::Plugin::DeviceProperties ov::hetero::Plugin::get_properties_per_device(const std::string& device_priorities,
                                                                                   const ov::AnyMap& properties) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (const auto& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it)
            device_properties[device_name] = get_core()->get_supported_property(device_name, properties, false);
    }
    return device_properties;
}

void ov::hetero::Plugin::get_device_memory_map(const std::vector<std::string>& device_names,
                                               std::map<std::string, size_t>& available_device_mem_map) const {
    // TODO: add unified API to get device memory.
    // There is no unified API to get device memory. So this feature get memory of specific device with specific method.
    // Skip device which cannot get device memory size.
    for (const auto& device_name : device_names) {
        if (device_name.find("CPU") != std::string::npos) {
            // Assuming the CPU has enough memory
            available_device_mem_map["CPU"] = -1;
        } else if (device_name.find("GPU") != std::string::npos) {
            try {
                size_t device_mem = get_core()->get_property(device_name, ov::intel_gpu::device_total_mem_size);
                available_device_mem_map[device_name] = device_mem;
            } catch (const ov::Exception&) {
            }
        }
    }
}

std::pair<ov::SupportedOpsMap, ov::hetero::SubgraphsMappingInfo> ov::hetero::Plugin::query_model_update(
    std::shared_ptr<ov::Model>& model,
    const ov::AnyMap& properties,
    bool allow_exception) const {
    std::map<std::string, size_t> available_device_mem_map;
    Configuration full_config{properties, m_cfg};
    DeviceProperties properties_per_device =
        get_properties_per_device(full_config.device_priorities, full_config.get_device_properties());

    //  WARNING: Here is devices with user set priority
    auto device_names = ov::DeviceIDParser::get_hetero_devices(full_config.device_priorities);
    bool hetero_query_model_by_device = false;
    if (full_config.modelDistributionPolicy.count(ov::hint::ModelDistributionPolicy::PIPELINE_PARALLEL) != 0) {
        get_device_memory_map(device_names, available_device_mem_map);
        // Will disable hetero query model by device if there is no device's available memory is obtained.
        if (available_device_mem_map.size() != 0) {
            hetero_query_model_by_device = true;
        }
    }

    auto update_supported_ops = [](ov::SupportedOpsMap& final_results, const ov::SupportedOpsMap& device_results) {
        for (const auto& layer_query_result : device_results)
            final_results.emplace(layer_query_result);
    };

    auto has_subgraph_ops = [](std::shared_ptr<ov::Model>& model) {
        for (auto& op : model->get_ordered_ops()) {
            if (ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                return true;
            }
        }
        return false;
    };

    auto update_config = [&](ov::AnyMap& device_config,
                             const std::shared_ptr<const ov::Model>& model,
                             std::string device_name,
                             bool fallback_device) {
        auto internal_supported_properties = get_core()->get_property(device_name, ov::internal::supported_properties);
        if (ov::util::contains(internal_supported_properties, ov::internal::query_model_ratio)) {
            if (fallback_device) {
                device_config[ov::internal::query_model_ratio.name()] = 1.0f;
            } else if (available_device_mem_map.count(device_name)) {
                size_t total_ops_size = 0;
                size_t available_discrete_device_memory = 0;
                for (auto&& op : model->get_ordered_ops()) {
                    if (ov::op::util::is_constant(op)) {
                        total_ops_size += op->get_element_type().size() * shape_size(op->get_shape());
                    }
                }
                for (auto& device_mem_info : available_device_mem_map) {
                    if (device_mem_info.first.find("CPU") != 0)
                        available_discrete_device_memory += device_mem_info.second;
                }
                // Estimate the memory size required for the model is 1.2 * total_ops_size
                // 1. Check if current device that can take the entire model
                // 2. Check if all left devices can take the entire model
                if (available_device_mem_map[device_name] >= 1.2 * total_ops_size || device_name.find("CPU") == 0) {
                    device_config[ov::internal::query_model_ratio.name()] = 1.0f;
                } else if (available_discrete_device_memory >= 1.2 * total_ops_size ||
                           available_device_mem_map.count("CPU")) {
                    float model_ratio =
                        total_ops_size > 0
                            ? static_cast<float>(available_device_mem_map[device_name] * 1.0 / (1.2 * total_ops_size))
                            : 1.0f;
                    if (total_ops_size < available_device_mem_map[device_name]) {
                        model_ratio = 1.0f;
                    }
                    device_config[ov::internal::query_model_ratio.name()] = model_ratio;
                } else {
                    float model_ratio = available_discrete_device_memory > 0
                                            ? static_cast<float>(available_device_mem_map[device_name] * 1.0 /
                                                                 available_discrete_device_memory)
                                            : 1.0f;
                    device_config[ov::internal::query_model_ratio.name()] = model_ratio;
                }
                // Remove the current device
                available_device_mem_map.erase(device_name);
            }
        }
    };

    ov::SupportedOpsMap supported_ops_temp;
    ov::SupportedOpsMap supported_ops_temp_1;
    ov::SupportedOpsMap supported_ops_final;
    std::map<std::string, ov::SupportedOpsMap> query_results;
    ov::hetero::SubgraphsMappingInfo mapping_info;
    ResultVector new_outputs;
    for (auto& param : model->get_parameters()) {
        if (param->get_users().size() == 0) {
            auto result = std::make_shared<ov::op::v0::Result>(param);
            ov::copy_runtime_info(param->shared_from_this(), result);
            new_outputs.push_back(result);
            independent_submodel_size++;
        }
    }
    model->add_results(new_outputs);
    for (const auto& device_name : device_names) {
        // If there are some unsupported operations and it is a last device
        // exception should be raised when allowed
        bool fallback_device = (device_name == device_names.back());
        const auto& default_device = (!allow_exception || !fallback_device) ? get_device_name() : "";
        auto& device_config = properties_per_device.at(device_name);
        if (!has_subgraph_ops(model)) {
            if (hetero_query_model_by_device)
                update_config(device_config, model, device_name, fallback_device);
            query_results[device_name] = get_core()->query_model(model, device_name, device_config);
            update_supported_ops(supported_ops_temp, query_results[device_name]);
            update_supported_ops(supported_ops_final, query_results[device_name]);
            mapping_info = ov::hetero::mask_model_subgraphs_by_ops(model,
                                                                   supported_ops_temp,
                                                                   m_cfg.dump_dot_files(),
                                                                   default_device);
        } else {
            // Mask supported nodes and left nodes to Subgraph in graph, and query model use subgraph, keep the
            // model in query_model same as compile
            auto temp_model = model->clone();
            update_supported_ops(supported_ops_temp_1, supported_ops_temp);
            for (auto&& node : temp_model->get_ops()) {
                supported_ops_temp_1.emplace(node->get_friendly_name(), "HETERO-TEMP");
            }
            auto mapping_info_temp =
                ov::hetero::mask_model_subgraphs_by_ops(temp_model, supported_ops_temp_1, false, default_device);
            for (const auto& op : temp_model->get_ordered_ops()) {
                if (const auto& subgraph = ov::as_type_ptr<ov::hetero::op::DeviceSubgraph>(op)) {
                    if (subgraph->get_affinity() == "HETERO-TEMP") {
                        if (hetero_query_model_by_device)
                            update_config(device_config, subgraph->get_function(), device_name, fallback_device);
                        query_results[device_name] =
                            get_core()->query_model(subgraph->get_function(), device_name, device_config);
                        update_supported_ops(supported_ops_temp, query_results[device_name]);
                        update_supported_ops(supported_ops_final, query_results[device_name]);
                    }
                }
            }
            mapping_info = ov::hetero::mask_model_subgraphs_by_ops(model,
                                                                   supported_ops_temp,
                                                                   m_cfg.dump_dot_files(),
                                                                   default_device);
        }
    }
    return {supported_ops_final, mapping_info};
}

ov::SupportedOpsMap ov::hetero::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                    const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::query_model");

    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    std::shared_ptr<ov::Model> query_model = model->clone();

    return query_model_update(query_model, properties).first;
}

void ov::hetero::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg, true};
}

ov::Any ov::hetero::Plugin::get_property(const std::string& name, const ov::AnyMap& properties) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities, ov::hint::model_distribution_policy};
        return rw_properties;
    };

    Configuration full_config{properties, m_cfg};
    if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(std::move(supported_properties));
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{get_device_name()};
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        return caching_device_properties(full_config.device_priorities);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return full_config.get(name);
    }
}

ov::Any ov::hetero::Plugin::caching_device_properties(const std::string& device_priorities) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    // Vector of caching properties per device
    std::vector<ov::AnyMap> result = {};
    for (const auto& device_name : device_names) {
        ov::AnyMap properties = {};
        auto supported_properties = get_core()->get_property(device_name, ov::supported_properties);
        auto supported_internal_properties = get_core()->get_property(device_name, ov::internal::supported_properties);
        if (ov::util::contains(supported_internal_properties, ov::internal::caching_properties)) {
            auto caching_properties = get_core()->get_property(device_name, ov::internal::caching_properties);
            for (const auto& property_name : caching_properties) {
                properties[property_name] = get_core()->get_property(device_name, std::string(property_name), {});
            }
        } else if (ov::util::contains(supported_properties, ov::device::architecture)) {
            // If caching properties are not supported by device, try to add at least device architecture
            auto device_architecture = get_core()->get_property(device_name, ov::device::architecture);
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
        } else {
            // Device architecture is not supported, add device name w/o id as achitecture
            ov::DeviceIDParser parser(device_name);
            properties = ov::AnyMap{{ov::device::architecture.name(), parser.get_device_name()}};
        }
        result.emplace_back(properties);
    }
    return ov::Any(result);
}

ov::SoPtr<ov::IRemoteContext> ov::hetero::Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::SoPtr<ov::IRemoteContext> ov::hetero::Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}
