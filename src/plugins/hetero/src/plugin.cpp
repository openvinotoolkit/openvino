// Copyright (C) 2018-2023 Intel Corporation
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
#include "ie/ie_plugin_config.hpp"
#include "itt.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "openvino/runtime/internal_properties.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/util/common_util.hpp"
#include "properties.hpp"

ov::hetero::Plugin::Plugin() {
    set_device_name("HETERO");
}

std::shared_ptr<ov::ICompiledModel> ov::hetero::Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                                      const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::Hetero, "Plugin::compile_model");

    auto config = Configuration{properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(model->clone(), shared_from_this(), config);
    return compiled_model;
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

    auto config = Configuration{properties, m_cfg};
    auto compiled_model = std::make_shared<CompiledModel>(model, shared_from_this(), config);
    return compiled_model;
}

ov::hetero::Plugin::DeviceProperties ov::hetero::Plugin::get_properties_per_device(const std::string& device_priorities,
                                                                                   const ov::AnyMap& properties) const {
    auto device_names = ov::DeviceIDParser::get_hetero_devices(device_priorities);
    DeviceProperties device_properties;
    for (const auto& device_name : device_names) {
        auto properties_it = device_properties.find(device_name);
        if (device_properties.end() == properties_it)
            device_properties[device_name] = get_core()->get_supported_property(device_name, properties);
    }
    return device_properties;
}

std::pair<ov::SupportedOpsMap, ov::hetero::SubgraphsMappingInfo> ov::hetero::Plugin::query_model_update(
    std::shared_ptr<ov::Model>& model,
    const ov::AnyMap& properties,
    bool allow_exception) const {
    Configuration full_config{properties, m_cfg};
    DeviceProperties properties_per_device =
        get_properties_per_device(full_config.device_priorities, full_config.get_device_properties());

    //  WARNING: Here is devices with user set priority
    auto device_names = ov::DeviceIDParser::get_hetero_devices(full_config.device_priorities);

    auto update_supported_ops = [](ov::SupportedOpsMap& final_results, const ov::SupportedOpsMap& device_results) {
        for (const auto& layer_query_result : device_results)
            final_results.emplace(layer_query_result);
    };

    ov::SupportedOpsMap supported_ops_temp;
    ov::SupportedOpsMap supported_ops_final;
    std::map<std::string, ov::SupportedOpsMap> query_results;
    ov::hetero::SubgraphsMappingInfo mapping_info;
    for (const auto& device_name : device_names) {
        // If there are some unsupported operations and it is a last device
        // exception should be raised when allowed
        const auto& default_device = (!allow_exception || device_name != device_names.back()) ? get_device_name() : "";
        const auto& device_config = properties_per_device.at(device_name);
        query_results[device_name] = get_core()->query_model(model, device_name, device_config);
        // Update supported operations map which includes new operations
        update_supported_ops(supported_ops_temp, query_results[device_name]);
        // Update supported operations map which includes original operations only
        update_supported_ops(supported_ops_final, query_results[device_name]);
        mapping_info =
            ov::hetero::mask_model_subgraphs_by_ops(model, supported_ops_temp, m_cfg.dump_dot_files(), default_device);
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
    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };

    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };

    Configuration full_config{properties, m_cfg};
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();

        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        add_ro_properties(METRIC_KEY(IMPORT_EXPORT_SUPPORT), metrics);
        return to_string_vector(metrics);
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return to_string_vector(full_config.get_supported());
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::internal::supported_properties == name) {
        return decltype(ov::internal::supported_properties)::value_type{
            ov::PropertyName{ov::internal::caching_properties.name(), ov::PropertyMutability::RO}};
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{get_device_name()};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        return true;
    } else if (ov::internal::caching_properties == name) {
        return decltype(ov::internal::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        return caching_device_properties(full_config.device_priorities);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return full_config.get(name);
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
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
