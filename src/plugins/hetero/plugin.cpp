// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "ie_metric_helpers.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "plugin.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>
#include "ie_plugin_config.hpp"
#include "executable_network.hpp"
#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/runtime/properties.hpp"
#include "properties.hpp"
#include "openvino/util/common_util.hpp"
// clang-format on

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;


// ! [plugin:ctor]
ov::hetero_plugin::Plugin::Plugin() {
    // TODO: fill with actual device name, backend engine
    set_device_name("HETERO");

    // // create ngraph backend which performs inference using ngraph reference implementations
    // m_backend = ov::runtime::Backend::create();

    // // create default stream executor with a given name
    // m_waitExecutor = get_executor_manager()->get_idle_cpu_streams_executor({wait_executor_name});
}
// ! [plugin:ctor]

// ! [plugin:dtor]
ov::hetero_plugin::Plugin::~Plugin() {
    // Plugin should remove executors from executor cache to avoid threads number growth in the whole application
    // get_executor_manager()->clear(stream_executor_name);
    // get_executor_manager()->clear(wait_executor_name);
}
// ! [plugin:dtor]

// // ! [plugin:compile_model]
// std::shared_ptr<ov::ICompiledModel> ov::hetero_plugin::Plugin::compile_model(
//     const std::shared_ptr<const ov::Model>& model,
//     const ov::AnyMap& properties) const {
//     return compile_model(model, properties, {});
// }
// // ! [plugin:compile_model]

// // ! [plugin:compile_model_with_remote]
// std::shared_ptr<ov::ICompiledModel> ov::hetero_plugin::Plugin::compile_model(
//     const std::shared_ptr<const ov::Model>& model,
//     const ov::AnyMap& properties,
//     const ov::RemoteContext& context) const {
//     OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, "Plugin::compile_model");

//     auto fullConfig = Configuration{properties, m_cfg};
//     // auto compiled_model = std::make_shared<CompiledModel>(
//     //     model->clone(),
//     //     shared_from_this(),
//     //     context,
//     //     fullConfig.exclusive_async_requests
//     //         ? get_executor_manager()->get_executor(template_exclusive_executor)
//     //         : get_executor_manager()->get_idle_cpu_streams_executor(streamsExecutorConfig),
//     //     fullConfig);
    
//     return std::make_shared<HeteroExecutableNetwork>(ov::legacy_convert::convert_model(model, true), ov::any_copy(fullConfig.GetDeviceConfig()), this);
// }
// // ! [plugin:compile_model_with_remote]

InferenceEngine::IExecutableNetworkInternal::Ptr ov::hetero_plugin::Plugin::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                            const ov::hetero_plugin::Configs& user_config) {
    if (get_core() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "HETERO device supports only nGraph model representation";
    }

    return std::make_shared<HeteroExecutableNetwork>(network, user_config, this);
}

InferenceEngine::IExecutableNetworkInternal::Ptr ov::hetero_plugin::Plugin::ImportNetwork(
    std::istream& heteroModel,
    const std::map<std::string, std::string>& user_config) {
    return std::make_shared<HeteroExecutableNetwork>(heteroModel, user_config, this, true);
}

ov::hetero_plugin::Plugin::DeviceMetaInformationMap ov::hetero_plugin::Plugin::GetDevicePlugins(const std::string& targetFallback,
                                                                                                const ov::AnyMap& properties) const {
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    DeviceMetaInformationMap metaDevices;
    for (auto&& deviceName : fallbackDevices) {
        auto itPlugin = metaDevices.find(deviceName);
        if (metaDevices.end() == itPlugin) {
            metaDevices[deviceName] = get_core()->get_supported_property(deviceName, properties);
        }
    }
    return metaDevices;
}

ov::SupportedOpsMap ov::hetero_plugin::Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                                           const ov::AnyMap& properties) const {
    OV_ITT_SCOPED_TASK(itt::domains::HeteroPlugin, "Plugin::query_model");

    Configuration fullConfig{properties, m_cfg, false};
    
    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");

    std::string fallbackDevicesStr = fullConfig.device_priorities;
    // std::string fallbackDevicesStr = GetTargetFallback(parsed_config.hetero_config);
    
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, fullConfig.GetDeviceConfig());
    // DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, parsed_config.device_config);

    std::map<std::string, ov::SupportedOpsMap> queryResults;
    for (auto&& metaDevice : metaDevices) {
        const auto& deviceName = metaDevice.first;
        const auto& device_config = metaDevice.second;
        queryResults[deviceName] = get_core()->query_model(model, deviceName, device_config);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(fallbackDevicesStr);

    ov::SupportedOpsMap res;
    for (auto&& deviceName : fallbackDevices) {
        for (auto&& layerQueryResult : queryResults[deviceName]) {
            res.emplace(layerQueryResult);
        }
    }

    return res;
}

// ! [plugin:set_property]
void ov::hetero_plugin::Plugin::set_property(const ov::AnyMap& properties) {
    m_cfg = Configuration{properties, m_cfg};
}
// ! [plugin:set_property]

ov::Any ov::hetero_plugin::Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };

    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::supported_properties,
                                                    ov::device::full_name,
                                                    ov::device::capabilities,
                                                    ov::caching_properties};
                                                    //ov::available_devices,
                                                    //ov::device::architecture,
                                                    //ov::range_for_async_infer_requests
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
                                                    //ov::device::id,
                                                    //ov::enable_profiling,
                                                    //ov::hint::performance_mode,
                                                    //ov::exclusive_async_requests,
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        auto metrics = default_ro_properties();

        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        add_ro_properties(METRIC_KEY(IMPORT_EXPORT_SUPPORT), metrics);
        return to_string_vector(metrics);
        // IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
        //                     // TODO: check list
        //                      std::vector<std::string>{METRIC_KEY(SUPPORTED_METRICS),
        //                                               ov::device::full_name.name(),
        //                                               METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        //                                               METRIC_KEY(IMPORT_EXPORT_SUPPORT),
        //                                               ov::caching_properties.name(),
        //                                               ov::device::capabilities.name()});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return to_string_vector(default_rw_properties());
        // IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, getSupportedConfigKeys());
    } else if (ov::supported_properties == name) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"HETERO"};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        return true;
    } else if (ov::caching_properties == name) {
        // TODO vurusovs: RECHECK WITH ov::hetero_plugin::caching_device_properties
        return decltype(ov::caching_properties)::value_type{ov::hetero_plugin::caching_device_properties.name()};
    } else if (ov::hetero_plugin::caching_device_properties == name) {
        // std::string targetFallback = GetTargetFallback(user_options);
        // it = hetero_config.find(ov::device::priorities.name());

        // TODO vurusovs: CHECK `target_fallback` is empty or not
        // TODO vurusovs: RECHECK WITH ov::caching_properties
        auto target_fallback = m_cfg.device_priorities;
        return decltype(ov::hetero_plugin::caching_device_properties)::value_type{DeviceCachingProperties(target_fallback)};
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        return m_cfg.Get(name);
    }
}

std::string ov::hetero_plugin::Plugin::DeviceCachingProperties(const std::string& targetFallback) const {
    // TODO: CHECK FUNCTION WORKS CORRECTLY

    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    // Vector of caching configs for devices
    std::vector<ov::AnyMap> result = {};
    for (const auto& device : fallbackDevices) {
        ov::DeviceIDParser parser(device);
        ov::AnyMap properties = {};
        // Use name without id
        auto device_name = parser.get_device_name();
        auto supported_properties =
            get_core()->get_property(device, ov::supported_properties);
        if (ov::util::contains(supported_properties, ov::caching_properties)) {
            auto caching_properties =
                get_core()->get_property(device, ov::caching_properties);
            for (auto& property_name : caching_properties) {
                properties[property_name] = get_core()->get_property(device, std::string(property_name), {});
            }
            // If caching properties are not supported by device, try to add at least device architecture
        } else if (ov::util::contains(supported_properties, ov::device::architecture)) {
            auto device_architecture = get_core()->get_property(device, ov::device::architecture);
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
            // Device architecture is not supported, add device name as achitecture
        } else {
            properties = ov::AnyMap{{ov::device::architecture.name(), device_name}};
        }
        result.emplace_back(properties);
    }
    return result.empty() ? "" : ov::Any(result).as<std::string>();
}


// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "hetero_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::hetero_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]
