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
#include "internal_properties.hpp"
#include "openvino/util/common_util.hpp"
// clang-format on

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;

namespace {

const std::vector<std::string>& getHeteroSupportedConfigKeys() {
    static const std::vector<std::string> supported_configKeys = {HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
                                                                  "TARGET_FALLBACK",
                                                                  ov::device::priorities.name()};

    return supported_configKeys;
}

const std::vector<std::string>& getHeteroDeviceSupportedConfigKeys() {
    static const std::vector<std::string> supported_configKeys = {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)};
    return supported_configKeys;
}

std::vector<std::string> getSupportedConfigKeys() {
    std::vector<std::string> supported_configKeys = getHeteroSupportedConfigKeys();
    for (auto&& key : getHeteroDeviceSupportedConfigKeys())
        supported_configKeys.emplace_back(key);
    return supported_configKeys;
}

ov::AnyMap any_copy(const Configs& params) {
    ov::AnyMap result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second);
    }
    return result;
}

Configs any_copy(const ov::AnyMap& params) {
    Configs result;
    for (auto&& value : params) {
        result.emplace(value.first, value.second.as<std::string>());
    }
    return result;
}

ov::AnyMap clone_map(const ov::AnyMap& m) {
    ov::AnyMap rm;
    for (auto&& kvp : m) {
        rm[kvp.first] = kvp.second.is<ov::AnyMap>() ? ov::Any(clone_map(kvp.second.as<ov::AnyMap>())) : kvp.second;
    }

    return rm;
}

}  // namespace

Engine::Engine() {
    _pluginName = "HETERO";
    _config[HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)] = NO;
    _device_config[CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)] = YES;
}

ParsedConfig<ov::AnyMap> Engine::MergeConfigs(const ov::AnyMap& user_config) const {
    auto device_config = clone_map(user_config);
    auto hetero_config = _config;

    // after API 1.0 removal, replace with the loop over getHeteroSupportedConfigKeys()
    {
        auto try_merge_property = [&](const std::string& property_name) -> bool {
            auto property_it = device_config.find(property_name);
            if (property_it != device_config.end()) {
                // migrate HETERO property to hetero_config
                hetero_config[property_it->first] = property_it->second.as<std::string>();
                // and erase it from device_config
                device_config.erase(property_it->first);
                return true;
            }

            return false;
        };

        try_merge_property(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));

        // if we have not found TARGET_FALLBACK in user_config, let's try to find device::priorities
        // Note: we can have conflicts here like
        //   core.set_property(HETERO, TARGET_FALLBACK=MULTI,CPU)
        //   core.compile_model(HETERO, DEVICE_PRIORITIES=GPU.0,GPU.1)
        // so, we need to check whether TARGET_FALLBACK was set before in set_property
        // This check can be removed after API 1.0 is removed
        if (!try_merge_property("TARGET_FALLBACK") && hetero_config.find("TARGET_FALLBACK") == hetero_config.end()) {
            try_merge_property(ov::device::priorities.name());
        }
    }

    // merge device_config settings
    for (auto&& key : getHeteroDeviceSupportedConfigKeys()) {
        auto user_config_it = user_config.find(key);
        if (user_config_it != user_config.end()) {
            device_config[user_config_it->first] = user_config_it->second;
        }
    }

    return {hetero_config, device_config};
}

ParsedConfig<Configs> Engine::MergeConfigs(const Configs& user_config) const {
    auto parsed_config = MergeConfigs(any_copy(user_config));
    return {parsed_config.hetero_config, any_copy(parsed_config.device_config)};
}

std::string Engine::GetTargetFallback(const Configs& user_config, bool raise_exception) const {
    return GetTargetFallback(any_copy(user_config), raise_exception);
}

std::string Engine::GetTargetFallback(const ov::AnyMap& user_config, bool raise_exception) const {
    auto hetero_config = MergeConfigs(user_config).hetero_config;
    auto it = hetero_config.find("TARGET_FALLBACK");
    if (it == hetero_config.end()) {
        it = hetero_config.find(ov::device::priorities.name());
    }
    if (it == hetero_config.end()) {
        if (raise_exception)
            IE_THROW() << "The '" << ov::device::priorities.name()
                       << "' option was not defined for heterogeneous plugin";
        return std::string("");
    }
    return it->second;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                            const Configs& user_config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << "HETERO device supports only nGraph model representation";
    }

    return std::make_shared<HeteroExecutableNetwork>(network, user_config, this);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(
    std::istream& heteroModel,
    const std::map<std::string, std::string>& user_config) {
    return std::make_shared<HeteroExecutableNetwork>(heteroModel, user_config, this, true);
}

Engine::DeviceMetaInformationMap Engine::GetDevicePlugins(const std::string& targetFallback,
                                                          const Configs& device_config) const {
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    Engine::DeviceMetaInformationMap metaDevices;
    for (auto&& deviceName : fallbackDevices) {
        auto itPlugin = metaDevices.find(deviceName);
        if (metaDevices.end() == itPlugin) {
            metaDevices[deviceName] = GetCore()->GetSupportedConfig(deviceName, device_config);
        }
    }
    return metaDevices;
}

void Engine::SetConfig(const Configs& user_config) {
    for (auto&& kvp : user_config) {
        const auto& name = kvp.first;
        if (ov::util::contains(getHeteroSupportedConfigKeys(), name))
            _config[name] = kvp.second;
        else if (ov::util::contains(getHeteroDeviceSupportedConfigKeys(), name))
            _device_config[name] = kvp.second;
        else
            IE_THROW() << "Unsupported HETERO config key: " << name;
    }
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const Configs& user_config) const {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via ov::Core object";
    }

    auto parsed_config = MergeConfigs(user_config);
    std::string fallbackDevicesStr = GetTargetFallback(parsed_config.hetero_config);
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, parsed_config.device_config);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO device supports just nGraph model representation";
    }

    std::map<std::string, QueryNetworkResult> queryResults;
    for (auto&& metaDevice : metaDevices) {
        const auto& deviceName = metaDevice.first;
        const auto& device_config = metaDevice.second;
        queryResults[deviceName] = GetCore()->QueryNetwork(network, deviceName, device_config);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(fallbackDevicesStr);

    QueryNetworkResult qr;
    for (auto&& deviceName : fallbackDevices) {
        for (auto&& layerQueryResult : queryResults[deviceName].supportedLayersMap) {
            qr.supportedLayersMap.emplace(layerQueryResult);
        }
    }

    // set OK status
    qr.rc = StatusCode::OK;

    return qr;
}

Parameter Engine::GetMetric(const std::string& name, const ov::AnyMap& user_options) const {
    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::caching_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RW}};
    } else if (ov::caching_properties == name) {
        return decltype(ov::caching_properties)::value_type{ov::hetero::caching_device_properties.name()};
    } else if (ov::hetero::caching_device_properties == name) {
        std::string targetFallback = GetTargetFallback(user_options);
        return decltype(ov::hetero::caching_device_properties)::value_type{DeviceCachingProperties(targetFallback)};
    } else if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{METRIC_KEY(SUPPORTED_METRICS),
                                                      ov::device::full_name.name(),
                                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                      METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                      ov::caching_properties.name(),
                                                      ov::device::capabilities.name()});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, getSupportedConfigKeys());
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"HETERO"};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else {
        IE_THROW() << "Unsupported HETERO metric key: " << name;
    }
}

std::string Engine::DeviceCachingProperties(const std::string& targetFallback) const {
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    // Vector of caching configs for devices
    std::vector<ov::AnyMap> result = {};
    for (const auto& device : fallbackDevices) {
        ov::DeviceIDParser parser(device);
        ov::AnyMap properties = {};
        // Use name without id
        auto device_name = parser.get_device_name();
        auto supported_properties =
            GetCore()->GetMetric(device, ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
        if (ov::util::contains(supported_properties, ov::caching_properties.name())) {
            auto caching_properties =
                GetCore()->GetMetric(device, ov::caching_properties.name()).as<std::vector<ov::PropertyName>>();
            for (auto& property_name : caching_properties) {
                properties[property_name] = GetCore()->GetMetric(device, property_name);
            }
            // If caching properties are not supported by device, try to add at least device architecture
        } else if (ov::util::contains(supported_properties, ov::device::architecture.name())) {
            auto device_architecture = GetCore()->GetMetric(device, ov::device::architecture.name());
            properties = ov::AnyMap{{ov::device::architecture.name(), device_architecture}};
            // Device architecture is not supported, add device name as achitecture
        } else {
            properties = ov::AnyMap{{ov::device::architecture.name(), device_name}};
        }
        result.emplace_back(properties);
    }
    return result.empty() ? "" : ov::Any(result).as<std::string>();
}

Parameter Engine::GetConfig(const std::string& name, const ov::AnyMap& options) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(name);
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return {dump};
    } else if (name == ov::device::priorities) {
        std::string targetFallback = GetTargetFallback(options);
        auto priorities = ov::util::from_string(targetFallback, ov::device::priorities);
        return decltype(ov::device::priorities)::value_type{priorities};
    } else if (name == "TARGET_FALLBACK") {
        return GetTargetFallback(options);
    } else if (name == CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)) {
        auto it = _device_config.find(name);
        IE_ASSERT(it != _device_config.end());
        bool exclusive_async = it->second == YES;
        return {exclusive_async};
    } else {
        IE_THROW() << "Unsupported HETERO config key: " << name;
    }
}

static Version heteroPluginDescription = {
    {2, 1},  // plugin API version
    CI_BUILD_NUMBER,
    "heteroPlugin"  // plugin description message
};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, heteroPluginDescription)
