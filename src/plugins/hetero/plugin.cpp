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
#include "openvino/runtime/properties.hpp"
// clang-format on

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;

Engine::Engine() {
    _pluginName = "HETERO";
    _config[KEY_EXCLUSIVE_ASYNC_REQUESTS] = YES;
    _config[HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)] = NO;
}

namespace {

Engine::Configs mergeConfigs(Engine::Configs config, const Engine::Configs& local) {
    for (auto&& kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

Engine::Configs mergeConfigs(Engine::Configs config, const ov::AnyMap& local) {
    for (auto&& kvp : local) {
        config[kvp.first] = kvp.second.as<std::string>();
    }
    return config;
}

const std::vector<std::string>& getSupportedConfigKeys() {
    static const std::vector<std::string> supported_configKeys = {HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
                                                                  "TARGET_FALLBACK",
                                                                  ov::device::priorities.name(),
                                                                  CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)};

    return supported_configKeys;
}

}  // namespace

std::string Engine::GetTargetFallback(const Engine::Configs& config, bool raise_exception) const {
    auto it = config.find("TARGET_FALLBACK");
    if (it == config.end()) {
        it = config.find(ov::device::priorities.name());
    }
    if (it == config.end()) {
        if (raise_exception)
            IE_THROW() << "The '" << ov::device::priorities.name()
                       << "' option was not defined for heterogeneous plugin";
        return std::string("");
    }
    return it->second;
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                            const Configs& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }
    auto tconfig = mergeConfigs(_config, config);
    std::string fallbackDevicesStr = GetTargetFallback(tconfig);
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, tconfig);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO device supports just ngraph network representation";
    }

    return std::make_shared<HeteroExecutableNetwork>(network, mergeConfigs(_config, config), this);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(
    std::istream& heteroModel,
    const std::map<std::string, std::string>& config) {
    return std::make_shared<HeteroExecutableNetwork>(heteroModel, mergeConfigs(_config, config), this);
}

Engine::DeviceMetaInformationMap Engine::GetDevicePlugins(const std::string& targetFallback,
                                                          const Configs& localConfig) const {
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    Engine::DeviceMetaInformationMap metaDevices;
    for (auto&& deviceName : fallbackDevices) {
        auto itPlugin = metaDevices.find(deviceName);
        if (metaDevices.end() == itPlugin) {
            metaDevices[deviceName] = GetCore()->GetSupportedConfig(deviceName, mergeConfigs(_config, localConfig));
        }
    }
    return metaDevices;
}

void Engine::SetConfig(const Configs& configs) {
    for (auto&& kvp : configs) {
        const auto& name = kvp.first;
        const auto& supported_configKeys = getSupportedConfigKeys();
        if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name))
            _config[name] = kvp.second;
        else
            IE_THROW() << "Unsupported config key: " << name;
    }
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const Configs& config) const {
    QueryNetworkResult qr;

    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }

    auto tconfig = mergeConfigs(_config, config);
    std::string fallbackDevicesStr = GetTargetFallback(tconfig);
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, tconfig);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO device supports just ngraph network representation";
    }

    std::map<std::string, QueryNetworkResult> queryResults;
    for (auto&& metaDevice : metaDevices) {
        auto& deviceName = metaDevice.first;
        queryResults[deviceName] = GetCore()->QueryNetwork(network, deviceName, metaDevice.second);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(fallbackDevicesStr);

    for (auto&& deviceName : fallbackDevices) {
        for (auto&& layerQueryResult : queryResults[deviceName].supportedLayersMap) {
            qr.supportedLayersMap.emplace(layerQueryResult);
        }
    }

    // set OK status
    qr.rc = StatusCode::OK;

    return qr;
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter>& options) const {
    if (ov::supported_properties == name) {
        return decltype(ov::supported_properties)::value_type{
            ov::PropertyName{ov::supported_properties.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::full_name.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::architecture.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::capabilities.name(), ov::PropertyMutability::RO},
            ov::PropertyName{ov::device::priorities.name(), ov::PropertyMutability::RW}};
    } else if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS,
                             std::vector<std::string>{METRIC_KEY(SUPPORTED_METRICS),
                                                      ov::device::full_name.name(),
                                                      METRIC_KEY(SUPPORTED_CONFIG_KEYS),
                                                      ov::device::architecture.name(),
                                                      METRIC_KEY(IMPORT_EXPORT_SUPPORT),
                                                      ov::device::capabilities.name()});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, getSupportedConfigKeys());
    } else if (ov::device::full_name == name) {
        return decltype(ov::device::full_name)::value_type{"HETERO"};
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (ov::device::capabilities == name) {
        return decltype(ov::device::capabilities)::value_type{{ov::device::capability::EXPORT_IMPORT}};
    } else if (ov::device::architecture == name) {
        auto tconfig = mergeConfigs(_config, options);
        std::string targetFallback = GetTargetFallback(tconfig);
        return decltype(ov::device::architecture)::value_type{DeviceArchitecture(targetFallback)};
    } else {
        IE_THROW() << "Unsupported metric key: " << name;
    }
}
std::string Engine::DeviceArchitecture(const std::string& targetFallback) const {
    auto fallbackDevices = ov::DeviceIDParser::get_hetero_devices(targetFallback);
    std::string resArch;
    for (const auto& device : fallbackDevices) {
        ov::DeviceIDParser parser(device);

        auto supportedMetricKeys = GetCore()
                                       ->GetMetric(parser.get_device_name(), METRIC_KEY(SUPPORTED_METRICS))
                                       .as<std::vector<std::string>>();
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), METRIC_KEY(DEVICE_ARCHITECTURE));
        auto arch = (it != supportedMetricKeys.end())
                        ? GetCore()->GetMetric(device, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>()
                        : parser.get_device_name();
        resArch += " " + arch;
    }
    return resArch;
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return {dump};
    } else if (name == ov::device::priorities) {
        std::string targetFallback = GetTargetFallback(_config);
        auto priorities = ov::util::from_string(targetFallback, ov::device::priorities);
        return decltype(ov::device::priorities)::value_type{priorities};
    } else if (name == "TARGET_FALLBACK") {
        return GetTargetFallback(_config);
    } else {
        IE_THROW() << "Unsupported config key: " << name;
    }
}

static Version heteroPluginDescription = {
    {2, 1},  // plugin API version
    CI_BUILD_NUMBER,
    "heteroPlugin"  // plugin description message
};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, heteroPluginDescription)
