// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_plugin.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>
#include "ie_plugin_config.hpp"
#include "hetero_executable_network.hpp"
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>

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

Engine::Configs mergeConfigs(Engine::Configs config, const Engine::Configs & local) {
    for (auto && kvp : local) {
        config[kvp.first] = kvp.second;
    }
    return config;
}

}  // namespace

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork&    network,
                                                                           const Configs&                   config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }
    auto tconfig = mergeConfigs(_config, config);
    auto it = tconfig.find("TARGET_FALLBACK");
    if (it == tconfig.end()) {
        IE_THROW() << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
    }
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(it->second, tconfig);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO plugin supports just ngraph network representation";
    }

    return std::make_shared<HeteroExecutableNetwork>(network, mergeConfigs(_config, config), this);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(std::istream& heteroModel, const std::map<std::string, std::string>& config) {
    return std::make_shared<HeteroExecutableNetwork>(heteroModel, mergeConfigs(_config, config), this);
}

Engine::Configs Engine::GetSupportedConfig(const Engine::Configs& config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    Engine::Configs supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

Engine::DeviceMetaInformationMap Engine::GetDevicePlugins(const std::string& targetFallback,
                                                          const Configs & localConfig) const {
    auto getDeviceConfig = [&](const std::string & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        Configs tconfig = mergeConfigs(_config, localConfig);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(targetFallback);
    Engine::DeviceMetaInformationMap metaDevices;
    for (auto&& deviceName : fallbackDevices) {
        auto itPlugin = metaDevices.find(deviceName);
        if (metaDevices.end() == itPlugin) {
            metaDevices[deviceName] = getDeviceConfig(deviceName);
        }
    }
    return metaDevices;
}

void Engine::SetConfig(const Configs &configs) {
    for (auto&& config : configs) {
        _config[config.first] = config.second;
    }
}

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork &network, const Configs& config) const {
    QueryNetworkResult qr;

    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }

    auto tconfig = mergeConfigs(_config, config);
    auto it = tconfig.find("TARGET_FALLBACK");
    if (it == tconfig.end()) {
        IE_THROW() << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
    }

    std::string fallbackDevicesStr = it->second;
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(fallbackDevicesStr, tconfig);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO plugin supports just ngraph network representation";
    }

    std::map<std::string, QueryNetworkResult> queryResults;
    for (auto&& metaDevice : metaDevices) {
        auto& deviceName = metaDevice.first;
        queryResults[deviceName] = GetCore()->QueryNetwork(network, deviceName, metaDevice.second);
    }

    //  WARNING: Here is devices with user set priority
    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(fallbackDevicesStr);

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
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS),
            METRIC_KEY(DEVICE_ARCHITECTURE),
            METRIC_KEY(IMPORT_EXPORT_SUPPORT)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            "TARGET_FALLBACK",
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)});
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, std::string{"HETERO"});
    } else if (METRIC_KEY(IMPORT_EXPORT_SUPPORT) == name) {
        IE_SET_METRIC_RETURN(IMPORT_EXPORT_SUPPORT, true);
    } else if (METRIC_KEY(DEVICE_ARCHITECTURE) == name) {
        auto deviceIt = options.find("TARGET_FALLBACK");
        std::string targetFallback;
        if (deviceIt != options.end()) {
            targetFallback = deviceIt->second.as<std::string>();
        } else {
            targetFallback = GetConfig("TARGET_FALLBACK", {}).as<std::string>();
        }
        IE_SET_METRIC_RETURN(DEVICE_ARCHITECTURE, DeviceArchitecture(targetFallback));
    } else {
        IE_THROW() << "Unsupported Plugin metric: " << name;
    }
}
std::string Engine::DeviceArchitecture(const std::string& targetFallback) const {
    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(targetFallback);
    std::string resArch;
    for (const auto& device : fallbackDevices) {
        InferenceEngine::DeviceIDParser parser(device);

        std::vector<std::string> supportedMetricKeys = GetCore()->GetMetric(
                parser.getDeviceName(), METRIC_KEY(SUPPORTED_METRICS));
        auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(),
                            METRIC_KEY(DEVICE_ARCHITECTURE));
        auto arch = (it != supportedMetricKeys.end()) ?
                GetCore()->GetMetric(device, METRIC_KEY(DEVICE_ARCHITECTURE)).as<std::string>() : parser.getDeviceName();
        resArch += " " + arch;
    }
    return resArch;
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return { dump };
    } else if (name == "TARGET_FALLBACK") {
        auto it = _config.find("TARGET_FALLBACK");
        if (it == _config.end()) {
            IE_THROW() << "Value for TARGET_FALLBACK is not set";
        } else {
            return { it->second };
        }
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
