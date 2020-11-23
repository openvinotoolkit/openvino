// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>

#include <ie_metric_helpers.hpp>
#include <legacy/ie_util_internal.hpp>
#include <multi-device/multi_device_config.hpp>
#include "multi_device_plugin.hpp"

// ------------------------------MultiDeviceInferencePlugin----------------------------
namespace MultiDevicePlugin {
    using namespace InferenceEngine;
namespace {
    std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                    const std::map<std::string, std::string> & local) {
        for (auto && kvp : local) {
            config[kvp.first] = kvp.second;
        }
        return config;
    }
}  // namespace

std::map<std::string, std::string> MultiDeviceInferencePlugin::GetSupportedConfig(
    const std::map<std::string, std::string> & config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

std::vector<DeviceInformation> MultiDeviceInferencePlugin::ParseMetaDevices(const std::string& priorities,
                                                                          const std::map<std::string, std::string> & config) const {
    std::vector<DeviceInformation> metaDevices;

    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = priorities.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(priorities.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // last token in the string (which has no comma after that)
    devicesWithRequests.push_back(priorities.substr(i, priorities.length() - i));

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    for (auto && d : devicesWithRequests) {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto deviceName = d.substr(0, openingBracket);

        int numRequests = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            numRequests = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (numRequests <= 0) {
                THROW_IE_EXCEPTION << "Priority value for '" << deviceName << "' must be > 0, while " << numRequests
                    << "is passed";
            }
        }

        // create meta device
        auto cfg = getDeviceConfig(deviceName);
        std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        if (std::find(std::begin(supportedConfigKeys), std::end(supportedConfigKeys), CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN))
            != std::end(supportedConfigKeys)) {
            cfg.emplace(CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN), "");
        }
        metaDevices.push_back({ deviceName, cfg, numRequests });
    }

    return metaDevices;
}

Parameter MultiDeviceInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, Parameter> & options) const {
    if (name == MULTI_CONFIG_KEY(DEVICE_PRIORITIES)) {
        auto it = _config.find(MULTI_CONFIG_KEY(DEVICE_PRIORITIES));
        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "Value for KEY_MULTI_DEVICE_PRIORITIES is not set";
        } else {
            return { it->second };
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    for (auto && kvp : config) {
        _config[kvp.first] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MultiDevicePlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MultiDeviceInferencePlugin, version)

MultiDeviceInferencePlugin::MultiDeviceInferencePlugin() {
    _pluginName = "MULTI";
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = { "MULTI" };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys = {
            MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES,
            CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN)};
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else {
        THROW_IE_EXCEPTION << "Unsupported metric key " << name;
    }
}

ExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadExeNetworkImpl(const ICNNNetwork &network,
                                                                              const std::map<std::string, std::string>& config) {
    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with MULTI device via InferencEngine::Core object";
    }

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        THROW_IE_EXCEPTION << "KEY_MULTI_DEVICE_PRIORITIES key is not set for MULTI device";
    }

    auto metaDevices = ParseMetaDevices(priorities->second, fullConfig);

    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> multiNetworkConfig;
    multiNetworkConfig.insert(*priorities);

    DeviceMap<ExecutableNetwork> executableNetworkPerDevice;
    for (auto& p : metaDevices) {
        auto & deviceName = p.deviceName;
        auto & deviceConfig = p.config;
        auto clonedNetwork = cloneNetwork(network);
        executableNetworkPerDevice.insert({ deviceName, GetCore()->LoadNetwork(CNNNetwork{clonedNetwork}, deviceName, deviceConfig) });
        multiNetworkConfig.insert(deviceConfig.begin(), deviceConfig.end());
    }
    if (executableNetworkPerDevice.empty())
        THROW_IE_EXCEPTION << NOT_FOUND_str << "Failed to load Executable network to any device "
                                            <<  "that the MULTI device is initialized to work with";

    auto perfConfig = fullConfig.find(PluginConfigParams::KEY_PERF_COUNT);
    bool enablePerfCounters = (fullConfig.end() != perfConfig) && (perfConfig->second == PluginConfigParams::YES);

    return std::make_shared<MultiDeviceExecutableNetwork>(executableNetworkPerDevice,
                                                          metaDevices,
                                                          multiNetworkConfig,
                                                          enablePerfCounters);
}

QueryNetworkResult MultiDeviceInferencePlugin::QueryNetwork(const ICNNNetwork&                        network,
                                                            const std::map<std::string, std::string>& config) const {
    QueryNetworkResult queryResult;

    if (GetCore() == nullptr) {
        THROW_IE_EXCEPTION << "Please, work with MULTI device via InferencEngine::Core object";
    }

    queryResult.rc = StatusCode::OK;
    queryResult.supportedLayersMap.clear();

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        THROW_IE_EXCEPTION << "KEY_MULTI_DEVICE_PRIORITIES key is not set for MULTI device";
    }

    auto metaDevices = ParseMetaDevices(priorities->second, fullConfig);
    std::unordered_set<std::string> supportedLayers;

    auto allSupportsNgraph =
        std::all_of(std::begin(metaDevices), std::end(metaDevices),
            [&] (const DeviceInformation& value) -> bool {
                auto clonedNetwork = cloneNetwork(network);
                try { GetCore()->QueryNetwork(*clonedNetwork, value.deviceName, value.config); }
                catch (const InferenceEngine::details::InferenceEngineException & ex) {
                    std::string message = ex.what();
                    return message.find(NOT_IMPLEMENTED_str) == std::string::npos;
                }
                return true;
            });

    for (auto&& value : metaDevices) {
        auto queryNetwork = [&] (const InferenceEngine::ICNNNetwork & networkObject) {
            auto clonedNetwork = cloneNetwork(networkObject);
            auto deviceQr = GetCore()->QueryNetwork(*clonedNetwork, value.deviceName, value.config);
            std::unordered_set<std::string> deviceSupportedLayers;
            for (auto&& layerQr : deviceQr.supportedLayersMap) {
                deviceSupportedLayers.emplace(layerQr.first);
            }
            supportedLayers = supportedLayers.empty()
                            ? deviceSupportedLayers : (deviceSupportedLayers.empty()
                            ? supportedLayers : Intersection(supportedLayers, deviceSupportedLayers));
        };

        if (network.getFunction()) {
            if (!allSupportsNgraph) {
                if (contains(fullConfig, CONFIG_KEY_INTERNAL(AGGREGATED_PLUGIN))) {
                    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
                } else {
                    auto cnnNetworkImpl = std::make_shared<details::CNNNetworkImpl>(network);
                    if (cnnNetworkImpl == nullptr)
                        THROW_IE_EXCEPTION << "Cannot create CNNNetworkImpl shared_ptr";
                    queryNetwork(*cnnNetworkImpl);
                }
            } else {
                queryNetwork(network);
            }
        } else {
            queryNetwork(network);
        }
    }

    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }

    return queryResult;
}

}  // namespace MultiDevicePlugin
