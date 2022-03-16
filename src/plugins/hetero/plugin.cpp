// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include "ie_metric_helpers.hpp"
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
#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <openvino/runtime/properties.hpp>
// clang-format on

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;

Engine::Engine() {
    _pluginName = "HETERO";
    _properties.set_name(_pluginName)
        .add(KEY_EXCLUSIVE_ASYNC_REQUESTS, true)
        .add(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT), false)
        .add(ov::device::priorities)
        .add(
            "TARGET_FALLBACK",
            [this] {
                return _properties.get(ov::device::priorities);
            },
            [this](const std::string& str) {
                _properties.set(ov::device::priorities(str));
            })
        .add(ov::device::full_name, "HETERO")
        .add(ov::device::capabilities, {ov::device::capability::EXPORT_IMPORT})
        .add(ov::device::architecture, [this] (const ov::AnyMap& options) {
            auto deviceIt = options.find("TARGET_FALLBACK");
            std::string targetFallback;
            if (deviceIt != options.end()) {
                targetFallback = deviceIt->second.as<std::string>();
            } else {
                deviceIt = options.find(ov::device::priorities.name());
                if (deviceIt != options.end()) {
                    targetFallback = deviceIt->second.as<std::string>();
                } else {
                    targetFallback = _properties.get(ov::device::priorities);
                }
            }
            auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(targetFallback);
            std::string resArch;
            for (const auto& device : fallbackDevices) {
                InferenceEngine::DeviceIDParser parser(device);
                auto supportedMetricKeys = GetCore()->get_property(device, ov::supported_properties);
                auto it = std::find(supportedMetricKeys.begin(), supportedMetricKeys.end(), ov::device::architecture);
                auto arch = (it != supportedMetricKeys.end())
                                ? GetCore()->get_property(device, ov::device::architecture)
                                : parser.getDeviceName();
                resArch += " " + arch;
            }
            return decltype(ov::device::architecture)::value_type{resArch};
        });
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const InferenceEngine::CNNNetwork& network,
                                                                            const Configs& config) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }
    auto all_properties = _properties.merge(config);
    auto device_priorities = all_properties.at("TARGET_FALLBACK");
    if (device_priorities.empty()) {
        device_priorities = all_properties.at(ov::device::priorities.name());
    }
    if (device_priorities.empty()) {
        IE_THROW() << "The '" << ov::device::priorities.name() << "' option was not defined for heterogeneous plugin";
    }
    DeviceMetaInformationMap metaDevices = GetDevicePlugins(device_priorities, all_properties);

    auto function = network.getFunction();
    if (function == nullptr) {
        IE_THROW() << "HETERO device supports just ngraph network representation";
    }

    return std::make_shared<HeteroExecutableNetwork>(network, all_properties, this);
}

InferenceEngine::IExecutableNetworkInternal::Ptr Engine::ImportNetwork(
    std::istream& heteroModel,
    const std::map<std::string, std::string>& config) {
    return std::make_shared<HeteroExecutableNetwork>(heteroModel, _properties.merge(config), this);
}

Engine::DeviceMetaInformationMap Engine::GetDevicePlugins(const std::string& targetFallback,
                                                          const Configs& localConfig) const {
    auto getDeviceConfig = [&](const std::string& deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        auto tconfig = _properties.merge(localConfig);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetCore()->GetSupportedConfig(deviceName, tconfig);
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

QueryNetworkResult Engine::QueryNetwork(const CNNNetwork& network, const Configs& config) const {
    QueryNetworkResult qr;

    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with HETERO device via InferencEngine::Core object";
    }

    auto all_properties = _properties.merge(config);
    auto device_priorities = all_properties.at("TARGET_FALLBACK");
    if (device_priorities.empty()) {
        device_priorities = all_properties.at(ov::device::priorities.name());
    }
    if (device_priorities.empty()) {
        IE_THROW() << "The '" << ov::device::priorities.name() << "' option was not defined for heterogeneous plugin";
    }

    DeviceMetaInformationMap metaDevices = GetDevicePlugins(device_priorities, all_properties);

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
    auto fallbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(device_priorities);

    for (auto&& deviceName : fallbackDevices) {
        for (auto&& layerQueryResult : queryResults[deviceName].supportedLayersMap) {
            qr.supportedLayersMap.emplace(layerQueryResult);
        }
    }

    // set OK status
    qr.rc = StatusCode::OK;

    return qr;
}

static Version heteroPluginDescription = {
    {2, 1},  // plugin API version
    CI_BUILD_NUMBER,
    "heteroPlugin"  // plugin description message
};

IE_DEFINE_PLUGIN_CREATE_FUNCTION(Engine, heteroPluginDescription)
