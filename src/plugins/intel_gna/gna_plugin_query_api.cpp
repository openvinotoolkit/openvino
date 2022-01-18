// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_parameter.hpp>
#include "gna_plugin.hpp"
#include "gna/gna_config.hpp"

#include <string>
#include <map>
#include <vector>
#include <unordered_map>
#include <memory>

using namespace GNAPluginNS;
using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;

Parameter GNAPlugin::GetConfig(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    return config.GetParameter(name);
}

Parameter GNAPlugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const std::unordered_map<std::string, std::function<Parameter()>> queryApiSupported = {
        {METRIC_KEY(AVAILABLE_DEVICES), [this]() {return GetAvailableDevices();}},
        {METRIC_KEY(SUPPORTED_CONFIG_KEYS), [this]() {return config.GetSupportedKeys();}},
        {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS), [this]() {
            uint32_t nireq = 1;
            return nireq;
        }},
        {METRIC_KEY(FULL_DEVICE_NAME), [&options, this]() {
            auto availableDevices = GetAvailableDevices().as<std::vector<std::string>>();

            if (availableDevices.empty()) {
                THROW_GNA_EXCEPTION << "No devices available.";
            }

            if (!options.count(KEY_DEVICE_ID)) {
                if (availableDevices.size() == 1 || availableDevices.size() == 2) {
                    return availableDevices.back(); // detection order is GNA_SW_EXACT, GNA_HW
                } else {
                    THROW_GNA_EXCEPTION << "KEY_DEVICE_ID not set in request for FULL_DEVICE_NAME";
                }
            }

            auto deviceName = options.at(KEY_DEVICE_ID).as<std::string>();
            return deviceName;
        }},
        {METRIC_KEY(GNA_LIBRARY_FULL_VERSION), [this]() {return GNADeviceHelper::GetGnaLibraryVersion();}},
        {METRIC_KEY(SUPPORTED_METRICS), [&queryApiSupported, this]() {
            std::vector<std::string> availablesMetrics;
            for (auto && supportedAPI : queryApiSupported) {
                availablesMetrics.push_back(supportedAPI.first);
            }
            return availablesMetrics;
        }},
        {METRIC_KEY(IMPORT_EXPORT_SUPPORT), []() {return true;}}
    };

    auto it = queryApiSupported.find(name);
    if (it == queryApiSupported.end()) {
        THROW_GNA_EXCEPTION << "Unsupported parameters for GetMetric: " << name;
    }

    return it->second();
}

Parameter GNAPlugin::GetAvailableDevices() const {
    std::vector<std::string> devices;

    try {
        GNADeviceHelper helper;
        devices.push_back("GNA_SW_EXACT");
        if (helper.hasGnaHw()) {
            devices.push_back("GNA_HW");
        }
    }catch(...) {}

    return devices;
}
