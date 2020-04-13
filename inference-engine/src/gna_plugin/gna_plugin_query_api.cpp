// Copyright (C) 2018-2020 Intel Corporation
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

Parameter GNAPlugin::GetConfig(const std::string& name, const std::map<std::string, Parameter> & options) const {
    auto configKeys = supportedConfigKeysWithDefaults();
    auto result = configKeys.find(name);
    if (result == configKeys.end()) {
        THROW_GNA_EXCEPTION << "unsupported config key: " << name;
    }
    return result->second;
}

Parameter GNAPlugin::GetMetric(const std::string& name, const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const std::unordered_map<std::string, std::function<Parameter()>> queryApiSupported = {
        {METRIC_KEY(AVAILABLE_DEVICES), [this]() {return GetAvailableDevices();}},
        {METRIC_KEY(SUPPORTED_CONFIG_KEYS), [this]() {return supportedConfigKeys();}},
        {METRIC_KEY(FULL_DEVICE_NAME), [&options, this]() {
            auto availableDevices = GetAvailableDevices().as<std::vector<std::string>>();

            if (availableDevices.empty()) {
                THROW_GNA_EXCEPTION << "No devices available.";
            }

            if (!options.count(KEY_DEVICE_ID)) {
                if (availableDevices.size() == 1) {
                    return availableDevices[0];
                } else {
                    THROW_GNA_EXCEPTION << "KEY_DEVICE_ID not set in request for FULL_DEVICE_NAME";
                }
            }

            auto deviceName = options.at(KEY_DEVICE_ID).as<std::string>();
            return deviceName;
        }},
        {METRIC_KEY(SUPPORTED_METRICS), [&queryApiSupported, this]() {
            std::vector<std::string> availablesMetrics;
            for (auto && supportedAPI : queryApiSupported) {
                availablesMetrics.push_back(supportedAPI.first);
            }
            return availablesMetrics;
        }}
    };

    auto it = queryApiSupported.find(name);
    if (it == queryApiSupported.end()) {
        THROW_GNA_EXCEPTION << "Unsupported parameters for GetMetric: " << name;
    }

    return it->second();
}

Parameter GNAPlugin::GetAvailableDevices() const {
    std::vector<std::string> devices;
    // probing for gna-sw-exact, or gna-sw implementation part of libgna
    try {
#if GNA_LIB_VER == 2
        GNADeviceHelper swHelper(Gna2AccelerationModeSoftware);
#else
        GNADeviceHelper swHelper(GNA_SOFTWARE);
#endif
        devices.push_back("GNA_SW");
    }catch(...) {}

    try {
#if GNA_LIB_VER == 2
        GNADeviceHelper hwHelper(Gna2AccelerationModeHardware);
#else
        GNADeviceHelper hwHelper(GNA_HARDWARE);
#endif
#if GNA_LIB_VER == 1
        try {
            intel_nnet_type_t neuralNetwork = { 0 };
            hwHelper.propagate(&neuralNetwork, nullptr, 0);
        }catch (...) {
            if (hwHelper.getGNAStatus() != GNA_DEVNOTFOUND) {
                devices.push_back("GNA_HW");
            }
        }
#else
        if (hwHelper.hasGnaHw()) {
            devices.push_back("GNA_HW");
        }
#endif
    }catch(...) {}

    return devices;
}

std::map<std::string, std::string> GNAPlugin::supportedConfigKeysWithDefaults() const {
    std::map<std::string, std::string>  options = {
        {GNA_CONFIG_KEY(SCALE_FACTOR), "1.0"},
        {GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE), ""},
        {GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION), ""},
        {GNA_CONFIG_KEY(DEVICE_MODE), GNAConfigParams::GNA_AUTO},
        {GNA_CONFIG_KEY(COMPACT_MODE), CONFIG_VALUE(NO)},
        {CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS), CONFIG_VALUE(NO)},
        {GNA_CONFIG_KEY(PRECISION), Precision(Precision::I8).name()},
        {GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN), CONFIG_VALUE(YES)},
        {CONFIG_KEY(PERF_COUNT), CONFIG_VALUE(NO)},
        {GNA_CONFIG_KEY(LIB_N_THREADS), "1"},
        {CONFIG_KEY(SINGLE_THREAD), CONFIG_VALUE(YES)}
    };
    return options;
}


std::vector<std::string> GNAPlugin::supportedConfigKeys()const {
    std::vector<std::string> result;
    for (auto && configOption : supportedConfigKeysWithDefaults()) {
        result.push_back(configOption.first);
    }
    return result;
}
