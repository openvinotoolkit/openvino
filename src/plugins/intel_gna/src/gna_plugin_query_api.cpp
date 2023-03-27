// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cpp_interfaces/interface/ie_internal_plugin_config.hpp>
#include <ie_parameter.hpp>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "gna/gna_config.hpp"
#include "gna_plugin.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;

Parameter GNAPlugin::GetConfig(const std::string& name, const std::map<std::string, Parameter>& /*options*/) const {
    return config.GetParameter(name);
}

Parameter GNAPlugin::GetMetric(const std::string& name,
                               const std::map<std::string, InferenceEngine::Parameter>& options) const {
    if (ov::supported_properties == name) {
        return config.GetSupportedProperties();
    } else if (ov::available_devices == name) {
        return GetAvailableDevices().as<std::vector<std::string>>();
    } else if (ov::optimal_number_of_infer_requests == name) {
        uint32_t nireq = 1;
        return nireq;
    } else if (ov::range_for_async_infer_requests == name) {
        std::tuple<unsigned int, unsigned int, unsigned int> range{1, 1, 1};
        return range;
    } else if (ov::device::capabilities == name) {
        std::vector<std::string> supported_capabilities = {
            ov::device::capability::INT16,
            ov::device::capability::INT8,
            ov::device::capability::EXPORT_IMPORT,
        };
        if (options.count(ov::device::id.name())) {
            if (options.at(ov::device::id.name()).as<std::string>().compare("GNA_HW") != 0) {
                supported_capabilities.emplace_back(ov::device::capability::FP32);
            }
        }
        return supported_capabilities;
    } else if (ov::device::full_name == name) {
        if (!options.count(ov::device::id.name())) {
            auto availableDevices = GetAvailableDevices().as<std::vector<std::string>>();
            if (availableDevices.empty()) {
                THROW_GNA_EXCEPTION << "No devices available.";
            } else if (availableDevices.size() > 2) {
                THROW_GNA_EXCEPTION << "ov::device::id not set in request for ov::device::full_name";
            }
            return availableDevices.back();
        } else {
            return options.at(ov::device::id.name());
        }
    } else if (ov::intel_gna::library_full_version == name) {
        return GNADeviceHelper::GetGnaLibraryVersion();
    } else if (ov::execution_devices == name) {
        return decltype(ov::execution_devices)::value_type{GetName()};
    } else if (ov::model_name == name) {
        return _network_name;
    } else if (name == ov::caching_properties) {
        auto cachingProperties =
            Config::GetImpactingModelCompilationProperties(true).as<std::vector<ov::PropertyName>>();
        cachingProperties.push_back(ov::PropertyName(ov::log::level.name(), ov::PropertyMutability::RO));
        return decltype(ov::caching_properties)::value_type(cachingProperties);
    } else {
        const std::unordered_map<std::string, std::function<Parameter()>> queryApiSupported = {
            {METRIC_KEY(AVAILABLE_DEVICES),
             [this]() {
                 return GetAvailableDevices();
             }},
            {METRIC_KEY(SUPPORTED_CONFIG_KEYS),
             [this]() {
                 return config.GetSupportedKeys();
             }},
            {METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS),
             []() {
                 uint32_t nireq = 1;
                 return nireq;
             }},
            {METRIC_KEY(FULL_DEVICE_NAME),
             [&options, this]() {
                 auto availableDevices = GetAvailableDevices().as<std::vector<std::string>>();

                 if (availableDevices.empty()) {
                     THROW_GNA_EXCEPTION << "No devices available.";
                 }

                 if (!options.count(KEY_DEVICE_ID)) {
                     if (availableDevices.size() == 1 || availableDevices.size() == 2) {
                         return availableDevices.back();  // detection order is GNA_SW, GNA_HW
                     } else {
                         THROW_GNA_EXCEPTION << "KEY_DEVICE_ID not set in request for FULL_DEVICE_NAME";
                     }
                 }

                 auto deviceName = options.at(KEY_DEVICE_ID).as<std::string>();
                 return deviceName;
             }},
            {METRIC_KEY(GNA_LIBRARY_FULL_VERSION),
             []() {
                 return GNADeviceHelper::GetGnaLibraryVersion();
             }},
            {METRIC_KEY(SUPPORTED_METRICS),
             [&queryApiSupported]() {
                 std::vector<std::string> availablesMetrics;
                 for (auto&& supportedAPI : queryApiSupported) {
                     availablesMetrics.push_back(supportedAPI.first);
                 }
                 return availablesMetrics;
             }},
            {METRIC_KEY(IMPORT_EXPORT_SUPPORT), []() {
                 return true;
             }}};

        auto it = queryApiSupported.find(name);
        if (it == queryApiSupported.end()) {
            THROW_GNA_EXCEPTION << "Unsupported parameters for GetMetric: " << name;
        }

        return it->second();
    }
}

Parameter GNAPlugin::GetAvailableDevices() const {
    std::vector<std::string> devices;

    try {
        GNADeviceHelper helper;
        devices.push_back("GNA_SW");
        if (helper.is_hw_detected()) {
            devices.push_back("GNA_HW");
        }
    } catch (...) {
    }

    return devices;
}
