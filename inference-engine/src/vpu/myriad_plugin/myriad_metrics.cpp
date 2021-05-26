// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_metrics.h"
#include "vpu/private_plugin_config.hpp"

#include <algorithm>

#include <vpu/utils/error.hpp>

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine;
using namespace VPUConfigParams;
using namespace PluginConfigParams;

//------------------------------------------------------------------------------
// Implementation of methods of class MyriadMetrics
//------------------------------------------------------------------------------

MyriadMetrics::MyriadMetrics() {
    _supportedMetrics = {
        METRIC_KEY(AVAILABLE_DEVICES),
        METRIC_KEY(FULL_DEVICE_NAME),
        METRIC_KEY(SUPPORTED_METRICS),
        METRIC_KEY(SUPPORTED_CONFIG_KEYS),
        METRIC_KEY(OPTIMIZATION_CAPABILITIES),
        METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS),
        METRIC_KEY(DEVICE_THERMAL),
        METRIC_KEY(DEVICE_ARCHITECTURE),
        METRIC_KEY(IMPORT_EXPORT_SUPPORT),
    };

IE_SUPPRESS_DEPRECATED_START
    // TODO: remove once all options are migrated
    _supportedConfigKeys = {
        MYRIAD_CUSTOM_LAYERS,
        MYRIAD_ENABLE_FORCE_RESET,

        // deprecated
        KEY_VPU_CUSTOM_LAYERS,
        KEY_VPU_MYRIAD_FORCE_RESET,
        KEY_VPU_MYRIAD_PLATFORM,

        CONFIG_KEY(CONFIG_FILE),
    };
IE_SUPPRESS_DEPRECATED_END

    _optimizationCapabilities = { METRIC_VALUE(FP16) };
    _rangeForAsyncInferRequests = RangeType(3, 6, 1);

    _idToDeviceFullNameMap = {
        {"5", "Intel Movidius Myriad 2 VPU"},
        {"8", "Intel Movidius Myriad X VPU"},
    };
}

std::vector<std::string> MyriadMetrics::AvailableDevicesNames(
    const std::shared_ptr<IMvnc> &mvnc,
    const std::vector<DevicePtr> &devicePool) const {
    std::vector<std::string> availableDevices;

    auto unbootedDevices = mvnc->AvailableDevicesNames();
    availableDevices.insert(availableDevices.begin(),
                            unbootedDevices.begin(), unbootedDevices.end());

    for (auto & device : devicePool) {
        availableDevices.push_back(device->_name);
    }

    std::sort(availableDevices.begin(), availableDevices.end());
    return availableDevices;
}

std::string MyriadMetrics::FullName(std::string deviceName) const {
    std::string nameDelimiter("-ma");
    unsigned int indexLength = 4;
    unsigned int placeOfTypeId = 2;

    auto indexStr = deviceName;
    indexStr.erase(0, indexStr.find(nameDelimiter) + nameDelimiter.length());

    if (indexLength != indexStr.length()) {
        return deviceName;
    } else {
        auto myriadId = std::string(1, indexStr[placeOfTypeId]);
        if (_idToDeviceFullNameMap.count(myriadId)) {
            return _idToDeviceFullNameMap.at(myriadId);
        }
    }

    return deviceName;
}

float MyriadMetrics::DevicesThermal(const DevicePtr& device) const {
    VPU_THROW_UNLESS(device != nullptr, "No device specified to get its thermal");
    return MyriadExecutor::GetThermal(device);
}

const std::unordered_set<std::string>& MyriadMetrics::SupportedMetrics() const {
    return _supportedMetrics;
}

const std::unordered_set<std::string>& MyriadMetrics::SupportedConfigKeys() const {
    return _supportedConfigKeys;
}

const std::unordered_set<std::string>& MyriadMetrics::OptimizationCapabilities() const {
    return _optimizationCapabilities;
}

std::string MyriadMetrics::DeviceArchitecture(const std::map<std::string, InferenceEngine::Parameter> & options) const {
    // TODO: Task 49309. Return same architecture for devices which can share same cache
    // E.g. when device "MYRIAD.ma2480-1" is loaded, options.at("DEVICE_ID") will be "ma2480-1"
    // For DEVICE_ID="ma2480-0" and DEVICE_ID="ma2480-1" this method shall return same string, like "ma2480"
    // In this case inference engine will be able to reuse cached model and total reduce load network time
    return "MYRIAD";
}

RangeType MyriadMetrics::RangeForAsyncInferRequests(
    const std::map<std::string, std::string>& config) const {

    auto throughput_streams_str = config.find(InferenceEngine::MYRIAD_THROUGHPUT_STREAMS);
    if (throughput_streams_str != config.end()) {
        try {
            int throughput_streams = std::stoi(throughput_streams_str->second);
            if (throughput_streams > 0) {
                return RangeType(throughput_streams+1, throughput_streams*3, 1);
            }
        }
        catch(...) {
            IE_THROW() << "Invalid config value for MYRIAD_THROUGHPUT_STREAMS, can't cast to int";
        }
    }

    return _rangeForAsyncInferRequests;
}
