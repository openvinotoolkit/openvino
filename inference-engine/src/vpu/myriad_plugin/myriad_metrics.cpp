// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "myriad_metrics.h"
#include <algorithm>

using namespace vpu::MyriadPlugin;
using namespace InferenceEngine::VPUConfigParams;
using namespace InferenceEngine::PluginConfigParams;

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
        METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)
    };

    _supportedConfigKeys = {
        KEY_VPU_HW_STAGES_OPTIMIZATION,
        KEY_VPU_LOG_LEVEL,
        KEY_VPU_PRINT_RECEIVE_TENSOR_TIME,
        KEY_VPU_NETWORK_CONFIG,
        KEY_VPU_COMPUTE_LAYOUT,
        KEY_VPU_CUSTOM_LAYERS,
        KEY_VPU_IGNORE_IR_STATISTIC,
        KEY_VPU_MYRIAD_FORCE_RESET,
        KEY_VPU_MYRIAD_PLATFORM,
        KEY_EXCLUSIVE_ASYNC_REQUESTS,
        KEY_LOG_LEVEL,
        KEY_PERF_COUNT,
        KEY_CONFIG_FILE,
        KEY_DEVICE_ID
    };

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
    unsigned int indexLenght = 4;
    unsigned int placeOfTypeId = 2;

    auto indexStr = deviceName;
    indexStr.erase(0, indexStr.find(nameDelimiter) + nameDelimiter.length());

    if (indexLenght != indexStr.length()) {
        return deviceName;
    } else {
        auto myriadId = std::string(1, indexStr[placeOfTypeId]);
        if (_idToDeviceFullNameMap.count(myriadId)) {
            return _idToDeviceFullNameMap.at(myriadId);
        }
    }

    return deviceName;
}

const std::vector<std::string>& MyriadMetrics::SupportedMetrics() const {
    return _supportedMetrics;
}

const std::vector<std::string>& MyriadMetrics::SupportedConfigKeys() const {
    return _supportedConfigKeys;
}

const std::vector<std::string>& MyriadMetrics::OptimizationCapabilities() const {
    return _optimizationCapabilities;
}

RangeType MyriadMetrics::RangeForAsyncInferRequests(
    const std::map<std::string, std::string>& config) const {

    auto throughput_streams_str = config.find(KEY_VPU_MYRIAD_THROUGHPUT_STREAMS);
    if (throughput_streams_str != config.end()) {
        try {
            int throughput_streams = std::stoi(throughput_streams_str->second);
            if (throughput_streams > 0) {
                return RangeType(throughput_streams+1, throughput_streams*3, 1);
            }
        }
        catch(...) {
            THROW_IE_EXCEPTION << "Invalid config value for VPU_MYRIAD_THROUGHPUT_STREAMS, can't cast to int";
        }
    }

    return _rangeForAsyncInferRequests;
}
