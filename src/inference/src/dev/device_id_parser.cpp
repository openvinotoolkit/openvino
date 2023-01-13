// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_icore.hpp"

namespace InferenceEngine {

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    deviceName = deviceNameWithID;

    auto pos = deviceName.find('.');
    if (pos != std::string::npos) {
        deviceName = deviceNameWithID.substr(0, pos);
        deviceID = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

std::string DeviceIDParser::getDeviceID() const {
    return deviceID;
}

std::string DeviceIDParser::getDeviceName() const {
    return deviceName;
}

std::vector<std::string> DeviceIDParser::getHeteroDevices(std::string fallbackDevice) {
    std::vector<std::string> deviceNames;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallbackDevice.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallbackDevice.substr(0, pos));
        fallbackDevice.erase(0, pos + 1);
    }

    if (!fallbackDevice.empty())
        deviceNames.push_back(fallbackDevice);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::getMultiDevices(std::string devicesList) {
    std::set<std::string> deviceNames;
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    std::string device;
    char delimiter = ',';
    size_t pos = 0;
    // in addition to the list of devices, every device can have a #requests in the brackets e.g. "CPU(100)"
    // we skip the #requests info here
    while ((pos = devicesList.find(delimiter)) != std::string::npos) {
        auto d = devicesList.substr(0, pos);
        if (d.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = d.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::getBatchDevice(d.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(d));
        }
        devicesList.erase(0, pos + 1);
    }

    if (!devicesList.empty()) {
        if (devicesList.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = devicesList.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::getBatchDevice(devicesList.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(devicesList));
        }
    }
    return std::vector<std::string>(deviceNames.begin(), deviceNames.end());
}

std::string DeviceIDParser::getBatchDevice(std::string device) {
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    return trim_request_info(device);
}
}  // namespace InferenceEngine
