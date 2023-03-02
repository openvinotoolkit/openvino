// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/device_id_parser.hpp"

#include <set>

ov::DeviceIDParser::DeviceIDParser(const std::string& device_name_with_id) {
    m_device_name = device_name_with_id;

    auto pos = m_device_name.find('.');
    if (pos != std::string::npos) {
        m_device_name = device_name_with_id.substr(0, pos);
        m_device_id = device_name_with_id.substr(pos + 1, device_name_with_id.size());
    }
}

std::string ov::DeviceIDParser::get_device_id() const {
    return m_device_id;
}

std::string ov::DeviceIDParser::get_device_name() const {
    return m_device_name;
}

std::vector<std::string> ov::DeviceIDParser::get_hetero_devices(std::string fallbackDevice) {
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

std::vector<std::string> ov::DeviceIDParser::get_multi_devices(std::string devicesList) {
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
                deviceNames.insert(ov::DeviceIDParser::get_batch_device(d.substr(p + 1)));
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
                deviceNames.insert(ov::DeviceIDParser::get_batch_device(devicesList.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(devicesList));
        }
    }
    return std::vector<std::string>(deviceNames.begin(), deviceNames.end());
}

std::string ov::DeviceIDParser::get_batch_device(std::string device) {
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    return trim_request_info(device);
}
