// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/device_id_parser.hpp"

#include <set>

namespace ov {

DeviceIDParser::DeviceIDParser(const std::string& deviceNameWithID) {
    m_device_name = deviceNameWithID;

    auto pos = m_device_name.find('.');
    if (pos != std::string::npos) {
        m_device_name = deviceNameWithID.substr(0, pos);
        m_device_id = deviceNameWithID.substr(pos + 1, deviceNameWithID.size());
    }
}

const std::string& DeviceIDParser::get_device_id() const {
    return m_device_id;
}

const std::string& DeviceIDParser::get_device_name() const {
    return m_device_name;
}

std::vector<std::string> DeviceIDParser::get_hetero_devices(const std::string& fallbackDevice) {
    std::vector<std::string> deviceNames;
    std::string fallback_dev = fallbackDevice;

    std::string cdevice;
    char delimiter = ',';
    size_t pos = 0;

    while ((pos = fallback_dev.find(delimiter)) != std::string::npos) {
        deviceNames.push_back(fallback_dev.substr(0, pos));
        fallback_dev.erase(0, pos + 1);
    }

    if (!fallback_dev.empty())
        deviceNames.push_back(fallback_dev);

    return deviceNames;
}

std::vector<std::string> DeviceIDParser::get_multi_devices(const std::string& devicesList) {
    std::string dev_list = devicesList;
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
    while ((pos = dev_list.find(delimiter)) != std::string::npos) {
        auto d = dev_list.substr(0, pos);
        if (d.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = d.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::get_batch_device(d.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(d));
        }
        dev_list.erase(0, pos + 1);
    }

    if (!dev_list.empty()) {
        if (dev_list.find("BATCH") == 0) {
            deviceNames.insert("BATCH");
            auto p = dev_list.find_first_of(":");
            if (p != std::string::npos)
                deviceNames.insert(DeviceIDParser::get_batch_device(dev_list.substr(p + 1)));
        } else {
            deviceNames.insert(trim_request_info(dev_list));
        }
    }
    return std::vector<std::string>(deviceNames.begin(), deviceNames.end());
}

std::string DeviceIDParser::get_batch_device(const std::string& device) {
    if (device.find(",") != std::string::npos) {
        OPENVINO_THROW("BATCH accepts only one device in list but got '", device, "'");
    }
    if (device.find("-") != std::string::npos) {
        OPENVINO_THROW("Invalid device name '", device, "' for BATCH");
    }
    auto trim_request_info = [](const std::string& device_with_requests) {
        auto opening_bracket = device_with_requests.find_first_of('(');
        return device_with_requests.substr(0, opening_bracket);
    };
    return trim_request_info(device);
}
}  // namespace ov
