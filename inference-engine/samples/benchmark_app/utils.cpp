// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"

uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device) {
    static const std::map<std::string, uint32_t> deviceDefaultDurationInSeconds {
            { "CPU",     60  },
            { "GPU",     60  },
            { "VPU",     60  },
            { "MYRIAD",  60  },
            { "HDDL",    60  },
            { "FPGA",    120 },
            { "UNKNOWN", 120 }
    };
    uint32_t duration = 0;
    for (const auto& deviceDurationInSeconds : deviceDefaultDurationInSeconds) {
        if (device.find(deviceDurationInSeconds.first) != std::string::npos) {
            duration = std::max(duration, deviceDurationInSeconds.second);
        }
    }
    if (duration == 0) {
        const auto unknownDeviceIt = find_if(
            deviceDefaultDurationInSeconds.begin(),
            deviceDefaultDurationInSeconds.end(),
            [](std::pair<std::string, uint32_t> deviceDuration) { return deviceDuration.first == "UNKNOWN"; });

        if (unknownDeviceIt == deviceDefaultDurationInSeconds.end()) {
            throw std::logic_error("UNKNOWN device was not found in the device duration list");
        }
        duration = unknownDeviceIt->second;
        slog::warn << "Default duration " << duration << " seconds for unknown device '" << device << "' is used" << slog::endl;
    }
    return duration;
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(item);
    }
    return result;
}

std::vector<std::string> parseDevices(const std::string& device_string) {
    std::string comma_separated_devices = device_string;
    if (comma_separated_devices.find(":") != std::string::npos) {
        comma_separated_devices = comma_separated_devices.substr(comma_separated_devices.find(":") + 1);
    }
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find_first_of(".("));
    return devices;
}

std::map<std::string, uint32_t> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
                                                            const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    auto values_string_upper = values_string;
    std::map<std::string, uint32_t> result;
    auto device_value_strings = split(values_string_upper, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec =  split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto device_name = device_value_vec.at(0);
            auto nstreams = device_value_vec.at(1);
            auto it = std::find(devices.begin(), devices.end(), device_name);
            if (it != devices.end()) {
                result[device_name] = std::stoi(nstreams);
            } else {
                throw std::logic_error("Can't set nstreams value " + std::string(nstreams) +
                                       " for device '" + device_name + "'! Incorrect device name!");
            }
        } else if (device_value_vec.size() == 1) {
            uint32_t value = std::stoi(device_value_vec.at(0));
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}
