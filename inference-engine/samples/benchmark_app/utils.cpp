// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// clang-format off
#include <algorithm>
#include <map>
#include <regex>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <string>
#include <utility>
#include <vector>

#include "utils.hpp"
// clang-format on

#ifdef USE_OPENCV
    #include <opencv2/core.hpp>
#endif

namespace benchmark_app {
bool InputInfo::isImage() const {
    if ((layout != "NCHW") && (layout != "NHWC") && (layout != "CHW") && (layout != "HWC"))
        return false;
    return (channels() == 3);
}
bool InputInfo::isImageInfo() const {
    if (layout != "NC")
        return false;
    return (channels() >= 2);
}
size_t InputInfo::getDimentionByLayout(char character) const {
    size_t pos = layout.find(character);
    if (pos == std::string::npos)
        throw std::runtime_error("Error: Can't get " + std::string(character, 1) + " from layout " + layout);
    return shape.at(pos);
}
size_t InputInfo::width() const {
    return getDimentionByLayout('W');
}
size_t InputInfo::height() const {
    return getDimentionByLayout('H');
}
size_t InputInfo::channels() const {
    return getDimentionByLayout('C');
}
size_t InputInfo::batch() const {
    return getDimentionByLayout('N');
}
size_t InputInfo::depth() const {
    return getDimentionByLayout('D');
}
}  // namespace benchmark_app

uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device) {
    static const std::map<std::string, uint32_t> deviceDefaultDurationInSeconds {{"CPU", 60},  {"GPU", 60},   {"VPU", 60},     {"MYRIAD", 60},
                                                                                 {"HDDL", 60}, {"FPGA", 120}, {"UNKNOWN", 120}};
    uint32_t duration = 0;
    for (const auto& deviceDurationInSeconds : deviceDefaultDurationInSeconds) {
        if (device.find(deviceDurationInSeconds.first) != std::string::npos) {
            duration = std::max(duration, deviceDurationInSeconds.second);
        }
    }
    if (duration == 0) {
        const auto unknownDeviceIt =
            find_if(deviceDefaultDurationInSeconds.begin(), deviceDefaultDurationInSeconds.end(), [](std::pair<std::string, uint32_t> deviceDuration) {
                return deviceDuration.first == "UNKNOWN";
            });

        if (unknownDeviceIt == deviceDefaultDurationInSeconds.end()) {
            throw std::logic_error("UNKNOWN device was not found in the device duration list");
        }
        duration = unknownDeviceIt->second;
        slog::warn << "Default duration " << duration << " seconds for unknown device '" << device << "' is used" << slog::endl;
    }
    return duration;
}

std::vector<std::string> split(const std::string& s, char delim) {
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
    if ((comma_separated_devices == "MULTI") || (comma_separated_devices == "HETERO"))
        return std::vector<std::string>();
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find_first_of(".("));
    return devices;
}

std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices, const std::string& values_string) {
    //  Format: <device1>:<value1>,<device2>:<value2> or just <value>
    std::map<std::string, std::string> result;
    auto device_value_strings = split(values_string, ',');
    for (auto& device_value_string : device_value_strings) {
        auto device_value_vec = split(device_value_string, ':');
        if (device_value_vec.size() == 2) {
            auto device_name = device_value_vec.at(0);
            auto nstreams = device_value_vec.at(1);
            auto it = std::find(devices.begin(), devices.end(), device_name);
            if (it != devices.end()) {
                result[device_name] = nstreams;
            } else {
                throw std::logic_error("Can't set nstreams value " + std::string(nstreams) + " for device '" + device_name + "'! Incorrect device name!");
            }
        } else if (device_value_vec.size() == 1) {
            auto value = device_value_vec.at(0);
            for (auto& device : devices) {
                result[device] = value;
            }
        } else if (device_value_vec.size() != 0) {
            throw std::runtime_error("Unknown string format: " + values_string);
        }
    }
    return result;
}

size_t getBatchSize(const benchmark_app::InputsInfo& inputs_info) {
    size_t batch_size = 0;
    for (auto& info : inputs_info) {
        std::size_t batch_index = info.second.layout.find("N");
        if (batch_index != std::string::npos) {
            if (batch_size == 0)
                batch_size = info.second.shape[batch_index];
            else if (batch_size != info.second.shape[batch_index])
                throw std::logic_error("Can't deterimine batch size: batch is "
                                       "different for different inputs!");
        }
    }
    if (batch_size == 0)
        batch_size = 1;
    return batch_size;
}

std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes) {
    std::stringstream ss;
    for (auto& shape : shapes) {
        if (!ss.str().empty())
            ss << ", ";
        ss << "\'" << shape.first << "': [";
        for (size_t i = 0; i < shape.second.size(); i++) {
            if (i > 0)
                ss << ", ";
            ss << shape.second.at(i);
        }
        ss << "]";
    }
    return ss.str();
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename, const std::map<std::string, std::map<std::string, std::string>>& config) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        throw std::runtime_error("Error: Can't open config file : " + filename);
    for (auto device_it = config.begin(); device_it != config.end(); ++device_it) {
        fs << device_it->first << "{:";
        for (auto param_it = device_it->second.begin(); param_it != device_it->second.end(); ++param_it)
            fs << param_it->first << param_it->second;
        fs << "}";
    }
    fs.release();
}

void load_config(const std::string& filename, std::map<std::string, std::map<std::string, std::string>>& config) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        throw std::runtime_error("Error: Can't load config file : " + filename);
    cv::FileNode root = fs.root();
    for (auto it = root.begin(); it != root.end(); ++it) {
        auto device = *it;
        if (!device.isMap()) {
            throw std::runtime_error("Error: Can't parse config file : " + filename);
        }
        for (auto iit = device.begin(); iit != device.end(); ++iit) {
            auto item = *iit;
            config[device.name()][item.name()] = item.string();
        }
    }
}
#endif