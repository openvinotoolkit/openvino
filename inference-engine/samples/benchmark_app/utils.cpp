// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string>
#include <algorithm>
#include <utility>
#include <vector>
#include <map>
#include <regex>

#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"

#ifdef USE_OPENCV
#include <opencv2/core.hpp>
#endif

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
    if ((comma_separated_devices == "MULTI") || (comma_separated_devices == "HETERO"))
        return std::vector<std::string>();
    auto devices = split(comma_separated_devices, ',');
    for (auto& device : devices)
        device = device.substr(0, device.find_first_of(".("));
    return devices;
}

std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices,
                                                               const std::string& values_string) {
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
                throw std::logic_error("Can't set nstreams value " + std::string(nstreams) +
                                       " for device '" + device_name + "'! Incorrect device name!");
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

bool adjustShapesBatch(InferenceEngine::ICNNNetwork::InputShapes& shapes,
                       const size_t batch_size, const InferenceEngine::InputsDataMap& input_info) {
    bool updated = false;
    for (auto& item : input_info) {
        auto layout = item.second->getTensorDesc().getLayout();

        int batch_index = -1;
        if ((layout == InferenceEngine::Layout::NCHW) || (layout == InferenceEngine::Layout::NCDHW) ||
            (layout == InferenceEngine::Layout::NHWC) || (layout == InferenceEngine::Layout::NDHWC) ||
            (layout == InferenceEngine::Layout::NC)) {
            batch_index = 0;
        } else if (layout == InferenceEngine::Layout::CN) {
            batch_index = 1;
        }
        if ((batch_index != -1) && (shapes.at(item.first).at(batch_index) != batch_size)) {
            shapes[item.first][batch_index] = batch_size;
            updated = true;
        }
    }
    return updated;
}

bool updateShapes(InferenceEngine::ICNNNetwork::InputShapes& shapes,
                  const std::string shapes_string, const InferenceEngine::InputsDataMap& input_info) {
    bool updated = false;
    std::string search_string = shapes_string;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        auto input_shape = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        std::vector<size_t> parsed_shape;
        for (auto& dim : split(input_shape, ',')) {
            parsed_shape.push_back(std::stoi(dim));
        }
        if (!input_name.empty()) {
            shapes[input_name] = parsed_shape;
            updated = true;
        } else {
            for (auto& item : input_info) {
                shapes[item.first] = parsed_shape;
            }
            updated = true;
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse `shape` parameter: " + shapes_string);
    return updated;
}

std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes) {
    std::stringstream ss;
    for (auto& shape : shapes) {
        if (!ss.str().empty()) ss << ", ";
        ss << "\'" << shape.first << "': [";
        for (size_t i = 0; i < shape.second.size(); i++) {
            if (i > 0) ss << ", ";
            ss << shape.second.at(i);
        }
        ss << "]";
    }
    return ss.str();
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename,
                 const std::map<std::string, std::map<std::string, std::string>>& config) {
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        throw std::runtime_error("Error: Can't open config file : " + filename);
    for (auto device_it = config.begin(); device_it != config.end(); ++device_it) {
        fs << device_it->first  << "{:";
        for (auto param_it = device_it->second.begin(); param_it != device_it->second.end(); ++param_it)
            fs << param_it->first << param_it->second;
        fs << "}";
    }
    fs.release();
}

void load_config(const std::string& filename,
                 std::map<std::string, std::map<std::string, std::string>>& config) {
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