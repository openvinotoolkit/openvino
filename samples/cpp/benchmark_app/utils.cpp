// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <map>
#include <regex>
#include <string>
#include <utility>
#include <vector>

// clang-format off
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>

#include "utils.hpp"
// clang-format on

#ifdef USE_OPENCV
#    include <opencv2/core.hpp>
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
    return dataShape.at(pos);
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
    static const std::map<std::string, uint32_t> deviceDefaultDurationInSeconds{{"CPU", 60},
                                                                                {"GPU", 60},
                                                                                {"VPU", 60},
                                                                                {"MYRIAD", 60},
                                                                                {"HDDL", 60},
                                                                                {"UNKNOWN", 120}};
    uint32_t duration = 0;
    for (const auto& deviceDurationInSeconds : deviceDefaultDurationInSeconds) {
        if (device.find(deviceDurationInSeconds.first) != std::string::npos) {
            duration = std::max(duration, deviceDurationInSeconds.second);
        }
    }
    if (duration == 0) {
        const auto unknownDeviceIt = find_if(deviceDefaultDurationInSeconds.begin(),
                                             deviceDefaultDurationInSeconds.end(),
                                             [](std::pair<std::string, uint32_t> deviceDuration) {
                                                 return deviceDuration.first == "UNKNOWN";
                                             });

        if (unknownDeviceIt == deviceDefaultDurationInSeconds.end()) {
            throw std::logic_error("UNKNOWN device was not found in the device duration list");
        }
        duration = unknownDeviceIt->second;
        slog::warn << "Default duration " << duration << " seconds for unknown device '" << device << "' is used"
                   << slog::endl;
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

std::vector<float> splitFloat(const std::string& s, char delim) {
    std::vector<float> result;
    std::stringstream ss(s);
    std::string item;

    while (getline(ss, item, delim)) {
        result.push_back(std::stof(item));
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
                throw std::logic_error("Can't set nstreams value " + std::string(nstreams) + " for device '" +
                                       device_name + "'! Incorrect device name!");
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
                batch_size = info.second.dataShape[batch_index];
            else if (batch_size != info.second.dataShape[batch_index])
                throw std::logic_error("Can't deterimine batch size: batch is "
                                       "different for different inputs!");
        }
    }
    if (batch_size == 0)
        batch_size = 1;
    return batch_size;
}

InferenceEngine::Layout getLayoutFromString(const std::string& string_layout) {
    static const std::unordered_map<std::string, InferenceEngine::Layout> layouts = {
        {"NCHW", InferenceEngine::Layout::NCHW},
        {"NHWC", InferenceEngine::Layout::NHWC},
        {"NCDHW", InferenceEngine::Layout::NCDHW},
        {"NDHWC", InferenceEngine::Layout::NDHWC},
        {"C", InferenceEngine::Layout::C},
        {"CHW", InferenceEngine::Layout::CHW},
        {"HWC", InferenceEngine::Layout::HWC},
        {"HW", InferenceEngine::Layout::HW},
        {"NC", InferenceEngine::Layout::NC},
        {"CN", InferenceEngine::Layout::CN}};
    auto it = layouts.find(string_layout);
    if (it != layouts.end()) {
        return it->second;
    }
    IE_THROW() << "Unknown layout with name '" << string_layout << "'.";
}

std::string getShapeString(const InferenceEngine::SizeVector& shape) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0)
            ss << ", ";
        ss << shape.at(i);
    }
    ss << "]";
    return ss.str();
}

std::string getShapesString(const benchmark_app::PartialShapes& shapes) {
    std::stringstream ss;
    for (auto& shape : shapes) {
        if (!ss.str().empty())
            ss << ", ";
        ss << "\'" << shape.first << "': " << shape.second;
    }
    return ss.str();
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

std::map<std::string, std::vector<float>> parseScaleOrMean(const std::string& scale_mean,
                                                           const benchmark_app::InputsInfo& inputs_info) {
    //  Format: data:[255,255,255],info[255,255,255]
    std::map<std::string, std::vector<float>> return_value;

    std::string search_string = scale_mean;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        auto input_value_string = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        auto input_value = splitFloat(input_value_string, ',');

        if (!input_name.empty()) {
            if (inputs_info.count(input_name)) {
                return_value[input_name] = input_value;
            }
            // ignore wrong input name
        } else {
            for (auto& item : inputs_info) {
                if (item.second.isImage())
                    return_value[item.first] = input_value;
            }
            search_string.clear();
            break;
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + scale_mean);
    return return_value;
}

std::vector<ngraph::Dimension> parsePartialShape(const std::string& partial_shape) {
    std::vector<ngraph::Dimension> shape;
    for (auto& dim : split(partial_shape, ',')) {
        if (dim == "?" || dim == "-1") {
            shape.push_back(ngraph::Dimension::dynamic());
        } else {
            const std::string range_divider = "..";
            size_t range_index = dim.find(range_divider);
            if (range_index != std::string::npos) {
                std::string min = dim.substr(0, range_index);
                std::string max = dim.substr(range_index + range_divider.length());
                shape.push_back(ngraph::Dimension(min.empty() ? 0 : std::stoi(min),
                                                  max.empty() ? ngraph::Interval::s_max : std::stoi(max)));
            } else {
                shape.push_back(std::stoi(dim));
            }
        }
    }

    return shape;
}

InferenceEngine::SizeVector parseTensorShape(const std::string& dataShape) {
    std::vector<size_t> shape;
    for (auto& dim : split(dataShape, ',')) {
        shape.push_back(std::stoi(dim));
    }
    return shape;
}

std::pair<std::string, std::vector<std::string>> parseInputFiles(const std::string& file_paths_string) {
    auto search_string = file_paths_string;
    std::string input_name = "";
    std::vector<std::string> file_paths;

    // parse strings like <input1>:file1,file2,file3 and get name from them
    size_t semicolon_pos = search_string.find_first_of(":");
    size_t quote_pos = search_string.find_first_of("\"");
    if (semicolon_pos != std::string::npos && quote_pos != std::string::npos && semicolon_pos > quote_pos) {
        // if : is found after opening " symbol - this means that " belongs to pathname
        semicolon_pos = std::string::npos;
    }
    if (search_string.length() > 2 && semicolon_pos == 1 && search_string[2] == '\\') {
        // Special case like C:\ denotes drive name, not an input name
        semicolon_pos = std::string::npos;
    }

    if (semicolon_pos != std::string::npos) {
        input_name = search_string.substr(0, semicolon_pos);
        search_string = search_string.substr(semicolon_pos + 1);
    }

    // parse file1,file2,file3 and get vector of paths
    size_t coma_pos = 0;
    do {
        coma_pos = search_string.find_first_of(',');
        file_paths.push_back(search_string.substr(0, coma_pos));
        if (coma_pos == std::string::npos) {
            search_string = "";
            break;
        }
        search_string = search_string.substr(coma_pos + 1);
    } while (coma_pos != std::string::npos);

    if (!search_string.empty())
        throw std::logic_error("Can't parse file paths for input " + input_name +
                               " in input parameter string: " + file_paths_string);

    return {input_name, file_paths};
}

std::map<std::string, std::vector<std::string>> parseInputArguments(const std::vector<std::string>& args) {
    std::map<std::string, std::vector<std::string>> mapped_files = {};
    auto args_it = begin(args);
    const auto is_image_arg = [](const std::string& s) {
        return s == "-i";
    };
    const auto is_arg = [](const std::string& s) {
        return s.front() == '-';
    };
    while (args_it != args.end()) {
        const auto files_start = std::find_if(args_it, end(args), is_image_arg);
        if (files_start == end(args)) {
            break;
        }
        const auto files_begin = std::next(files_start);
        const auto files_end = std::find_if(files_begin, end(args), is_arg);
        for (auto f = files_begin; f != files_end; ++f) {
            auto files = parseInputFiles(*f);
            if (mapped_files.find(files.first) == mapped_files.end()) {
                mapped_files[files.first] = {};
            }

            for (auto& file : files.second) {
                readInputFilesArguments(mapped_files[files.first], file);
            }
        }
        args_it = files_end;
    }
    size_t max_files = 20;
    for (auto& files : mapped_files) {
        if (files.second.size() <= max_files) {
            slog::info << "For input " << files.first << " " << files.second.size() << " files were added. "
                       << slog::endl;
        } else {
            slog::info << "For input " << files.first << " " << files.second.size() << " files were added. "
                       << " The number of files will be limited to " << max_files << "." << slog::endl;
            files.second.resize(20);
        }
    }

    return mapped_files;
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename, const std::map<std::string, std::map<std::string, std::string>>& config) {
    auto plugin_to_opencv_format = [](const std::string& str) -> std::string {
        if (str.find("_") != std::string::npos) {
            slog::warn
                << "Device name contains \"_\" and will be changed during loading of configuration due to limitations."
                   "This configuration file could not be loaded correctly."
                << slog::endl;
        }
        std::string new_str(str);
        auto pos = new_str.find(".");
        if (pos != std::string::npos) {
            new_str.replace(pos, 1, "_");
        }
        return new_str;
    };
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);
    if (!fs.isOpened())
        throw std::runtime_error("Error: Can't open config file : " + filename);
    for (auto device_it = config.begin(); device_it != config.end(); ++device_it) {
        fs << plugin_to_opencv_format(device_it->first) << "{:";
        for (auto param_it = device_it->second.begin(); param_it != device_it->second.end(); ++param_it)
            fs << param_it->first << param_it->second;
        fs << "}";
    }
    fs.release();
}

void load_config(const std::string& filename, std::map<std::string, std::map<std::string, std::string>>& config) {
    auto opencv_to_plugin_format = [](const std::string& str) -> std::string {
        std::string new_str(str);
        auto pos = new_str.find("_");
        if (pos != std::string::npos) {
            new_str.replace(pos, 1, ".");
        }
        return new_str;
    };
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
            config[opencv_to_plugin_format(device.name())][item.name()] = item.string();
        }
    }
}
#endif
