// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

namespace benchmark_app {
struct InputInfo {
    InferenceEngine::Precision precision;
    InferenceEngine::SizeVector shape;
    std::string layout;
    bool isImage() const;
    bool isImageInfo() const;
    size_t getDimentionByLayout(char character) const;
    size_t width() const;
    size_t height() const;
    size_t channels() const;
    size_t batch() const;
    size_t depth() const;
};
using InputsInfo = std::map<std::string, InputInfo>;
}  // namespace benchmark_app

std::vector<std::string> parseDevices(const std::string& device_string);
uint32_t deviceDefaultDeviceDurationInSeconds(const std::string& device);
std::map<std::string, std::string> parseNStreamsValuePerDevice(const std::vector<std::string>& devices, const std::string& values_string);
std::string getShapesString(const InferenceEngine::ICNNNetwork::InputShapes& shapes);
size_t getBatchSize(const benchmark_app::InputsInfo& inputs_info);
std::vector<std::string> split(const std::string& s, char delim);

template <typename T>
std::map<std::string, std::string> parseInputParameters(const std::string parameter_string, const std::map<std::string, T>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::string> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        auto input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[input_name] = input_value;
        } else {
            for (auto& item : input_info) {
                return_value[item.first] = input_value;
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || search_string.front() != ',')
            break;
        search_string = search_string.substr(1);
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}

template <typename T>
benchmark_app::InputsInfo getInputsInfo(const std::string& shape_string, const std::string& layout_string, const size_t batch_size,
                                        const std::map<std::string, T>& input_info, bool& reshape_required) {
    std::map<std::string, std::string> shape_map = parseInputParameters(shape_string, input_info);
    std::map<std::string, std::string> layout_map = parseInputParameters(layout_string, input_info);
    reshape_required = false;
    benchmark_app::InputsInfo info_map;
    for (auto& item : input_info) {
        benchmark_app::InputInfo info;
        auto name = item.first;
        auto descriptor = item.second->getTensorDesc();
        // Precision
        info.precision = descriptor.getPrecision();
        // Shape
        if (shape_map.count(name)) {
            std::vector<size_t> parsed_shape;
            for (auto& dim : split(shape_map.at(name), ',')) {
                parsed_shape.push_back(std::stoi(dim));
            }
            info.shape = parsed_shape;
            reshape_required = true;
        } else {
            info.shape = descriptor.getDims();
        }
        // Layout
        if (layout_map.count(name)) {
            info.layout = layout_map.at(name);
            std::transform(info.layout.begin(), info.layout.end(), info.layout.begin(), ::toupper);
        } else {
            std::stringstream ss;
            ss << descriptor.getLayout();
            info.layout = ss.str();
        }
        // Update shape with batch if needed
        if (batch_size != 0) {
            std::size_t batch_index = info.layout.find("N");
            if ((batch_index != std::string::npos) && (info.shape.at(batch_index) != batch_size)) {
                info.shape[batch_index] = batch_size;
                reshape_required = true;
            }
        }
        info_map[name] = info;
    }
    return info_map;
}

template <typename T>
benchmark_app::InputsInfo getInputsInfo(const std::string& shape_string, const std::string& layout_string, const size_t batch_size,
                                        const std::map<std::string, T>& input_info) {
    bool reshape_required = false;
    return getInputsInfo<T>(shape_string, layout_string, batch_size, input_info, reshape_required);
}

#ifdef USE_OPENCV
void dump_config(const std::string& filename, const std::map<std::string, std::map<std::string, std::string>>& config);
void load_config(const std::string& filename, std::map<std::string, std::map<std::string, std::string>>& config);
#endif