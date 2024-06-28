// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include <gflags/gflags.h>

#include "openvino/openvino.hpp"

struct InputInfo {
    ov::element::Type type;
    ov::PartialShape partialShape;
    ov::Shape dataShape;
    ov::Layout layout;
};
using InputsInfo = std::map<std::string, InputInfo>;

std::string parameterNameToTensorName(std::string& name, std::vector<ov::Output<ov::Node>>& inputs_info) {
    auto count_name = std::any_of(inputs_info.begin(), inputs_info.end(), [name](ov::Output<ov::Node>& port) {
        return port.get_names().count(name) > 0;
    });
    if (count_name) {
        return name;
    } else {
        auto inputInfo = std::find_if(inputs_info.begin(), inputs_info.end(), [name](ov::Output<ov::Node>& port) {
            return name == port.get_node()->get_friendly_name();
        });
        if (inputInfo == inputs_info.end()) {
            throw std::runtime_error("Provided I/O name \"" + name +
                                     "\" is not found neither in tensor names nor in nodes names.");
        }
        return inputInfo->get_any_name();
    }
}

std::map<std::string, std::vector<std::string>> parseInputParameters(std::string& parameter_string,
                                                                     std::vector<ov::Output<ov::Node>>& input_info) {
    // Parse parameter string like "input0[value0],input1[value1]" or "[value]" (applied to all
    // inputs)
    std::map<std::string, std::vector<std::string>> return_value;
    std::string search_string = parameter_string;
    auto start_pos = search_string.find_first_of('[');
    auto input_name = search_string.substr(0, start_pos);
    while (start_pos != std::string::npos) {
        auto end_pos = search_string.find_first_of(']');
        if (end_pos == std::string::npos)
            break;
        input_name = search_string.substr(0, start_pos);
        auto input_value = search_string.substr(start_pos + 1, end_pos - start_pos - 1);
        if (!input_name.empty()) {
            return_value[parameterNameToTensorName(input_name, input_info)].push_back(input_value);
        } else {
            for (auto& item : input_info) {
                return_value[item.get_any_name()].push_back(input_value);
            }
        }
        search_string = search_string.substr(end_pos + 1);
        if (search_string.empty() || (search_string.front() != ',' && search_string.front() != '['))
            break;
        if (search_string.front() == ',') {
            if (search_string.length() > 1)
                search_string = search_string.substr(1);
            else
                throw std::logic_error("Can't parse input parameter string, there is nothing after the comma " +
                                       parameter_string);
        }
        start_pos = search_string.find_first_of('[');
    }
    if (!search_string.empty())
        throw std::logic_error("Can't parse input parameter string: " + parameter_string);
    return return_value;
}
