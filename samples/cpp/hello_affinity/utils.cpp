// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string trim(std::string value) {
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    value.erase(std::find_if(value.rbegin(),
                             value.rend(),
                             [](unsigned char ch) {
                                 return !std::isspace(ch);
                             })
                    .base(),
                value.end());
    return value;
}

std::string resolve_input_tensor_name(const std::string& name, const std::vector<ov::Output<const ov::Node>>& inputs) {
    for (const auto& input : inputs) {
        if (input.get_any_name() == name || input.get_node_shared_ptr()->get_friendly_name() == name ||
            input.get_node_shared_ptr()->get_name() == name) {
            return input.get_any_name();
        }
    }

    OPENVINO_THROW("Cannot find model input named '", name, "'");
}

}  // namespace

std::string to_lower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains_substring(const std::string& value, const std::string& substring) {
    return to_lower(value).find(to_lower(substring)) != std::string::npos;
}

std::string format_double(double value) {
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(2) << value;
    return stream.str();
}

std::string format_duration_ms(double value) {
    return format_double(value);
}

std::string partial_shapes_to_string(const std::map<std::string, ov::PartialShape>& shapes) {
    std::stringstream stream;
    bool first_shape = true;
    for (const auto& item : shapes) {
        if (!first_shape) {
            stream << ", ";
        }
        first_shape = false;
        stream << "'" << item.first << "': " << item.second;
    }
    return stream.str();
}

std::map<std::string, ov::PartialShape> parse_input_shapes(const std::string& shapes_string,
                                                           const std::vector<ov::Output<const ov::Node>>& inputs) {
    std::map<std::string, ov::PartialShape> shapes;
    std::string remaining = shapes_string;
    while (!remaining.empty()) {
        const auto open_bracket = remaining.find('[');
        const auto close_bracket = remaining.find(']', open_bracket);
        if (open_bracket == std::string::npos || close_bracket == std::string::npos) {
            OPENVINO_THROW("Cannot parse shape string: ", shapes_string);
        }

        const auto name = trim(remaining.substr(0, open_bracket));
        const auto shape = remaining.substr(open_bracket, close_bracket - open_bracket + 1);
        if (name.empty()) {
            for (const auto& input : inputs) {
                shapes[input.get_any_name()] = ov::PartialShape(shape);
            }
        } else {
            shapes[resolve_input_tensor_name(name, inputs)] = ov::PartialShape(shape);
        }

        remaining = trim(remaining.substr(close_bracket + 1));
        if (!remaining.empty()) {
            if (remaining.front() == '[') {
                OPENVINO_THROW("Multiple shape groups for the same input are not supported: ", shapes_string);
            }
            if (remaining.front() != ',') {
                OPENVINO_THROW("Cannot parse shape string: ", shapes_string);
            }
            remaining = trim(remaining.substr(1));
        }
    }

    return shapes;
}