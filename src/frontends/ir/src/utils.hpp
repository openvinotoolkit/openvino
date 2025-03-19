// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <pugixml.hpp>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
void operator>>(const std::stringstream& in, ov::element::Type& type);

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value);
bool get_dimension_from_attribute(const pugi::xml_node& node, const std::string& name, Dimension& value);
bool get_partial_shape_from_attribute(const pugi::xml_node& node, const std::string& name, PartialShape& value);

void str_to_container(const std::string& value, std::vector<std::string>& res);

template <class T>
void str_to_container(const std::string& value, T& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            OPENVINO_THROW("Cannot get vector of parameters! \"", value, "\" is incorrect");
        std::stringstream fs(field);
        typename T::value_type val;
        fs >> val;
        res.insert(res.end(), val);
    }
}

// separated function for set<string> to keep whitespaces in values
// because stringstream splits its values with whitespace delimiter
void str_to_set_of_strings(const std::string& value, std::set<std::string>& res);

template <class T>
bool getParameters(const pugi::xml_node& node, const std::string& name, std::vector<T>& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    str_to_container(param, value);
    return true;
}

template <class T>
T stringToType(const std::string& valStr) {
    T ret{0};
    std::istringstream ss(valStr);
    if (!ss.eof()) {
        ss >> ret;
    }
    return ret;
}
}  // namespace ov
