// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <xml_parse_utils.h>

#include <ie_ngraph_utils.hpp>
#include <istream>
#include <memory>
#include <ngraph/ngraph.hpp>
#include <pugixml.hpp>

namespace ov {
void operator>>(const std::stringstream& in, ngraph::element::Type& type);

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value);

template <class T>
void str_to_container(const std::string& value, T& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            IE_THROW() << "Cannot get vector of parameters! \"" << value << "\" is incorrect";
        std::stringstream fs(field);
        typename T::value_type val;
        fs >> val;
        res.insert(res.end(), val);
    }
}

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