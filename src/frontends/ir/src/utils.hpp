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
