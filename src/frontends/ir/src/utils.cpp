// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {

void operator>>(const std::stringstream& in, ov::element::Type& type) {
    type = ov::element::Type(ov::util::trim(in.str()));
}

bool getStrAttribute(const pugi::xml_node& node, const std::string& name, std::string& value) {
    if (!node)
        return false;

    auto attr = node.attribute(name.c_str());
    if (attr.empty())
        return false;
    value = std::string(attr.value());
    return true;
}

bool get_partial_shape_from_attribute(const pugi::xml_node& node, const std::string& name, PartialShape& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = PartialShape(param);
    return true;
}

bool get_dimension_from_attribute(const pugi::xml_node& node, const std::string& name, Dimension& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = Dimension(param);
    return true;
}

void str_to_set_of_strings(const std::string& value, std::set<std::string>& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        // trim leading and trailing whitespaces
        auto strBegin = field.find_first_not_of(" ");
        if (strBegin == std::string::npos)
            OPENVINO_THROW("Cannot get a set of strings from \"", value, "\". Value \"", field, "\" is incorrect");
        auto strRange = field.find_last_not_of(" ") - strBegin + 1;

        res.insert(field.substr(strBegin, strRange));
    }
}

void str_to_container(const std::string& value, std::vector<std::string>& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        field = ov::util::trim(field);
        if (!field.empty()) {
            res.emplace_back(field);
        }
    }
}

}  // namespace ov
