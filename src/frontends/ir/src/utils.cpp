// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "ie_ngraph_utils.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
void operator>>(const std::stringstream& in, ov::element::Type& type) {
    type = InferenceEngine::details::convertPrecision(ov::util::trim(in.str()));
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

Dimension strToDimension(const std::string& value) {
    if (value.find('?') != std::string::npos) {
        return {-1};
    }
    if (value.find("..") == std::string::npos) {
        return {stringToType<int64_t>(value)};
    }
    std::string min_value_str = value.substr(0, value.find(".."));
    int64_t min_value;
    if (min_value_str.find_first_not_of(' ') == std::string::npos)
        min_value = 0;
    else
        min_value = stringToType<int64_t>(min_value_str);

    std::string max_value_str = value.substr(value.find("..") + 2);
    int64_t max_value;
    if (max_value_str.find_first_not_of(' ') == std::string::npos)
        max_value = -1;
    else
        max_value = stringToType<int64_t>(max_value_str);

    return {min_value, max_value};
}

PartialShape strToPartialShape(const std::string& value) {
    if (value.find("...") != std::string::npos) {
        return PartialShape::dynamic();
    }
    PartialShape res;
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            IE_THROW() << "Cannot get vector of dimensions! \"" << value << "\" is incorrect";
        res.insert(res.end(), strToDimension(field));
    }
    return res;
}

bool getPartialShapeFromAttribute(const pugi::xml_node& node, const std::string& name, PartialShape& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = strToPartialShape(param);
    return true;
}

bool getDimensionFromAttribute(const pugi::xml_node& node, const std::string& name, Dimension& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = strToDimension(param);
    return true;
}

}  // namespace ov
