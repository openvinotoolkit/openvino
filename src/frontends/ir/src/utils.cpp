// Copyright (C) 2018-2022 Intel Corporation
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

bool check_all_digits(const std::string& value) {
    auto val = ov::util::trim(value);
    for (const auto& c : val) {
        if (!std::isdigit(c) || c == '-')
            return false;
    }
    return true;
}

Dimension str_to_dimension(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val == "?" || val == "-1") {
        return {-1};
    }
    if (val.find("..") == std::string::npos) {
        if (!check_all_digits(val))
            IE_THROW() << "Cannot parse dimension: \"" << val << "\"";
        return {stringToType<int64_t>(val)};
    }

    std::string min_value_str = val.substr(0, val.find(".."));
    if (!check_all_digits(min_value_str))
        IE_THROW() << "Cannot parse min bound: \"" << min_value_str << "\"";

    int64_t min_value;
    if (min_value_str.empty())
        min_value = 0;
    else
        min_value = stringToType<int64_t>(min_value_str);

    std::string max_value_str = val.substr(val.find("..") + 2);
    int64_t max_value;
    if (max_value_str.empty())
        max_value = -1;
    else
        max_value = stringToType<int64_t>(max_value_str);

    if (!check_all_digits(max_value_str))
        IE_THROW() << "Cannot parse max bound: \"" << max_value_str << "\"";

    return {min_value, max_value};
}

PartialShape str_to_partial_shape(const std::string& value) {
    auto val = ov::util::trim(value);
    if (val == "...") {
        return PartialShape::dynamic();
    }
    PartialShape res;
    std::stringstream ss(val);
    std::string field;
    while (getline(ss, field, ',')) {
        if (field.empty())
            IE_THROW() << "Cannot get vector of dimensions! \"" << val << "\" is incorrect";
        res.insert(res.end(), str_to_dimension(field));
    }
    return res;
}

bool get_partial_shape_from_attribute(const pugi::xml_node& node, const std::string& name, PartialShape& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = str_to_partial_shape(param);
    return true;
}

bool get_dimension_from_attribute(const pugi::xml_node& node, const std::string& name, Dimension& value) {
    std::string param;
    if (!getStrAttribute(node, name, param))
        return false;
    value = str_to_dimension(param);
    return true;
}

void str_to_set_of_strings(const std::string& value, std::set<std::string>& res) {
    std::stringstream ss(value);
    std::string field;
    while (getline(ss, field, ',')) {
        // trim leading and trailing whitespaces
        auto strBegin = field.find_first_not_of(" ");
        if (strBegin == std::string::npos)
            IE_THROW() << "Cannot get a set of strings from \"" << value << "\". Value \"" << field
                       << "\" is incorrect";
        auto strRange = field.find_last_not_of(" ") - strBegin + 1;

        res.insert(field.substr(strBegin, strRange));
    }
}

}  // namespace ov
