// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"

#include "openvino/core/type/element_type.hpp"
#include "openvino/util/common_util.hpp"

namespace pugixml {
namespace utils {

std::string get_str_attr(const pugi::xml_node& node, const char* str, const char* def) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        if (def != nullptr)
            return def;

        OPENVINO_THROW("node <",
                       node.name(),
                       "> is missing mandatory attribute: '",
                       str,
                       "' at offset ",
                       node.offset_debug());
    }
    return attr.value();
}

int64_t get_int64_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        OPENVINO_THROW("node <",
                       node.name(),
                       "> is missing mandatory attribute: ",
                       str,
                       " at offset ",
                       node.offset_debug());
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length())
        OPENVINO_THROW("node <",
                       node.name(),
                       "> has attribute \"",
                       str,
                       "\" = \"",
                       str_value,
                       "\" which is not an 64 bit integer",
                       " at offset ",
                       node.offset_debug());
    return static_cast<int64_t>(int_value);
}

int64_t get_int64_attr(const pugi::xml_node& node, const char* str, int64_t def) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return def;
    return get_int64_attr(node, str);
}

uint64_t get_uint64_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        OPENVINO_THROW("node <",
                       node.name(),
                       "> is missing mandatory attribute: ",
                       str,
                       " at offset ",
                       node.offset_debug());
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length() || int_value < 0)
        OPENVINO_THROW("node <",
                       node.name(),
                       "> has attribute \"",
                       str,
                       "\" = \"",
                       str_value,
                       "\" which is not an unsigned 64 bit integer",
                       " at offset ",
                       node.offset_debug());
    return static_cast<uint64_t>(int_value);
}

uint64_t get_uint64_attr(const pugi::xml_node& node, const char* str, uint64_t def) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return def;
    return get_uint64_attr(node, str);
}

float get_float_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        OPENVINO_THROW("node <",
                       node.name(),
                       "> is missing mandatory attribute: ",
                       str,
                       " at offset ",
                       node.offset_debug());
    std::string str_value = std::string(attr.value());
    std::stringstream str_stream(str_value);
    str_stream.imbue(std::locale("C"));
    float float_value;
    str_stream >> float_value;
    if (!str_stream.eof())
        OPENVINO_THROW("node <",
                       node.name(),
                       "> has attribute \"",
                       str,
                       "\" = \"",
                       str_value,
                       "\" which is not a floating point",
                       " at offset ",
                       node.offset_debug());
    return float_value;
}

}  // namespace utils
}  // namespace pugixml

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
