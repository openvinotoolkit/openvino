// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/xml_parse_utils.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <string>

namespace ov {
namespace util {

int pugixml::get_int_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }

    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    int int_value = std::stoi(str_value, &idx, 10);
    if (idx != str_value.length()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
           << "\" which is not an integer"
           << " at offset " << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return int_value;
}

int64_t pugixml::get_int64_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
           << "\" which is not a signed 64 bit integer"
           << " at offset " << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return static_cast<int64_t>(int_value);
}

uint64_t pugixml::get_uint64_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length() || int_value < 0) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
           << "\" which is not an unsigned 64 bit integer"
           << " at offset " << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return static_cast<uint64_t>(int_value);
}

unsigned int pugixml::get_uint_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length() || int_value < 0 || int_value > (std::numeric_limits<unsigned int>::max)()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
           << "\" which is not an unsigned integer"
           << " at offset " << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return static_cast<unsigned int>(int_value);
}

std::string pugixml::get_str_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return attr.value();
}

std::string pugixml::get_str_attr(const pugi::xml_node& node, const char* str, const char* def) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        if (def != nullptr)
            return def;
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return attr.value();
}

bool pugixml::get_bool_attr(const pugi::xml_node& node, const char* str, const bool def) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return def;
    std::string string_attr = attr.value();
    std::transform(string_attr.begin(), string_attr.end(), string_attr.begin(), [](char ch) {
        return std::tolower(static_cast<unsigned char>(ch));
    });
    std::set<std::string> true_names{"true", "1"};
    std::set<std::string> false_names{"false", "0"};

    bool is_true = true_names.find(string_attr) != true_names.end();
    bool is_false = false_names.find(string_attr) != false_names.end();

    if (!is_true && !is_false) {
        std::stringstream ss;
        ss << "Unsupported boolean attribute type: " << string_attr;
        throw std::runtime_error(ss.str());
    }

    return is_true;
}

bool pugixml::get_bool_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    std::string string_attr = attr.value();
    std::transform(string_attr.begin(), string_attr.end(), string_attr.begin(), [](char ch) {
        return std::tolower(static_cast<unsigned char>(ch));
    });
    std::set<std::string> true_names{"true", "1"};
    std::set<std::string> false_names{"false", "0"};

    bool is_true = true_names.find(string_attr) != true_names.end();
    bool is_false = false_names.find(string_attr) != false_names.end();

    if (!is_true && !is_false) {
        std::stringstream ss;
        ss << "Unsupported boolean attribute type: " << string_attr;
        throw std::runtime_error(ss.str());
    }

    return is_true;
}

float pugixml::get_float_attr(const pugi::xml_node& node, const char* str) {
    auto attr = node.attribute(str);
    if (attr.empty()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
           << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    std::string str_value = std::string(attr.value());
    std::stringstream str_stream(str_value);
    str_stream.imbue(std::locale("C"));
    float float_value;
    str_stream >> float_value;
    if (!str_stream.eof()) {
        std::stringstream ss;
        ss << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
           << "\" which is not a floating point"
           << " at offset " << node.offset_debug();
        throw std::runtime_error(ss.str());
    }
    return float_value;
}

int pugixml::get_int_attr(const pugi::xml_node& node, const char* str, int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return defVal;
    return get_int_attr(node, str);
}

int64_t pugixml::get_int64_attr(const pugi::xml_node& node, const char* str, int64_t defVal) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return defVal;
    return get_int64_attr(node, str);
}

uint64_t pugixml::get_uint64_attr(const pugi::xml_node& node, const char* str, uint64_t defVal) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return defVal;
    return get_uint64_attr(node, str);
}

unsigned int pugixml::get_uint_attr(const pugi::xml_node& node, const char* str, unsigned int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return defVal;
    return get_uint_attr(node, str);
}

float pugixml::get_float_attr(const pugi::xml_node& node, const char* str, float defVal) {
    auto attr = node.attribute(str);
    if (attr.empty())
        return defVal;
    return get_float_attr(node, str);
}

int pugixml::get_int_child(const pugi::xml_node& node, const char* str, int defVal) {
    auto child = node.child(str);
    if (child.empty())
        return defVal;
    return atoi(child.child_value());
}

}  // namespace util
}  // namespace ov