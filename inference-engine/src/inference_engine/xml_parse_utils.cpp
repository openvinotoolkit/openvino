// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xml_parse_utils.h"
#include "details/ie_exception.hpp"
#include "ie_precision.hpp"
#include <string>
#include <limits>

int XMLParseUtils::GetIntAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    int int_value = std::stoi(str_value, &idx, 10);
    if (idx != str_value.length())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
                           << "\" which is not an integer" << " at offset "
                           << node.offset_debug();
    return int_value;
}

uint64_t XMLParseUtils::GetUInt64Attr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length() || int_value < 0 || int_value > (std::numeric_limits<uint64_t>::max)())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
                           << "\" which is not an unsigned 64 bit integer" << " at offset "
                           << node.offset_debug();
    return static_cast<uint64_t>(int_value);
}

unsigned int XMLParseUtils::GetUIntAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    long long int_value = std::stoll(str_value, &idx, 10);
    if (idx != str_value.length() || int_value < 0 || int_value > (std::numeric_limits<unsigned int>::max)())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
                           << "\" which is not an unsigned integer" << " at offset "
                           << node.offset_debug();
    return static_cast<unsigned int>(int_value);
}

std::string XMLParseUtils::GetStrAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return attr.value();
}

std::string XMLParseUtils::GetStrAttr(const pugi::xml_node &node, const char *str, const char *def) {
    auto attr = node.attribute(str);
    if (attr.empty()) return def;
    return attr.value();
}

float XMLParseUtils::GetFloatAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    std::string str_value = std::string(attr.value());
    std::size_t idx = 0;
    float float_value = std::stof(str_value, &idx);
    if (idx != str_value.length())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has attribute \"" << str << "\" = \"" << str_value
                           << "\" which is not a floating point" << " at offset "
                           << node.offset_debug();
    return float_value;
}

InferenceEngine::Precision XMLParseUtils::GetPrecisionAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return InferenceEngine::Precision::FromStr(attr.value());
}

InferenceEngine::Precision XMLParseUtils::GetPrecisionAttr(const pugi::xml_node &node, const char *str,
                                                           InferenceEngine::Precision def) {
    auto attr = node.attribute(str);
    if (attr.empty()) return InferenceEngine::Precision(def);
    return InferenceEngine::Precision::FromStr(attr.value());
}

int XMLParseUtils::GetIntAttr(const pugi::xml_node &node, const char *str, int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return GetIntAttr(node, str);
}

uint64_t XMLParseUtils::GetUInt64Attr(const pugi::xml_node &node, const char *str, uint64_t defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return GetUInt64Attr(node, str);
}

unsigned int XMLParseUtils::GetUIntAttr(const pugi::xml_node &node, const char *str, unsigned int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return GetUIntAttr(node, str);
}

float XMLParseUtils::GetFloatAttr(const pugi::xml_node &node, const char *str, float defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return GetFloatAttr(node, str);
}

int XMLParseUtils::GetIntChild(const pugi::xml_node &node, const char *str, int defVal) {
    auto child = node.child(str);
    if (child.empty()) return defVal;
    return atoi(child.child_value());
}

std::string XMLParseUtils::NameFromFilePath(const char *filepath) {
    std::string baseName = filepath;
    auto slashPos = baseName.rfind('/');
    slashPos = slashPos == std::string::npos ? 0 : slashPos + 1;
    auto dotPos = baseName.rfind('.');
    if (dotPos != std::string::npos) {
        baseName = baseName.substr(slashPos, dotPos - slashPos);
    } else {
        baseName = baseName.substr(slashPos);
    }
    return baseName;
}

