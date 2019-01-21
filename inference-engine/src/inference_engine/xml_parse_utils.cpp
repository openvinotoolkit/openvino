// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "xml_parse_utils.h"
#include "details/ie_exception.hpp"
#include "ie_precision.hpp"

int XMLParseUtils::GetIntAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    return atoi(attr.value());
}

uint64_t XMLParseUtils::GetUInt64Attr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    int64_t value = atoll(attr.value());
    if (value < 0)
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has incorrect parameter: " << str << " at offset "
                           << node.offset_debug();
    return static_cast<uint64_t>(value);
}

unsigned int XMLParseUtils::GetUIntAttr(const pugi::xml_node &node, const char *str) {
    auto attr = node.attribute(str);
    if (attr.empty())
        THROW_IE_EXCEPTION << "node <" << node.name() << "> is missing mandatory attribute: " << str << " at offset "
                           << node.offset_debug();
    int value = atoi(attr.value());
    if (value < 0)
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has incorrect parameter: " << str << " at offset "
                           << node.offset_debug();
    return static_cast<unsigned int>(value);
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
    return static_cast<float>(atof(attr.value()));
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
    return atoi(attr.value());
}

uint64_t XMLParseUtils::GetUInt64Attr(const pugi::xml_node &node, const char *str, uint64_t defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    int64_t value = atoll(attr.value());
    if (value < 0)
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has incorrect parameter: " << str << " at offset "
                           << node.offset_debug();
    return static_cast<uint64_t>(value);
}

unsigned int XMLParseUtils::GetUIntAttr(const pugi::xml_node &node, const char *str, unsigned int defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    int value = atoi(attr.value());
    if (value < 0)
        THROW_IE_EXCEPTION << "node <" << node.name() << "> has incorrect parameter: " << str << " at offset "
                           << node.offset_debug();
    return static_cast<unsigned int>(value);
}

float XMLParseUtils::GetFloatAttr(const pugi::xml_node &node, const char *str, float defVal) {
    auto attr = node.attribute(str);
    if (attr.empty()) return defVal;
    return static_cast<float>(atof(attr.value()));
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

