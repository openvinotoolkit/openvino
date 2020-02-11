// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdlib>
#include <details/os/os_filesystem.hpp>
#include <fstream>
#include <ie_precision.hpp>
#include <memory>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "ie_api.h"
#include "ie_common.h"

#define FOREACH_CHILD(c, p, tag) for (auto c = p.child(tag); !c.empty(); c = c.next_sibling(tag))

namespace XMLParseUtils {

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node& node, const char* str, int defVal);

INFERENCE_ENGINE_API_CPP(int64_t) GetInt64Attr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(int64_t) GetInt64Attr(const pugi::xml_node& node, const char* str, int64_t defVal);

INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node& node, const char* str, uint64_t defVal);

INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node& node, const char* str, unsigned int defVal);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node& node, const char* str, const char* def);

INFERENCE_ENGINE_API_CPP(bool) GetBoolAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(bool) GetBoolAttr(const pugi::xml_node& node, const char* str, const bool def);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node& node, const char* str, float defVal);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision) GetPrecisionAttr(const pugi::xml_node& node, const char* str);

INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision)
GetPrecisionAttr(const pugi::xml_node& node, const char* str, InferenceEngine::Precision def);

INFERENCE_ENGINE_API_CPP(int) GetIntChild(const pugi::xml_node& node, const char* str, int defVal);

}  // namespace XMLParseUtils

struct parse_result {
    parse_result(std::unique_ptr<pugi::xml_document>&& xml, std::string error_msg)
        : xml(std::move(xml)), error_msg(std::move(error_msg)) {}

    // have to use ptr because xml_document non-copyable/non-movable
    std::unique_ptr<pugi::xml_document> xml;
    std::string error_msg {};
};

static parse_result ParseXml(const char* file_path) {
#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring wFilePath = InferenceEngine::details::multiByteCharToWString(file_path);
    const wchar_t* resolvedFilepath = wFilePath.c_str();
#else
    const char* resolvedFilepath = file_path;
#endif

    auto xml = std::unique_ptr<pugi::xml_document> {new pugi::xml_document {}};
    const auto load_result = xml->load_file(resolvedFilepath);

    const auto error_msg = [&]() -> std::string {
        if (load_result.status == pugi::status_ok) return {};

        std::ifstream file_stream(file_path);
        const auto file = std::string(std::istreambuf_iterator<char> {file_stream}, std::istreambuf_iterator<char> {});

        const auto error_offset = std::next(file.rbegin(), file.size() - load_result.offset);
        const auto line_begin = std::find(error_offset, file.rend(), '\n');
        const auto line = 1 + std::count(line_begin, file.rend(), '\n');
        const auto pos = std::distance(error_offset, line_begin);

        std::stringstream ss;
        ss << "Error loading XML file: " << file_path << ":" << line << ":" << pos << ": " << load_result.description();
        return ss.str();
    }();

    return {std::move(xml), error_msg};
}