// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic functions to safely extract values from `pugi::xml_node` and open `pugi::xml_document`
 * @file xml_parse_utils.h
 */

#pragma once

#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include <pugixml.hpp>

#include "ie_api.h"
#include "ie_precision.hpp"
#include "ie_common.h"
#include "file_utils.h"

/**
 * @ingroup    ie_dev_api_xml
 * @brief      Defines convinient for-each based cycle to iterate over node children
 *
 * @param      c     Child node name
 * @param      p     Parent node name
 * @param      tag   The tag represented as a string value
 */
#define FOREACH_CHILD(c, p, tag) for (auto c = p.child(tag); !c.empty(); c = c.next_sibling(tag))

/**
 * @brief XML helpers function to extract values from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 */
namespace XMLParseUtils {

/**
 * @brief      Gets the integer attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string
 * @return     An integer value
 */
INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the integer attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An integer value
 */
INFERENCE_ENGINE_API_CPP(int) GetIntAttr(const pugi::xml_node& node, const char* str, int defVal);

/**
 * @brief      Gets the `int64_t` attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An `int64_t` value
 */
INFERENCE_ENGINE_API_CPP(int64_t) GetInt64Attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the `int64_t` attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An `int64_t` value
 */
INFERENCE_ENGINE_API_CPP(int64_t) GetInt64Attr(const pugi::xml_node& node, const char* str, int64_t defVal);

/**
 * @brief      Gets the `uint64_t` attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An `uint64_t` value
 */
INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the `uint64_t` attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An `uint64_t` value
 */
INFERENCE_ENGINE_API_CPP(uint64_t) GetUInt64Attr(const pugi::xml_node& node, const char* str, uint64_t defVal);

/**
 * @brief      Gets the unsigned integer attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An unsigned integer value
 */
INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the unsigned integer attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An unsigned integer value
 */
INFERENCE_ENGINE_API_CPP(unsigned int) GetUIntAttr(const pugi::xml_node& node, const char* str, unsigned int defVal);

/**
 * @brief      Gets the string attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A string value
 */
INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the string attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @param[in]  def   The default value
 * @return     A string value
 */
INFERENCE_ENGINE_API_CPP(std::string) GetStrAttr(const pugi::xml_node& node, const char* str, const char* def);

/**
 * @brief      Gets the bool attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A boolean value
 */
INFERENCE_ENGINE_API_CPP(bool) GetBoolAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the bool attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @param[in]  def   The default value
 * @return     A boolean value
 */
INFERENCE_ENGINE_API_CPP(bool) GetBoolAttr(const pugi::xml_node& node, const char* str, const bool def);

/**
 * @brief      Gets the float attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A single-precision floating point value
 */
INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the float attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     A single-precision floating point value
 */
INFERENCE_ENGINE_API_CPP(float) GetFloatAttr(const pugi::xml_node& node, const char* str, float defVal);

/**
 * @brief      Gets the Precision attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A Precision value
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision) GetPrecisionAttr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the Precision attribute from `pugi::xml_node`
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @param[in]  def  The default value
 * @return     A Precision value
 */
INFERENCE_ENGINE_API_CPP(InferenceEngine::Precision)
GetPrecisionAttr(const pugi::xml_node& node, const char* str, InferenceEngine::Precision def);

/**
 * @brief      Gets the integer value located in a child node.
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  node    The node
 * @param[in]  str     The string value identifying a child node
 * @param[in]  defVal  The default value
 * @return     An ingeter value located in a child node, @p devVal otherwise.
 */
INFERENCE_ENGINE_API_CPP(int) GetIntChild(const pugi::xml_node& node, const char* str, int defVal);

}  // namespace XMLParseUtils

/**
 * @brief      A XML parse result structure with an error message and the `pugi::xml_document` document.
 * @ingroup    ie_dev_api_xml
 */
struct parse_result {
    /**
     * @brief      Constructs parse_result with `pugi::xml_document` and an error message
     *
     * @param      xml        The `pugi::xml_document`
     * @param[in]  error_msg  The error message
     */
    parse_result(std::unique_ptr<pugi::xml_document>&& xml, std::string error_msg)
        : xml(std::move(xml)), error_msg(std::move(error_msg)) {}

    /**
     * @brief A XML document. 
     */
    std::unique_ptr<pugi::xml_document> xml;

    /**
     * @brief An error message
     */
    std::string error_msg {};
};

/**
 * @brief      Parses a file and returns parse_result
 * @ingroup    ie_dev_api_xml
 *
 * @param[in]  file_path  The file path
 *
 * @return     The parse_result.
 */
inline parse_result ParseXml(const char* file_path) {
#ifdef ENABLE_UNICODE_PATH_SUPPORT
    std::wstring wFilePath = FileUtils::multiByteCharToWString(file_path);
    const wchar_t* resolvedFilepath = wFilePath.c_str();
#else
    const char* resolvedFilepath = file_path;
#endif

    try {
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
    } catch(std::exception& e) {
        return {std::move(nullptr), std::string("Error loading XML file: ") + e.what()};
    }
}
