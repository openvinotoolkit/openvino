// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic functions to safely extract values from `pugi::xml_node` and open `pugi::xml_document`
 * @file xml_parse_utils.hpp
 */

#pragma once
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <pugixml.hpp>
#include <sstream>
#include <string>
#include <utility>

#include "openvino/util/file_util.hpp"

/**
 * @brief      Defines convinient for-each based cycle to iterate over node children
 *
 * @param      c     Child node name
 * @param      p     Parent node name
 * @param      tag   The tag represented as a string value
 */
#define FOREACH_CHILD(c, p, tag) for (auto c = p.child(tag); !c.empty(); c = c.next_sibling(tag))

namespace ov {
namespace util {

/**
 * @brief XML helpers function to extract values from `pugi::xml_node`
 */
namespace pugixml {

/**
 * @brief      Gets the integer attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string
 * @return     An integer value
 */
int get_int_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the integer attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An integer value
 */
int get_int_attr(const pugi::xml_node& node, const char* str, int defVal);

/**
 * @brief      Gets the `int64_t` attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An `int64_t` value
 */
int64_t get_int64_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the `int64_t` attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An `int64_t` value
 */
int64_t get_int64_attr(const pugi::xml_node& node, const char* str, int64_t defVal);

/**
 * @brief      Gets the `uint64_t` attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An `uint64_t` value
 */
uint64_t get_uint64_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the `uint64_t` attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An `uint64_t` value
 */
uint64_t get_uint64_attr(const pugi::xml_node& node, const char* str, uint64_t defVal);

/**
 * @brief      Gets the unsigned integer attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     An unsigned integer value
 */
unsigned int get_uint_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the unsigned integer attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     An unsigned integer value
 */
unsigned int get_uint_attr(const pugi::xml_node& node, const char* str, unsigned int defVal);

/**
 * @brief      Gets the string attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A string value
 */
std::string get_str_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the string attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @param[in]  def   The default value
 * @return     A string value
 */
std::string get_str_attr(const pugi::xml_node& node, const char* str, const char* def);

/**
 * @brief      Gets the bool attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A boolean value
 */
bool get_bool_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the bool attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @param[in]  def   The default value
 * @return     A boolean value
 */
bool get_bool_attr(const pugi::xml_node& node, const char* str, const bool def);

/**
 * @brief      Gets the float attribute from `pugi::xml_node`
 *
 * @param[in]  node  The node
 * @param[in]  str   The string identifying value name
 * @return     A single-precision floating point value
 */
float get_float_attr(const pugi::xml_node& node, const char* str);

/**
 * @brief      Gets the float attribute from `pugi::xml_node`
 *
 * @param[in]  node    The node
 * @param[in]  str     The string identifying value name
 * @param[in]  defVal  The default value
 * @return     A single-precision floating point value
 */
float get_float_attr(const pugi::xml_node& node, const char* str, float defVal);

/**
 * @brief      Gets the integer value located in a child node.
 *
 * @param[in]  node    The node
 * @param[in]  str     The string value identifying a child node
 * @param[in]  defVal  The default value
 * @return     An ingeter value located in a child node, @p devVal otherwise.
 */
int get_int_child(const pugi::xml_node& node, const char* str, int defVal);

/**
 * @brief      A XML parse result structure with an error message and the `pugi::xml_document` document.
 * @ingroup    ov_dev_api_xml
 */
struct ParseResult {
    /**
     * @brief      Constructs ParseResult with `pugi::xml_document` and an error message
     *
     * @param      xml        The `pugi::xml_document`
     * @param[in]  error_msg  The error message
     */
    ParseResult(std::unique_ptr<pugi::xml_document>&& xml, std::string error_msg)
        : xml(std::move(xml)),
          error_msg(std::move(error_msg)) {}

    /**
     * @brief A XML document.
     */
    std::unique_ptr<pugi::xml_document> xml;

    /**
     * @brief An error message
     */
    std::string error_msg{};
};

/**
 * @brief      Parses a file and returns ParseResult
 * @ingroup    ov_dev_api_xml
 *
 * @param[in]  file_path  The file path
 *
 * @return     The ParseResult.
 */
inline ParseResult parse_xml(const char* file_path) {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    std::wstring wFilePath = ov::util::string_to_wstring(file_path);
    const wchar_t* resolvedFilepath = wFilePath.c_str();
#else
    const char* resolvedFilepath = file_path;
#endif

    try {
        auto xml = std::unique_ptr<pugi::xml_document>{new pugi::xml_document{}};
        const auto load_result = xml->load_file(resolvedFilepath);

        const auto error_msg = [&]() -> std::string {
            if (load_result.status == pugi::status_ok)
                return {};

            std::ifstream file_stream(file_path);
            const auto file =
                std::string(std::istreambuf_iterator<char>{file_stream}, std::istreambuf_iterator<char>{});

            const auto error_offset = std::next(file.rbegin(), file.size() - load_result.offset);
            const auto line_begin = std::find(error_offset, file.rend(), '\n');
            const auto line = 1 + std::count(line_begin, file.rend(), '\n');
            const auto pos = std::distance(error_offset, line_begin);

            std::stringstream ss;
            ss << "Error loading XML file: " << file_path << ":" << line << ":" << pos << ": "
               << load_result.description();
            return ss.str();
        }();

        return {std::move(xml), error_msg};
    } catch (std::exception& e) {
        return {std::move(nullptr), std::string("Error loading XML file: ") + e.what()};
    }
}
}  // namespace pugixml
}  // namespace util
}  // namespace ov
