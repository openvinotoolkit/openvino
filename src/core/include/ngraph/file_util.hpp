// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(NGRAPH_LEGACY_HEADER_INCLUDED)
#    define NGRAPH_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The nGraph API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

#include <functional>
#include <string>
#include <vector>

#include "ngraph/deprecated.hpp"
#include "ngraph/ngraph_visibility.hpp"

namespace ngraph {
namespace file_util {
/// \brief Returns the name with extension for a given path
/// \param path The path to the output file
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string get_file_name(const std::string& path);

/// \brief Returns the file extension
/// \param path The path to the output file
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string get_file_ext(const std::string& path);

/// \brief Returns the directory portion of the given path
/// \param path The path to the output file
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string get_directory(const std::string& path);

/// \brief Joins multiple paths into a single path
/// \param s1 Left side of path
/// \param s2 Right side of path
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string path_join(const std::string& s1, const std::string& s2);
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string path_join(const std::string& s1, const std::string& s2, const std::string& s3);
NGRAPH_API_DEPRECATED
NGRAPH_API
std::string path_join(const std::string& s1, const std::string& s2, const std::string& s3, const std::string& s4);

/// \brief Iterate through files and optionally directories. Symbolic links are skipped.
/// \param path The path to iterate over
/// \param func A callback function called with each file or directory encountered
/// \param recurse Optional parameter to enable recursing through path
NGRAPH_API_DEPRECATED
NGRAPH_API
void iterate_files(const std::string& path,
                   std::function<void(const std::string& file, bool is_dir)> func,
                   bool recurse = false,
                   bool include_links = false);

/// \brief Change Linux-style path ('/') to Windows-style ('\\')
/// \param path The path to change file separator
NGRAPH_API_DEPRECATED
NGRAPH_API void convert_path_win_style(std::string& path);

/// \brief Conversion from wide character string to a single-byte chain.
/// \param wstr A wide-char string
/// \return A multi-byte string
NGRAPH_API_DEPRECATED
NGRAPH_API std::string wstring_to_string(const std::wstring& wstr);

/// \brief Conversion from single-byte chain to wide character string.
/// \param str A null-terminated string
/// \return A wide-char string
NGRAPH_API_DEPRECATED
NGRAPH_API std::wstring multi_byte_char_to_wstring(const char* str);

/// \brief Remove path components which would allow traversing up a directory tree.
/// \param path A path to file
/// \return A sanitiazed path
NGRAPH_API_DEPRECATED
NGRAPH_API std::string sanitize_path(const std::string& path);
}  // namespace file_util
}  // namespace ngraph
