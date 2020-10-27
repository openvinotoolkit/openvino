//*****************************************************************************
// Copyright 2017-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include <functional>
#include <string>
#include <vector>

#include <ngraph/ngraph_visibility.hpp>

namespace ngraph
{
    namespace file_util
    {
        /// \brief Returns the name with extension for a given path
        /// \param path The path to the output file
        NGRAPH_API
        std::string get_file_name(const std::string& path);

        /// \brief Returns the file extension
        /// \param path The path to the output file
        NGRAPH_API
        std::string get_file_ext(const std::string& path);

        /// \brief Returns the directory portion of the given path
        /// \param path The path to the output file
        NGRAPH_API
        std::string get_directory(const std::string& path);

        /// \brief Joins multiple paths into a single path
        /// \param s1 Left side of path
        /// \param s2 Right side of path
        NGRAPH_API
        std::string path_join(const std::string& s1, const std::string& s2);
        NGRAPH_API
        std::string path_join(const std::string& s1, const std::string& s2, const std::string& s3);
        NGRAPH_API
        std::string path_join(const std::string& s1,
                              const std::string& s2,
                              const std::string& s3,
                              const std::string& s4);

        /// \brief Iterate through files and optionally directories. Symbolic links are skipped.
        /// \param path The path to iterate over
        /// \param func A callback function called with each file or directory encountered
        /// \param recurse Optional parameter to enable recursing through path
        NGRAPH_API
        void iterate_files(const std::string& path,
                           std::function<void(const std::string& file, bool is_dir)> func,
                           bool recurse = false,
                           bool include_links = false);

        /// \brief Change Linux-style path ('/') to Windows-style ('\\')
        /// \param path The path to change file separator
        NGRAPH_API void convert_path_win_style(std::string& path);

        /// \brief Conversion from wide character string to a single-byte chain.
        /// \param wstr A wide-char string
        /// \return A multi-byte string
        NGRAPH_API std::string wstring_to_string(const std::wstring& wstr);

        /// \brief Conversion from single-byte chain to wide character string.
        /// \param str A null-terminated string
        /// \return A wide-char string
        NGRAPH_API std::wstring multi_byte_char_to_wstring(const char* str);
    }
}
