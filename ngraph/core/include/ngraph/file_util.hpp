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

        /// \brief Returns the size in bytes of filename
        /// \param filename The name of the file
        NGRAPH_API
        size_t get_file_size(const std::string& filename);

        /// \brief Removes all files and directories starting at dir
        /// \param dir The path of the directory to remove
        NGRAPH_API
        void remove_directory(const std::string& dir);

        /// \brief Create a directory
        /// \param dir Path of the directory to create
        /// \return true if the directory was created, false otherwise
        NGRAPH_API
        bool make_directory(const std::string& dir);

        /// \brief Gets the path of the system temporary directory
        /// \return the path to the system temporary directory
        NGRAPH_API
        std::string get_temp_directory_path();

        /// \brief Removes a file from the filesystem
        /// \param file The path to the file to be removed
        NGRAPH_API
        void remove_file(const std::string& file);

        /// \brief Reads the contents of a file
        /// \param path The path of the file to read
        /// \return vector<char> of the file's contents
        NGRAPH_API
        std::vector<char> read_file_contents(const std::string& path);

        /// \brief Reads the contents of a file
        /// \param path The path of the file to read
        /// \return string of the file's contents
        NGRAPH_API
        std::string read_file_to_string(const std::string& path);

        /// \brief Iterate through files and optionally directories. Symbolic links are skipped.
        /// \param path The path to iterate over
        /// \param func A callback function called with each file or directory encountered
        /// \param recurse Optional parameter to enable recursing through path
        NGRAPH_API
        void iterate_files(const std::string& path,
                           std::function<void(const std::string& file, bool is_dir)> func,
                           bool recurse = false,
                           bool include_links = false);

        /// \brief Create a temporary file
        /// \param extension Optional extension for the temporary file
        /// \return Name of the temporary file
        NGRAPH_API
        std::string tmp_filename(const std::string& extension = "");

        /// \brief Test for the existence of a path or file
        /// \param path The path to test
        /// \return true if the path exists, false otherwise
        NGRAPH_API
        bool exists(const std::string& path);
    }
}
