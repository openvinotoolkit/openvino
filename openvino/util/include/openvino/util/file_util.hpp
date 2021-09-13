// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>

namespace ov {
namespace util {
std::string get_file_name(const std::string& path);
std::string get_file_ext(const std::string& path);
std::string get_directory(const std::string& path);
std::string path_join(const std::vector<std::string>& paths);

void iterate_files(const std::string& path,
                   const std::function<void(const std::string& file, bool is_dir)>& func,
                   bool recurse = false,
                   bool include_links = false);

void convert_path_win_style(std::string& path);

std::string wstring_to_string(const std::wstring& wstr);
std::wstring multi_byte_char_to_wstring(const char* str);
std::string sanitize_path(const std::string& path);
}  // namespace util
}  // namespace ov
