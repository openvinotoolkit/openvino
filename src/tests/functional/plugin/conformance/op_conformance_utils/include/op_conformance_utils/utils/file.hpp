// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <regex>

namespace ov {
namespace util {

std::vector<std::string>
get_filelist_recursive(const std::vector<std::string>& dir_paths,
                       const std::vector<std::regex>& patterns);

std::vector<std::string>
read_lst_file(const std::vector<std::string>& file_paths,
              const std::vector<std::regex>& patterns = {std::regex(".*")});

std::string replace_extension(std::string file, const std::string& new_extension);

inline void remove_path(const std::string& path) {
    if (!path.empty()) {
        std::remove(path.c_str());
    }
}

std::vector<std::string>
split_str(std::string paths, const char delimiter = ',');

}  // namespace util
}  // namespace ov
