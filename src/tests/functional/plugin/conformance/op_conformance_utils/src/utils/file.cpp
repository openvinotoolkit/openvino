// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_conformance_utils/utils/file.hpp"

#include <set>
#include <cstring>

#include "openvino/util/file_util.hpp"

namespace ov {
namespace util {

std::vector<std::string>
get_filelist_recursive(const std::vector<std::string>& dir_paths,
                       const std::vector<std::regex>& patterns) {
    std::vector<std::string> result;
    for (auto&& dir_path : dir_paths) {
        if (!ov::util::directory_exists(dir_path)) {
            std::string msg = "Input directory (" + dir_path + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        ov::util::iterate_files(
            dir_path,
            [&result, &patterns](const std::string& file_path, bool is_dir) {
                auto file = ov::util::get_file_name(file_path);
                if (ov::util::file_exists(file_path)) {
                    for (const auto& pattern : patterns) {
                        if (std::regex_match(file_path, pattern)) {
                            result.push_back(file_path);
                            break;
                        }
                    }
                }
            },
            true,
            false);
    }
    return result;
}

std::vector<std::string>
read_lst_file(const std::vector<std::string>& file_paths,
              const std::vector<std::regex>& patterns) {
    std::vector<std::string> res;
    for (const auto& file_path : file_paths) {
        if (!ov::util::file_exists(file_path)) {
            std::string msg = "Input directory (" + file_path + ") doesn't not exist!";
            throw std::runtime_error(msg);
        }
        std::ifstream file(file_path);
        if (file.is_open()) {
            std::string buffer;
            while (getline(file, buffer)) {
                if (buffer.find("#") == std::string::npos && !buffer.empty()) {
                    for (const auto& pattern : patterns) {
                        if (std::regex_match(file_path, pattern)) {
                            res.emplace_back(buffer);
                            break;
                        }
                    }
                }
            }
        } else {
            std::string msg = "Error in opening file: " + file_path;
            throw std::runtime_error(msg);
        }
        file.close();
    }
    return res;
}

std::string replace_extension(std::string file, const std::string& new_extension) {
    std::string::size_type pos = file.rfind('.', file.length());
    if (pos != std::string::npos) {
        if (new_extension == "") {
            file = file.substr(0, pos);
        } else {
            file.replace(pos + 1, new_extension.length(), new_extension);
        }
    }
    return file;
}

std::vector<std::string>
split_str(std::string paths, const char delimiter) {
    size_t delimiterPos;
    std::vector<std::string> splitPath;
    while ((delimiterPos = paths.find(delimiter)) != std::string::npos) {
        splitPath.push_back(paths.substr(0, delimiterPos));
        paths = paths.substr(delimiterPos + 1);
    }
    splitPath.push_back(paths);
    return splitPath;
}

}  // namespace util
}  // namespace ov