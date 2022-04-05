// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/file_util.hpp"

#include "openvino/util/file_util.hpp"

NGRAPH_SUPPRESS_DEPRECATED_START

using namespace std;
using namespace ngraph;

string file_util::get_file_name(const string& s) {
    return ov::util::get_file_name(s);
}

string file_util::get_file_ext(const string& s) {
    return ov::util::get_file_ext(s);
}

string file_util::get_directory(const string& s) {
    return ov::util::get_directory(s);
}

string file_util::path_join(const string& s1, const string& s2, const string& s3) {
    return ov::util::path_join({s1, s2, s3});
}

string file_util::path_join(const string& s1, const string& s2, const string& s3, const string& s4) {
    return ov::util::path_join({s1, s2, s3, s4});
}

string file_util::path_join(const string& s1, const string& s2) {
    return ov::util::path_join({s1, s2});
}

void file_util::iterate_files(const string& path,
                              function<void(const string& file, bool is_dir)> func,
                              bool recurse,
                              bool include_links) {
    ov::util::iterate_files(path, func, recurse, include_links);
}

std::string file_util::sanitize_path(const std::string& path) {
    return ov::util::sanitize_path(path);
}

void file_util::convert_path_win_style(std::string& path) {
    ov::util::convert_path_win_style(path);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::string file_util::wstring_to_string(const std::wstring& wstr) {
    return ov::util::wstring_to_string(wstr);
}

std::wstring file_util::multi_byte_char_to_wstring(const char* str) {
    return ov::util::string_to_wstring(str);
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
