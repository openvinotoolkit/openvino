// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <fstream>
#include <string>
#include <vector>

#include "common_test_utils/common_utils.hpp"
#include "common_test_utils/w_dirent.h"
#include "gtest/gtest.h"
#include "openvino/util/file_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

namespace ov {
namespace test {
namespace utils {

inline void fixSlashes(std::string& str) {
    std::replace(str.begin(), str.end(), '/', '\\');
}

inline void fixSlashes(std::wstring& str) {
    std::replace(str.begin(), str.end(), L'/', L'\\');
}

inline std::wstring stringToWString(std::string input) {
    return ::ov::util::string_to_wstring(input.c_str());
}

inline bool copyFile(std::wstring source_path, std::wstring dest_path) {
#    ifndef _WIN32
    std::ifstream source(ov::util::wstring_to_string(source_path), std::ios::binary);
    std::ofstream dest(ov::util::wstring_to_string(dest_path), std::ios::binary);
#    else
    fixSlashes(source_path);
    fixSlashes(dest_path);
    std::ifstream source(source_path.c_str(), std::ios::binary);
    std::ofstream dest(dest_path.c_str(), std::ios::binary);
#    endif
    bool result = source && dest;
    std::istreambuf_iterator<char> begin_source(source);
    std::istreambuf_iterator<char> end_source;
    std::ostreambuf_iterator<char> begin_dest(dest);
    copy(begin_source, end_source, begin_dest);

    source.close();
    dest.close();
    return result;
}

inline bool copyFile(std::string source_path, std::wstring dest_path) {
    return copyFile(stringToWString(source_path), dest_path);
}

inline std::wstring addUnicodePostfixToPath(std::string source_path, std::wstring postfix) {
    fixSlashes(source_path);
    auto result = stringToWString(source_path);
    auto extPos = result.rfind('.');
    auto extension = result.substr(extPos, result.size());
    auto file_name = result.substr(0, extPos);

    return file_name + postfix + extension;
}

inline void removeFile(std::wstring path) {
    int result = 0;
    if (!path.empty()) {
#    ifdef _WIN32
        result = _wremove(path.c_str());
#    else
        result = remove(ov::util::wstring_to_string(path).c_str());
#    endif
    }
    (void)result;
}

inline bool endsWith(const std::wstring& source, const std::wstring& expectedSuffix) {
    return expectedSuffix.size() <= source.size() &&
           source.compare(source.size() - expectedSuffix.size(), expectedSuffix.size(), expectedSuffix) == 0;
}

// Removes all files with extension=ext from the given directory
// Return value:
// < 0 - error
// >= 0 - count of removed files
inline int removeFilesWithExt(std::wstring path, std::wstring ext) {
    int ret = 0;
#    ifdef _WIN32
    struct _wdirent* ent;
    _WDIR* dir = _wopendir(path.c_str());
    if (dir != nullptr) {
        while ((ent = _wreaddir(dir)) != NULL) {
            auto file = ov::util::make_path(path, std::wstring(ent->wd_name));
            struct _stat64i32 stat_path;
            _wstat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && endsWith(file, L"." + ext)) {
                auto err = _wremove(file.c_str());
                if (err != 0) {
                    _wclosedir(dir);
                    return err;
                }
                ret++;
            }
        }
        _wclosedir(dir);
    }
#    else
    struct dirent* ent;
    auto path_mb = ov::util::wstring_to_string(path);
    auto ext_mb = ov::util::wstring_to_string(ext);
    DIR* dir = opendir(path_mb.c_str());
    if (dir != nullptr) {
        while ((ent = readdir(dir)) != NULL) {
            std::string file = ov::util::make_path(path_mb, std::string(ent->d_name));
            struct stat stat_path;
            stat(file.c_str(), &stat_path);
            if (!S_ISDIR(stat_path.st_mode) && endsWith(file, "." + ext_mb)) {
                auto err = std::remove(file.c_str());
                if (err != 0) {
                    closedir(dir);
                    return err;
                }
                ret++;
            }
        }
        closedir(dir);
    }
#    endif
    return ret;
}

inline int removeDir(std::wstring path) {
    int result = 0;
    if (!path.empty()) {
#    ifdef _WIN32
        result = _wrmdir(path.c_str());
#    else
        result = rmdir(ov::util::wstring_to_string(path).c_str());
#    endif
    }
    return result;
}

extern const std::vector<std::wstring> test_unicode_postfix_vector;

}  // namespace utils
}  // namespace test
}  // namespace ov
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
