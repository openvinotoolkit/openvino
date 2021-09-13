// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/util/file_util.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <dirent.h>
#    include <ftw.h>
#    include <sys/file.h>
#    include <sys/time.h>
#    include <unistd.h>
#    include <codecvt>
#    include <locale>
#endif

std::string ov::util::get_file_name(const std::string& s) {
    std::string rc = s;
    auto pos = s.find_last_of('/');
    if (pos != std::string::npos) {
        rc = s.substr(pos + 1);
    }
    return rc;
}

std::string ov::util::get_file_ext(const std::string& s) {
    std::string rc = get_file_name(s);
    auto pos = rc.find_last_of('.');
    if (pos != std::string::npos) {
        rc = rc.substr(pos);
    } else {
        rc = "";
    }
    return rc;
}

std::string ov::util::get_directory(const std::string& s) {
    std::string rc = s;
    // Linux-style separator
    auto pos = s.find_last_of('/');
    if (pos != std::string::npos) {
        rc = s.substr(0, pos);
        return rc;
    }
    // Windows-style separator
    pos = s.find_last_of('\\');
    if (pos != std::string::npos) {
        rc = s.substr(0, pos);
        return rc;
    }
    return rc;
}

namespace {

std::string join_paths(const std::string& s1, const std::string& s2) {
    std::string rc;
    if (s2.size() > 0) {
        if (s2[0] == '/') {
            rc = s2;
        } else if (s1.size() > 0) {
            rc = s1;
            if (rc[rc.size() - 1] != '/') {
                rc += "/";
            }
            rc += s2;
        } else {
            rc = s2;
        }
    } else {
        rc = s1;
    }
    return rc;
}
}  // namespace

std::string ov::util::path_join(const std::vector<std::string>& paths) {
    std::string result;
    if (paths.empty()) {
        return result;
    }
    result = paths[0];
    for (size_t i = 1; i < paths.size(); i++) {
        result = join_paths(result, paths[i]);
    }
    return result;
}

#ifndef _WIN32
static void iterate_files_worker(const std::string& path,
                                 const std::function<void(const std::string& file, bool is_dir)>& func,
                                 bool recurse,
                                 bool include_links) {
    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(path.c_str())) != nullptr) {
        try {
            while ((ent = readdir(dir)) != nullptr) {
                std::string name = ent->d_name;
                std::string path_name = ov::util::path_join({path, name});
                switch (ent->d_type) {
                case DT_DIR:
                    if (name != "." && name != "..") {
                        if (recurse) {
                            ov::util::iterate_files(path_name, func, recurse);
                        }
                        func(path_name, true);
                    }
                    break;
                case DT_LNK:
                    if (include_links) {
                        func(path_name, false);
                    }
                    break;
                case DT_REG:
                    func(path_name, false);
                    break;
                default:
                    break;
                }
            }
        } catch (...) {
            std::exception_ptr p = std::current_exception();
            closedir(dir);
            rethrow_exception(p);
        }
        closedir(dir);
    } else {
        throw std::runtime_error("error enumerating file " + path);
    }
}
#endif

void ov::util::iterate_files(const std::string& path,
                             const std::function<void(const std::string& file, bool is_dir)>& func,
                             bool recurse,
                             bool include_links) {
    std::vector<std::string> files;
    std::vector<std::string> dirs;
#ifdef _WIN32
    std::string file_match = path_join(path, "*");
    WIN32_FIND_DATAA data;
    HANDLE hFind = FindFirstFileA(file_match.c_str(), &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            bool is_dir = data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY;
            if (is_dir) {
                if (string(data.cFileName) != "." && string(data.cFileName) != "..") {
                    string dir_path = path_join(path, data.cFileName);
                    if (recurse) {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
            } else {
                string file_name = path_join(path, data.cFileName);
                func(file_name, false);
            }
        } while (FindNextFileA(hFind, &data));
        FindClose(hFind);
    }
#else
    iterate_files_worker(
        path,
        [&files, &dirs](const std::string& file, bool is_dir) {
            if (is_dir) {
                dirs.push_back(file);
            } else {
                files.push_back(file);
            }
        },
        recurse,
        include_links);
#endif

    for (const auto& f : files) {
        func(f, false);
    }
    for (const auto& f : dirs) {
        func(f, true);
    }
}

std::string ov::util::sanitize_path(const std::string& path) {
    const auto colon_pos = path.find(':');
    const auto sanitized_path = path.substr(colon_pos == std::string::npos ? 0 : colon_pos + 1);
    const std::string to_erase = "/.\\";
    const auto start = sanitized_path.find_first_not_of(to_erase);
    return (start == std::string::npos) ? "" : sanitized_path.substr(start);
}

void ov::util::convert_path_win_style(std::string& path) {
    std::replace(path.begin(), path.end(), '/', '\\');
}

std::string ov::util::wstring_to_string(const std::wstring& wstr) {
#    ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);  // NOLINT
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);  // NOLINT
    return strTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#    endif
}

std::wstring ov::util::multi_byte_char_to_wstring(const char* str) {
#    ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#    endif
}
