// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/util/file_util.hpp"

#include <sys/stat.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

#include "openvino/util/common_util.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <direct.h>
#    include <windows.h>
/// @brief Max length of absolute file path
#    define MAX_ABS_PATH _MAX_PATH
/// @brief Get absolute file path, returns NULL in case of error
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
/// @brief Windows-specific 'stat' wrapper
#    define stat _stat
/// @brief Windows-specific 'mkdir' wrapper
#    define makedir(dir) _mkdir(dir)
// Copied from linux libc sys/stat.h:
#    define S_ISDIR(m) (((m)&S_IFMT) == S_IFDIR)
#else
#    include <dirent.h>
#    include <dlfcn.h>
#    include <ftw.h>
#    include <sys/file.h>
#    include <sys/time.h>
#    include <unistd.h>

#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        include <codecvt>
#        include <locale>
#    endif

/// @brief Max length of absolute file path
#    define MAX_ABS_PATH                    PATH_MAX
/// @brief Get absolute file path, returns NULL in case of error
#    define get_absolute_path(result, path) realpath(path.c_str(), result)
/// @brief mkdir wrapper
#    define makedir(dir)                    mkdir(dir, 0755)
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
                rc += '/';
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
            std::rethrow_exception(p);
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
    std::string file_match = path_join({path, "*"});
    WIN32_FIND_DATAA data;
    HANDLE hFind = FindFirstFileA(file_match.c_str(), &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            bool is_dir = data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY;
            if (is_dir) {
                if (std::string(data.cFileName) != "." && std::string(data.cFileName) != "..") {
                    std::string dir_path = path_join({path, data.cFileName});
                    if (recurse) {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
            } else {
                std::string file_name = path_join({path, data.cFileName});
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

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::string ov::util::wstring_to_string(const std::wstring& wstr) {
#    ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#    endif
}

std::wstring ov::util::string_to_wstring(const std::string& string) {
    const char* str = string.c_str();
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
#endif

std::string ov::util::get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    auto absPath = get_absolute_path(&absolutePath[0], path);
    if (!absPath) {
        std::stringstream ss;
        ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
        throw std::runtime_error(ss.str());
    }
    absolutePath.resize(strlen(absPath));
    return absolutePath;
}

void ov::util::create_directory_recursive(const std::string& path) {
    if (path.empty() || directory_exists(path)) {
        return;
    }

    std::size_t pos = path.rfind(ov::util::FileTraits<char>::file_separator);
    if (pos != std::string::npos) {
        create_directory_recursive(path.substr(0, pos));
    }

    int err = makedir(path.c_str());
    if (err != 0 && errno != EEXIST) {
        std::stringstream ss;
        // TODO: in case of exception it may be needed to remove all created sub-directories
        ss << "Couldn't create directory [" << path << "], err=" << strerror(errno) << ")";
        throw std::runtime_error(ss.str());
    }
}

bool ov::util::directory_exists(const std::string& path) {
    struct stat sb;

    if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        return true;
    }
    return false;
}

namespace {

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
std::basic_string<C> get_path_name(const std::basic_string<C>& s) {
    size_t i = s.rfind(ov::util::FileTraits<C>::file_separator, s.length());
    if (i != std::string::npos) {
        return (s.substr(0, i));
    }

    return {};
}

static std::string get_ov_library_path_a() {
#ifdef _WIN32
    CHAR ov_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPSTR>(ov::util::get_ov_lib_path),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandle returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    GetModuleFileNameA(hm, (LPSTR)ov_library_path, sizeof(ov_library_path));
    return get_path_name(std::string(ov_library_path));
#elif defined(__APPLE__) || defined(__linux__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ov::util::get_ov_lib_path), &info);
    std::string result = get_path_name(ov::util::get_absolute_file_path(info.dli_fname)).c_str();
    if (!ov::util::ends_with(result, "/lib") && !ov::util::ends_with(result, "/lib/"))
        result = ov::util::path_join({result, "lib"});
    return result;
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

}  // namespace

std::string ov::util::get_ov_lib_path() {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    return ov::util::wstring_to_string(ov::util::get_ov_lib_path_w());
#else
    return get_ov_library_path_a();
#endif
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::wstring ov::util::get_ov_lib_path_w() {
#    ifdef _WIN32
    WCHAR ov_library_path[MAX_PATH];
    HMODULE hm = NULL;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(get_ov_lib_path),
                            &hm)) {
        std::stringstream ss;
        ss << "GetModuleHandle returned " << GetLastError();
        throw std::runtime_error(ss.str());
    }
    GetModuleFileNameW(hm, (LPWSTR)ov_library_path, sizeof(ov_library_path) / sizeof(ov_library_path[0]));
    return get_path_name(std::wstring(ov_library_path));
#    elif defined(__linux__) || defined(__APPLE__)
    return ov::util::string_to_wstring(get_ov_library_path_a());
#    else
#        error "Unsupported OS"
#    endif
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::vector<uint8_t> ov::util::load_binary(const std::string& path) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = ov::util::string_to_wstring(path);
    const wchar_t* filename = widefilename.c_str();
    FILE* fp = _wfopen(filename, L"rb");
#else
    const char* filename = path.c_str();
    FILE* fp = fopen(filename, "rb");
#endif

    if (fp) {
        fseek(fp, 0, SEEK_END);
        auto sz = ftell(fp);
        if (sz < 0) {
            fclose(fp);
            return {};
        }
        auto nsize = static_cast<size_t>(sz);

        fseek(fp, 0, SEEK_SET);

        std::vector<uint8_t> ret(nsize);

        auto res = fread(ret.data(), sizeof(uint8_t), nsize, fp);
        (void)res;
        fclose(fp);
        return ret;
    }

    return {};
}

void ov::util::save_binary(const std::string& path, std::vector<uint8_t> binary) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = ov::util::string_to_wstring(path);
    const wchar_t* filename = widefilename.c_str();
#else
    const char* filename = path.c_str();
#endif
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (out_file.is_open()) {
        out_file.write(reinterpret_cast<const char*>(&binary[0]), binary.size());
    } else {
        throw std::runtime_error("Could not save binary to " + path);
    }
}
