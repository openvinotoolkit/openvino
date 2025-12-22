// Copyright (C) 2018-2025 Intel Corporation
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
#    include <shlwapi.h>
#    include <windows.h>
/// @brief Max length of absolute file path
#    define MAX_ABS_PATH _MAX_PATH
/// @brief Get absolute file path, returns NULL in case of error
#    define get_absolute_path(result, path) _fullpath(result, path.c_str(), MAX_ABS_PATH)
/// @brief Windows-specific 'stat' wrapper
#    define stat _stat
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        define wstat _wstat
#    endif
/// @brief Windows-specific 'mkdir' wrapper
#    define makedir(dir) _mkdir(dir.c_str())
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        define wmakedir(dir) _wmkdir(dir.c_str())
#    endif
// Copied from linux libc sys/stat.h:
#    if !defined(__MINGW32__) && !defined(__MINGW64__)
#        define S_ISDIR(m) (((m) & S_IFMT) == S_IFDIR)
#    endif
#else
#    include <dirent.h>
#    include <dlfcn.h>
#    include <ftw.h>
#    include <limits.h>
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
#    define makedir(dir)                    mkdir(dir.c_str(), 0755)
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        define wmakedir(dir) mkdir(ov::util::wstring_to_string(dir).c_str(), 0755)
#    endif
#endif

std::string ov::util::get_file_name(const std::string& s) {
    if (const auto path = make_path(s); path.has_parent_path()) {
        return path_to_string(path.filename());
    } else {
        return s;
    }
}

std::string ov::util::get_file_ext(const std::string& path) {
    return path_to_string(make_path(path).extension());
}

std::filesystem::path ov::util::get_directory(const std::filesystem::path& path) {
    if (path.empty()) {
        return {};
    } else if (const auto& parent_path = path.parent_path(); parent_path.empty()) {
        return {"."};
    } else {
        return parent_path;
    }
}

template <class Container = std::initializer_list<std::filesystem::path>>
std::filesystem::path path_join(Container&& paths) {
    std::filesystem::path joined_path{};

    for (auto&& path : paths) {
        if (!path.empty()) {
            joined_path /= path;
        }
    }
    return joined_path;
}

// TODO: Remove string() / wstring() casts on function call site
std::filesystem::path ov::util::path_join(std::initializer_list<std::filesystem::path>&& paths) {
    return ::path_join<>(std::move(paths));
}

std::wstring ov::util::path_join_w(std::initializer_list<std::wstring>&& paths) {
    return ::path_join<>(std::move(paths)).wstring();
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
                case DT_UNKNOWN:
                    // Comment from READDIR(3):
                    //     only some filesystems have full support for returning the file type in d_type.
                    //     All applications must properly handle a return of DT_UNKNOWN.
                    func(path_name, false);
                    break;
                default:
                    break;
                }
            }
        } catch (...) {
            std::exception_ptr p = std::current_exception();
            closedir(dir);
            std::rethrow_exception(std::move(p));
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
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    std::wstring pathw = string_to_wstring(path);
    std::wstring file_match = path_join_w({pathw, L"*"});
    WIN32_FIND_DATAW data;
    HANDLE hFind = FindFirstFileW(file_match.c_str(), &data);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            bool is_dir = data.dwFileAttributes == FILE_ATTRIBUTE_DIRECTORY;
            if (is_dir) {
                if (std::wstring(data.cFileName) != L"." && std::wstring(data.cFileName) != L"..") {
                    std::wstring dir_pathw = path_join_w({pathw, data.cFileName});
                    std::string dir_path = wstring_to_string(dir_pathw);
                    if (recurse) {
                        iterate_files(dir_path, func, recurse);
                    }
                    func(dir_path, true);
                }
            } else {
                std::wstring file_namew = path_join_w({pathw, data.cFileName});
                std::string file_name = wstring_to_string(file_namew);
                func(file_name, false);
            }
        } while (FindNextFileW(hFind, &data));
        FindClose(hFind);
    }
#    else
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
#    endif
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

std::string ov::util::get_absolute_file_path(const std::string& path) {
    std::string absolutePath;
    absolutePath.resize(MAX_ABS_PATH);
    std::ignore = get_absolute_path(&absolutePath[0], path);
    if (!absolutePath.empty()) {
        // on Linux if file does not exist or no access, function will return NULL, but
        // `absolutePath` will contain resolved path
        absolutePath.resize(absolutePath.find('\0'));
        return absolutePath;
    }
    std::stringstream ss;
    ss << "Can't get absolute file path for [" << path << "], err = " << strerror(errno);
    throw std::runtime_error(ss.str());
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
void ov::util::create_directory_recursive(const std::wstring& path) {
    create_directory_recursive(make_path(path));
}
#endif

void ov::util::create_directory_recursive(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    auto dir_path = fs::weakly_canonical(path);

    if (!dir_path.empty() && !directory_exists(dir_path)) {
        // NOTE: Some standard library implementations (MSVC STL, libc++) may return `false`
        // from create_directories(path, ec) (with ec == 0) when the path ends with a
        // trailing separator (e.g. "a/b/c/"). Internally they create "a", "a/b", "a/b/c"
        // and then the extra mkdir on "a/b/c/" yields an "already exists" condition,
        // leading to a final `false` even though the directory tree was actually created.
        // libstdc++ (GCC) returns `true` in that situation. The extra exists() check
        // lets us treat "false + exists()" as success while still detecting a real failure
        // ("false + !exists()"), keeping behavior consistent across platforms.
        if (std::error_code ec; !fs::create_directories(dir_path, ec) && !std::filesystem::exists(dir_path)) {
            std::stringstream ss;
            ss << "Couldn't create directory [" << dir_path << "], err=" << ec.message() << ")";
            throw std::runtime_error(ss.str());
        }
    }
}

bool ov::util::directory_exists(const std::filesystem::path& path) {
    return std::filesystem::is_directory(std::filesystem::status(path));
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

#if defined __GNUC__ || defined __clang__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wunused-function"
#endif

std::string get_ov_library_path_a() {
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
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ov::util::get_ov_lib_path), &info);
    return get_path_name(ov::util::get_absolute_file_path(info.dli_fname)).c_str();
#else
#    error "Unsupported OS"
#endif  // _WIN32
}

#if defined __GNUC__ || defined __clang__
#    pragma GCC diagnostic pop
#endif

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
#    elif defined(__linux__) || defined(__APPLE__) || defined(__EMSCRIPTEN__)
    return ov::util::string_to_wstring(get_ov_library_path_a());
#    else
#        error "Unsupported OS"
#    endif
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::filesystem::path ov::util::get_plugin_path(const std::filesystem::path& plugin) {
    // Assume `plugin` may contain:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to working directory
    // 3. example library name - to be converted to 4th case
    // 4. libexample.so - path relative to working directory (if exists) or file to be found in ENV
    if (plugin.is_absolute()) {
        // 1st case
        return plugin;
    } else if (plugin.has_parent_path()) {
        // 2nd cases
        return std::filesystem::absolute(std::filesystem::weakly_canonical(plugin));
    } else {
        // 3-4 cases
        auto path = (plugin.extension() != library_extension()) ? make_plugin_library_name(plugin) : plugin;
        std::error_code ec;
        auto abs_path = std::filesystem::canonical(path, ec);
        return ec ? path : abs_path;
    }
}

std::filesystem::path ov::util::get_compiled_plugin_path(const std::filesystem::path& plugin) {
    const auto ov_library_path = make_path(get_ov_lib_path());

    // plugin can be found either:
    const std::filesystem::path sub_folder_path{std::string("openvino-") + OpenVINO_VERSION};

    if (auto plugin_path = ov_library_path / sub_folder_path / plugin; ov::util::file_exists(plugin_path)) {
        return plugin_path;
    } else if (plugin_path = ov_library_path / plugin; ov::util::file_exists(plugin_path)) {
        return plugin_path;
    } else {
        // cases 3-4
        return get_plugin_path(plugin);
    }
}

std::filesystem::path ov::util::get_plugin_path(const std::filesystem::path& plugin,
                                                const std::filesystem::path& xml_dir,
                                                bool as_abs_only) {
    // Assume `plugin` (from XML "location" record) contains only:
    // 1. /path/to/libexample.so absolute path
    // 2. ../path/to/libexample.so path relative to XML directory
    // 3. example library name - to be converted to 4th case
    // 4. libexample.so - path relative to XML directory (if exists) or file to be found in ENV
    // (if `as_abs_only` is false)
    if (plugin.is_absolute()) {
        // 1st case
        return plugin;
    } else if (const auto rel_path = get_directory((xml_dir.has_parent_path() ? xml_dir : "." / xml_dir));
               plugin.has_parent_path()) {
        // 2nd case
        return std::filesystem::absolute(std::filesystem::weakly_canonical(rel_path / plugin));
    } else {
        // 3-4 cases
        auto&& lib_name = ((plugin.extension() != library_extension()) ? make_plugin_library_name(plugin) : plugin);
        auto abs_path = std::filesystem::absolute(std::filesystem::weakly_canonical(rel_path / lib_name));
        return as_abs_only || file_exists(abs_path) ? abs_path : lib_name;
    }
}

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

void ov::util::save_binary(const std::string& path, const std::vector<uint8_t>& binary) {
    save_binary(path, reinterpret_cast<const char*>(&binary[0]), binary.size());
    return;
}

void ov::util::save_binary(const std::string& path, const char* binary, size_t bin_size) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widefilename = ov::util::string_to_wstring(path);
    const wchar_t* filename = widefilename.c_str();
#else
    const char* filename = path.c_str();
#endif
    std::ofstream out_file(filename, std::ios::out | std::ios::binary);
    if (out_file.is_open()) {
        out_file.write(binary, bin_size);
    } else {
        throw std::runtime_error("Could not save binary to " + path);
    }
}

const char* ov::util::trim_file_name(const char* const fname) {
    static const auto pattern_native_sep =
        std::string(OV_NATIVE_PARENT_PROJECT_ROOT_DIR) + FileTraits<char>::file_separator;

    const auto has_native_sep_pattern_ptr = std::strstr(fname, pattern_native_sep.c_str());
    auto fname_trim_ptr = has_native_sep_pattern_ptr ? has_native_sep_pattern_ptr + pattern_native_sep.size() : fname;

#if defined(_WIN32)
    // On windows check also forward slash as in some case the __FILE__ can have it instead native backward slash.
    if (fname_trim_ptr == fname) {
        static const auto pattern_fwd_sep = std::string(OV_NATIVE_PARENT_PROJECT_ROOT_DIR) + '/';
        if (const auto has_fwd_sep_pattern_ptr = std::strstr(fname, pattern_fwd_sep.c_str())) {
            fname_trim_ptr = has_fwd_sep_pattern_ptr + pattern_fwd_sep.size();
        }
    }
#endif
    return fname_trim_ptr;
}
