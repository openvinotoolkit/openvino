// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string_view>

#include "openvino/util/common_util.hpp"

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <direct.h>
#    include <shlwapi.h>
#    include <windows.h>
#else
#    include <dirent.h>
#    include <dlfcn.h>
#endif

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

namespace {
void process_dir_entry(const std::filesystem::directory_entry& dir_entry,
                       const std::function<void(const std::filesystem::path& file)>& func) {
    const auto& status = dir_entry.status();
    if (!std::filesystem::is_directory(status) && !std::filesystem::is_symlink(status)) {
        func(dir_entry.path());
    }
}

void process_dir_entry_include_links(const std::filesystem::directory_entry& dir_entry,
                                     const std::function<void(const std::filesystem::path& file)>& func) {
    const auto& status = dir_entry.status();
    if (!std::filesystem::is_directory(status)) {
        func(dir_entry.path());
    }
}
}  // namespace

void ov::util::iterate_files(const std::filesystem::path& path,
                             const std::function<void(const std::filesystem::path& file)>& func,
                             bool include_links) {
    std::error_code ec;
    const auto dir_iter = std::filesystem::directory_iterator(path, ec);
    if (ec) {
        throw std::runtime_error("error enumerating file " + path_to_string(path) + ", err: " + ec.message());
    }

    const auto dir_entry_func = include_links ? process_dir_entry_include_links : process_dir_entry;
    for (const auto& dir_entry : dir_iter) {
        dir_entry_func(dir_entry, func);
    }
}

void ov::util::recursive_iterate_files(const std::filesystem::path& path,
                                       const std::function<void(const std::filesystem::path& file)>& func,
                                       bool include_links) {
    std::error_code ec;
    const auto dir_iter = std::filesystem::recursive_directory_iterator(path, ec);
    if (ec) {
        throw std::runtime_error("error enumerating file " + path_to_string(path) + ", err: " + ec.message());
    }

    const auto dir_entry_func = include_links ? process_dir_entry_include_links : process_dir_entry;
    for (const auto& dir_entry : dir_iter) {
        dir_entry_func(dir_entry, func);
    }
}

std::string ov::util::sanitize_path(const std::string& path) {
    const auto colon_pos = path.find(':');
    const auto sanitized_path = path.substr(colon_pos == std::string::npos ? 0 : colon_pos + 1);
    const std::string to_erase = "/.\\";
    const auto start = sanitized_path.find_first_not_of(to_erase);
    return (start == std::string::npos) ? "" : sanitized_path.substr(start);
}

std::filesystem::path ov::util::get_absolute_file_path(const std::filesystem::path& path) {
    std::error_code ec;
    if (path.empty() || path.is_absolute()) {
        return path;
    } else if (auto abs_path = std::filesystem::absolute(std::filesystem::weakly_canonical(path, ec)); !ec) {
        return abs_path;
    } else {
        std::stringstream ss;
        ss << "Can't get absolute file path for [" << path_to_string(path) << "], err = " << ec.message();
        throw std::runtime_error(ss.str());
    }
}

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

std::filesystem::path ov::util::get_ov_lib_path() {
#ifdef _WIN32
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
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
    return std::filesystem::path(ov_library_path).parent_path();

#    else
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
    return make_path(ov_library_path).parent_path();
#    endif
#elif defined(__APPLE__) || defined(__linux__) || defined(__EMSCRIPTEN__)
    Dl_info info;
    dladdr(reinterpret_cast<void*>(ov::util::get_ov_lib_path), &info);
    return ov::util::get_absolute_file_path(info.dli_fname).parent_path();
#else
#    error "Unsupported OS"
#endif
}

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
    const auto ov_library_path = get_ov_lib_path();

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

std::vector<uint8_t> ov::util::load_binary(const std::filesystem::path& path) {
    std::vector<uint8_t> buffer;
    if (auto input = std::ifstream(path, std::ios::binary); input.is_open()) {
        buffer.reserve(std::filesystem::file_size(path));
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.capacity());
    }
    return buffer;
}

void ov::util::save_binary(const std::filesystem::path& path, const void* binary, size_t bin_size) {
    if (std::ofstream out_file(path, std::ios::binary); out_file.is_open()) {
        out_file.write(reinterpret_cast<const char*>(binary), bin_size);
    } else {
        throw std::runtime_error("Could not save binary to " + path_to_string(path));
    }
}

const char* ov::util::trim_file_name(const char* const fname) {
    constexpr auto parent_project_root = std::string_view(OV_NATIVE_PARENT_PROJECT_ROOT_DIR);
#ifdef _WIN32
    constexpr auto native_path_separator = '\\';
    constexpr auto portable_path_separator = '/';
#else
    constexpr auto native_path_separator = '/';
#endif

    if (auto fname_trim_ptr = std::strstr(fname, parent_project_root.data()); fname_trim_ptr) {
        fname_trim_ptr += parent_project_root.size();
        if (*fname_trim_ptr == native_path_separator
#ifdef _WIN32
            // On windows check also forward slash as in some case the __FILE__ can have it instead native backward
            // slash.
            || *fname_trim_ptr == portable_path_separator
#endif
        ) {
            return fname_trim_ptr + 1;
        }
    }

    return fname;
}
