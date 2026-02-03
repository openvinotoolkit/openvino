// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <string>
#include <vector>

#include "openvino/util/file_path.hpp"
#include "openvino/util/util.hpp"
#include "openvino/util/wstring_convert_util.hpp"

namespace ov::util {

inline std::filesystem::path library_prefix() {
#if defined(_WIN32) && (defined(__MINGW32__) || defined(__MINGW64__))
    return {"lib"};
#elif defined(_WIN32)
    return {};
#else
    return {"lib"};
#endif
}
inline std::filesystem::path library_extension() {
#ifdef _WIN32
    return {".dll"};
#else
    return {".so"};
#endif
}

/**
 * @brief Creates std::filesystem::path provided by source.
 *
 * The purpose of this function is to hide platform specific issue with path creation like on Windows create from
 * literal std::string with unicode characters can lead to different path name than expected.
 *
 * @param source  Source to create path. Supported types are same as for std::filesystem::path constructor.
 * @return std::filesystem::path object.
 */
template <class Source>
std::filesystem::path make_path(Source&& source) {
    if constexpr (std::is_same_v<std::add_const_t<std::remove_pointer_t<std::decay_t<Source>>>, const wchar_t>) {
        return {std::wstring(std::forward<Source>(source))};
    } else if constexpr (std::is_same_v<std::filesystem::path::string_type, std::wstring> &&
                         std::disjunction_v<std::is_same<std::decay_t<Source>, std::string>,
                                            std::is_same<std::decay_t<Source>, const char*>,
                                            std::is_same<std::decay_t<Source>, char*>,
                                            std::is_same<std::decay_t<Source>, std::string_view>>) {
        return {string_to_wstring(std::forward<Source>(source))};
    } else {
        return {std::forward<Source>(source)};
    }
}

/**
 * @brief Convert path as char string to to a single-byte chain.
 * @param path Path as char string.
 * @return Reference to input path (no conversion).
 */
template <class Path, typename std::enable_if_t<std::is_same_v<std::decay_t<Path>, std::string>>* = nullptr>
const std::string& path_to_string(const Path& path) {
    return path;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Convert path as wide character string to a single-byte chain.
 * @param path  Path as wide-char string.
 * @return A char string
 */
template <class Path, typename std::enable_if_t<std::is_same_v<std::decay_t<Path>, std::wstring>>* = nullptr>
std::string path_to_string(const Path& path) {
    return wstring_to_string(path);
}

#endif

/**
 * @brief Convert std::filesystem::path single-byte chain.
 * Function resolve issue when path create from std::string which contains unicode characters.
 * @param path  Path.
 * @return A char string.
 */
inline auto path_to_string(const std::filesystem::path& path) -> decltype(path_to_string(path.native())) {
    return path_to_string(path.native());
}

/// \brief Remove path components which would allow traversing up a directory tree.
/// \param path A path to file
/// \return A sanitized path
std::string sanitize_path(const std::string& path);

/**
 * @brief Interface function to get absolute path of file
 * @param path - path to file, can be relative to current working directory
 * @return Absolute path of file
 * @throw runtime_error if absolute path can't be resolved
 */
std::filesystem::path get_absolute_file_path(const std::filesystem::path& path);

/** @{ */
/**
 * @brief Interface function to create directories recursively by given path
 * @param path - path to file, can be relative to current working directory
 * @throw runtime_error if any error occurred
 */
void create_directory_recursive(const std::filesystem::path& path);

template <class Path>
void create_directory_recursive(const Path& path) {
    return create_directory_recursive(make_path(path));
}
/** @} */

/**
 * @brief Interface function to check if directory exists for given path
 * @param path - path to directory
 * @return true if directory exists, false otherwise
 */
bool directory_exists(const std::filesystem::path& path);

/** @{ */
/**
 * @brief      Returns file size for file
 * @param[in]  path  The file name
 * @return     file size
 */
inline int64_t file_size(const std::filesystem::path& path) noexcept {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? -1 : static_cast<int64_t>(size);
}

template <class Path>
inline int64_t file_size(const Path& path) noexcept {
    return ov::util::file_size(make_path(path));
}
/** @} */

/** @{ */
/**
 * @brief      Tests whether file exists at given path.
 * @param[in]  path  The file path.
 * @return     True if file exists, false otherwise.
 */
inline bool file_exists(const std::filesystem::path& path) noexcept {
#if defined(__ANDROID__) || defined(ANDROID)
    const auto pos = path.native().find('!');
    const auto f_status = (pos == std::string::npos) ? std::filesystem::status(path)
                                                     : std::filesystem::status(path.native().substr(0, pos));
#else
    const auto f_status = std::filesystem::status(path);
#endif
    return std::filesystem::exists(f_status) && !std::filesystem::is_directory(f_status);
}

template <class Path>
inline bool file_exists(const Path& path) noexcept {
    return file_exists(make_path(path));
}
/** @} */

std::filesystem::path get_directory(const std::filesystem::path& path);

std::filesystem::path path_join(std::initializer_list<std::filesystem::path>&& paths);
std::wstring path_join_w(std::initializer_list<std::wstring>&& paths);

/**
 * @brief Iterates over files in given directory and applies provided function to each file found.
 *
 * @param path Root directory path to iterate files from.
 * @param func Function to apply to each file found.
 * @param include_links Whether to include symbolic links.
 */
void iterate_files(const std::filesystem::path& path,
                   const std::function<void(const std::filesystem::path& file)>& func,
                   bool include_links = false);

/**
 * @brief Recusive iterates over files in given directory and applies provided function to each file found.
 *
 * @param path Root directory path to iterate files from.
 * @param func Function to apply to each file found.
 * @param recurse Whether to recurse into subdirectories.
 * @param include_links Whether to include symbolic links.
 */
void recursive_iterate_files(const std::filesystem::path& path,
                             const std::function<void(const std::filesystem::path& file)>& func,
                             bool include_links = false);

/**
 * @brief   Gets a path to OpenVINO libraries.
 * @return  Path to OpenVINO libraries.
 */
std::filesystem::path get_ov_lib_path();

inline std::filesystem::path make_plugin_library_name(const std::filesystem::path& lib_name) {
    return library_prefix().concat(lib_name.filename().native()).concat(library_extension().native());
}

inline std::filesystem::path make_plugin_library_name(const std::filesystem::path& dir_path,
                                                      const std::filesystem::path& lib_name) {
    return dir_path / make_plugin_library_name(lib_name);
}

template <typename C, typename = std::enable_if_t<(std::is_same_v<C, char> || std::is_same_v<C, wchar_t>)>>
inline std::basic_string<C> make_plugin_library_name(const std::basic_string<C>& path,
                                                     const std::basic_string<C>& input) {
    return path_to_string(make_plugin_library_name(make_path(path), make_path(input)));
}

/**
 * @brief Format plugin path (canonicalize, complete to absolute or complete to file name) for further
 * dynamic loading by OS
 * @param plugin - Path (absolute or relative) or name of a plugin. Depending on platform, `plugin` is wrapped with
 * shared library suffix and prefix to identify library full name
 * @return absolute path or file name with extension (to be found in ENV)
 */
std::filesystem::path get_plugin_path(const std::filesystem::path& plugin);

/**
 * @brief Find the plugins which are located together with OV library
 * @param plugin - Path (absolute or relative) or name of a plugin. Depending on platform, `plugin` is wrapped with
 * shared library suffix and prefix to identify library full name
 * @return absolute path or file name with extension (to be found in ENV)
 */
std::filesystem::path get_compiled_plugin_path(const std::filesystem::path& plugin);

/**
 * @brief Format plugin path (canonicalize, complete to absolute or complete to file name) for further
 * dynamic loading by OS
 * @param plugin - Path (absolute or relative) or name of a plugin. Depending on platform, `plugin` is wrapped with
 * shared library suffix and prefix to identify library full name
 * @param xml_path - Path (absolute or relative) to XML configuration file
 * @param as_abs_only - Bool value, allows return file names or not
 * @return absolute path or file name with extension (to be found in ENV)
 */
std::filesystem::path get_plugin_path(const std::filesystem::path& plugin,
                                      const std::filesystem::path& xml_path,
                                      bool as_abs_only = false);

/**
 * @brief load binary data from file
 * @param path - binary file path to load
 * @return binary vector
 */
std::vector<uint8_t> load_binary(const std::filesystem::path& path);

/**
 * @brief save binary data to file
 * @param path - binary file path to store
 */
void save_binary(const std::filesystem::path& path, const void* binary, size_t bin_size);

/**
 * @brief Trim OpenVINO project file name path if OpenVINO project directory found.
 *
 * Function use `OV_NATIVE_PARENT_PROJECT_ROOT_DIR` definition with project directory name defines
 * 'openvino_dir_name'. The input file name is scanned for OV_NATIVE_PARENT_PROJECT_ROOT_DIR,
 * if found returns pointer to trimmed name otherwise returns input pointer.
 *
 * e.g: OV_NATIVE_PARENT_PROJECT_ROOT_DIR = openvino
 * - /home/user/openvino/src/example.cpp -> src/example.cpp
 * - ../../../../openvino/src/example.cpp -> src/example.cpp
 *
 * @param fname  Pointer to OpenVINO file name path.
 * @return Pointer to trimmed file name path.
 */
const char* trim_file_name(const char* const fname);

}  // namespace ov::util
