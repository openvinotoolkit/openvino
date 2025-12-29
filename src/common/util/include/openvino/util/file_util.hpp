// Copyright (C) 2018-2025 Intel Corporation
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

namespace ov {
namespace util {

/// OS specific file traits
template <class C>
struct FileTraits;

template <>
struct FileTraits<char> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    static constexpr const auto dot_symbol = '.';
    static std::string library_ext() {
#ifdef _WIN32
        return {"dll"};
#else
        return {"so"};
#endif
    }
    static std::string library_prefix() {
#ifdef _WIN32
#    if defined(__MINGW32__) || defined(__MINGW64__)
        return {"lib"};
#    else
        return {""};
#    endif
#else
        return {"lib"};
#endif
    }
};

template <>
struct FileTraits<wchar_t> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        L'\\';
#else
        L'/';
#endif
    static constexpr const auto dot_symbol = L'.';
    static std::wstring library_ext() {
#ifdef _WIN32
        return {L"dll"};
#else
        return {L"so"};
#endif
    }
    static std::wstring library_prefix() {
#ifdef _WIN32
#    if defined(__MINGW32__) || defined(__MINGW64__)
        return {L"lib"};
#    else
        return {L""};
#    endif
#else
        return {L"lib"};
#endif
    }
};

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
                         std::is_same_v<std::decay_t<Source>, std::string>) {
        return {ov::util::string_to_wstring(std::forward<Source>(source))};
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
    return ov::util::wstring_to_string(path);
}

#endif

/**
 * @brief Convert std::filesystem::path single-byte chain.
 * Function resolve issue when path create from std::string which contains unicode characters.
 * @param path  Path.
 * @return A char string.
 */
inline auto path_to_string(const std::filesystem::path& path) -> decltype(path_to_string(path.native())) {
    return ov::util::path_to_string(path.native());
}

/// \brief Remove path components which would allow traversing up a directory tree.
/// \param path A path to file
/// \return A sanitized path
std::string sanitize_path(const std::string& path);

/// \brief Returns the name with extension for a given path
/// \param path The path to the output file
std::string get_file_name(const std::string& path);

/**
 * @brief Interface function to get absolute path of file
 * @param path - path to file, can be relative to current working directory
 * @return Absolute path of file
 * @throw runtime_error if absolute path can't be resolved
 */
std::string get_absolute_file_path(const std::string& path);

/**
 * @brief Interface function to create directories recursively by given path
 * @param path - path to file, can be relative to current working directory
 * @throw runtime_error if any error occurred
 */
void create_directory_recursive(const std::filesystem::path& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Interface function to create directorty recursively by given path
 * @param path - path to file wide-string, can be relative to current working directory
 * @throw runtime_error if any error occurred
 */
void create_directory_recursive(const std::wstring& path);
#endif

/**
 * @brief Interface function to check if directory exists for given path
 * @param path - path to directory
 * @return true if directory exists, false otherwise
 */
bool directory_exists(const std::filesystem::path& path);

/**
 * @brief      Returns file size for file
 * @param[in]  path  The file name
 * @return     file size
 */

inline int64_t file_size(const std::filesystem::path& path) {
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? -1 : static_cast<int64_t>(size);
}

#ifdef _MSC_VER
inline int64_t file_size(const char* path) {
    return file_size(ov::util::string_to_wstring(path));
}
#endif

/**
 * @brief      Tests whether file exists at given path.
 * @param[in]  path  The file path.
 * @return     True if file exists, false otherwise.
 * @{
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

template <class T>
inline bool file_exists(const std::basic_string<T>& path) noexcept {
    return file_exists(make_path(path));
}
/** @} */

std::string get_file_ext(const std::string& path);
std::filesystem::path get_directory(const std::filesystem::path& path);

std::filesystem::path path_join(std::initializer_list<std::filesystem::path>&& paths);
std::wstring path_join_w(std::initializer_list<std::wstring>&& paths);

void iterate_files(const std::string& path,
                   const std::function<void(const std::string& file, bool is_dir)>& func,
                   bool recurse = false,
                   bool include_links = false);

void convert_path_win_style(std::string& path);

std::string get_ov_lib_path();

// TODO: remove this using. replace with Path.
using FilePath = std::filesystem::path::string_type;

// TODO: remove this function after get_plugin_path using Path
inline std::string from_file_path(const std::filesystem::path& path) {
    return path.string();
}
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief   Returns a unicode path to openvino libraries
 * @return  A `std::wstring` path to openvino libraries
 */
std::wstring get_ov_lib_path_w();

inline std::wstring get_ov_library_path() {
    return get_ov_lib_path_w();
}

#else

inline std::string get_ov_library_path() {
    return get_ov_lib_path();
}

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
inline std::basic_string<C> make_plugin_library_name(const std::basic_string<C>& path,
                                                     const std::basic_string<C>& input) {
    std::basic_string<C> separator(1, FileTraits<C>::file_separator);
    if (path.empty())
        separator = {};
    return path + separator + FileTraits<C>::library_prefix() + input + FileTraits<C>::dot_symbol +
           FileTraits<C>::library_ext();
}

inline std::filesystem::path make_plugin_library_name(const std::filesystem::path& lib_name) {
    return library_prefix().concat(lib_name.filename().native()).concat(library_extension().native());
}

inline std::filesystem::path make_plugin_library_name(const std::filesystem::path& dir_path,
                                                      const std::filesystem::path& lib_name) {
    return dir_path / make_plugin_library_name(lib_name);
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
std::vector<uint8_t> load_binary(const std::string& path);

/**
 * @brief save binary data to file
 * @param path - binary file path to store
 */
void save_binary(const std::string& path, const std::vector<uint8_t>& binary);
void save_binary(const std::string& path, const char* binary, size_t bin_size);

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

template <typename C>
using enableIfSupportedChar =
    typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

template <typename C, typename = enableIfSupportedChar<C>>
inline std::basic_string<C> make_path(const std::basic_string<C>& folder, const std::basic_string<C>& file) {
    if (folder.empty())
        return file;
    return folder + ov::util::FileTraits<C>::file_separator + file;
}
}  // namespace util
}  // namespace ov
