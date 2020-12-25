// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic function to work with file system and UNICODE symbols
 * @file file_utils.h
 */

#pragma once

// clang-format off
#include <string>
#include <cstring>

#include "ie_api.h"
#include "details/ie_so_pointer.hpp"

namespace FileUtils {

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Conversion from wide character string to a single-byte chain.
 * @param wstr A wide-char string
 * @return A multi-byte string
 */
INFERENCE_ENGINE_API_CPP(std::string) wStringtoMBCSstringChar(const std::wstring& wstr);

/**
 * @brief Conversion from single-byte chain to wide character string.
 * @param str A null-terminated string
 * @return A wide-char string
 */
INFERENCE_ENGINE_API_CPP(std::wstring) multiByteCharToWString(const char* str);

#endif  // ENABLE_UNICODE_PATH_SUPPORT

template <typename T> struct FileTraits;

#ifdef _WIN32

/// @brief File path separator
const char FileSeparator = '\\';

template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string PluginLibraryPrefix() { return { }; }
    static std::string PluginLibraryExt() { return { "dll" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'\\';
    static std::wstring PluginLibraryPrefix() { return { }; }
    static std::wstring PluginLibraryExt() { return { L"dll" }; }
};
#elif defined __APPLE__
/// @brief File path separator
const char FileSeparator = '/';
template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string PluginLibraryPrefix() { return { "lib" }; }
    static std::string PluginLibraryExt() { return { "so" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'/';
    static std::wstring PluginLibraryPrefix() { return { L"lib" }; }
    static std::wstring PluginLibraryExt() { return { L"so" }; }
};
#else
/// @brief File path separator
const char FileSeparator = '/';
template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string PluginLibraryPrefix() { return { "lib" }; }
    static std::string PluginLibraryExt() { return { "so" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'/';
    static std::wstring PluginLibraryPrefix() { return { L"lib" }; }
    static std::wstring PluginLibraryExt() { return { L"so" }; }
};
#endif

/**
 * @brief Interface function to get the size of a file. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - name of the file
 * @return size of the file
 */
INFERENCE_ENGINE_API(long long) fileSize(const char *fileName);

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief      Returns file size for file with UNICODE path name
 * @ingroup    ie_dev_api_file_utils
 *
 * @param[in]  fileName  The file name
 *
 * @return     { description_of_the_return_value }
 */
inline long long fileSize(const wchar_t* fileName) {
    return fileSize(::FileUtils::wStringtoMBCSstringChar(fileName).c_str());
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Function to get the size of a file. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param f - string name of the file
 * @return size of the file
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline long long fileSize(const std::basic_string<C> &f) {
    return fileSize(f.c_str());
}

/**
 * @brief check if file with a given filename exists. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - given filename
 * @return true is exists
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline bool fileExist(const C * fileName) {
    return fileSize(fileName) >= 0;
}

/**
 * @brief check if file with a given filename exists.  The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param fileName - string with a given filename
 * @return true is exists
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline bool fileExist(const std::basic_string<C> &fileName) {
    return fileExist(fileName.c_str());
}

/**
 * @brief CPP Interface function to combint path with filename. The function supports UNICODE path
 * @ingroup ie_dev_api_file_utils
 * @param folder - path to add filename to
 * @param file - filename to add to path
 * @return string with combination of the path and the filename divided by file separator
 */

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline std::basic_string<C> makePath(const std::basic_string<C> &folder, const std::basic_string<C> &file) {
    if (folder.empty())
        return file;
    return folder + FileTraits<C>::FileSeparator + file;
}

template <typename C> struct DotSymbol;
template <> struct DotSymbol<char> { constexpr static const char value = '.'; };
template <> struct DotSymbol<wchar_t> { constexpr static const wchar_t value = L'.'; };

/**
 * @brief CPP Interface function to extract extension from filename
 * @ingroup ie_dev_api_file_utils
 * @param filename - string with the name of the file which extension should be extracted
 * @return string with extracted file extension
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline std::basic_string<C> fileExt(const std::basic_string<C> &filename) {
    auto pos = filename.rfind(DotSymbol<C>::value);
    if (pos == std::string::npos)
        return {};
    return filename.substr(pos + 1);
}

template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline std::basic_string<C> makePluginLibraryName(const std::basic_string<C> &path, const std::basic_string<C> &input) {
    std::basic_string<C> separator(1, FileTraits<C>::FileSeparator);
    if (path.empty())
        separator = {};
    return path + separator + FileTraits<C>::PluginLibraryPrefix() + input + DotSymbol<C>::value + FileTraits<C>::PluginLibraryExt();
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

using FilePath = std::wstring;

inline std::string fromFilePath(const FilePath & path) {
    return ::FileUtils::wStringtoMBCSstringChar(path);
}

inline FilePath toFilePath(const std::string & path) {
    return ::FileUtils::multiByteCharToWString(path.c_str());
}

#else

using FilePath = std::string;

inline std::string fromFilePath(const FilePath & path) {
    return path;
}

inline FilePath toFilePath(const std::string & path) {
    return path;
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

}  // namespace FileUtils
// clang-format on

namespace InferenceEngine {

/**
 * @brief   Returns a path to Inference Engine library
 * @ingroup ie_dev_api_file_utils
 * @return  A `std::string` path to Inference Engine library
 */
INFERENCE_ENGINE_API_CPP(std::string) getIELibraryPath();

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief   Returns a unicode path to Inference Engine library
 * @ingroup ie_dev_api_file_utils
 * @return  A `std::wstring` path to Inference Engine library
 */
INFERENCE_ENGINE_API_CPP(std::wstring) getIELibraryPathW();

inline ::FileUtils::FilePath getInferenceEngineLibraryPath() {
    return getIELibraryPathW();
}

#else

inline ::FileUtils::FilePath getInferenceEngineLibraryPath() {
    return getIELibraryPath();
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

}  // namespace InferenceEngine
