// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic function to work with file system and UNICDE symbols
 * @file file_utils.h
 */

#pragma once

// clang-format off
#include <string>
#include <cstring>

#include "ie_api.h"
#include "ie_unicode.hpp"
#include "details/os/os_filesystem.hpp"

namespace FileUtils {

template <typename T> struct FileTraits;

#ifdef _WIN32

/// @brief File path separator
const char FileSeparator = '\\';

template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string SharedLibraryPrefix() { return { }; }
    static std::string SharedLibraryExt() { return { "dll" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'\\';
    static std::wstring SharedLibraryPrefix() { return { }; }
    static std::wstring SharedLibraryExt() { return { L"dll" }; }
};
#elif defined __APPLE__
/// @brief File path separator
const char FileSeparator = '/';
template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string SharedLibraryPrefix() { return { "lib" }; }
    static std::string SharedLibraryExt() { return { "dylib" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'/';
    static std::wstring SharedLibraryPrefix() { return { L"lib" }; }
    static std::wstring SharedLibraryExt() { return { L"dylib" }; }
};
#else
/// @brief File path separator
const char FileSeparator = '/';
template<> struct FileTraits<char> {
    constexpr static const auto FileSeparator = ::FileUtils::FileSeparator;
    static std::string SharedLibraryPrefix() { return { "lib" }; }
    static std::string SharedLibraryExt() { return { "so" }; }
};
template<> struct FileTraits<wchar_t> {
    constexpr static const auto FileSeparator = L'/';
    static std::wstring SharedLibraryPrefix() { return { L"lib" }; }
    static std::wstring SharedLibraryExt() { return { L"so" }; }
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
    return fileSize(InferenceEngine::details::wStringtoMBCSstringChar(fileName).c_str());
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
inline std::basic_string<C> makeSharedLibraryName(const std::basic_string<C> &path, const std::basic_string<C> &input) {
    std::basic_string<C> separator(1, FileTraits<C>::FileSeparator);
    if (path.empty())
        separator = {};
    return path + separator + FileTraits<C>::SharedLibraryPrefix() + input + DotSymbol<C>::value + FileTraits<C>::SharedLibraryExt();
}

#ifdef ENABLE_UNICODE_PATH_SUPPORT

using FilePath = std::wstring;

inline std::string fromFilePath(const FilePath & path) {
    return InferenceEngine::details::wStringtoMBCSstringChar(path);
}

inline FilePath toFilePath(const std::string & path) {
    return InferenceEngine::details::multiByteCharToWString(path.c_str());
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
