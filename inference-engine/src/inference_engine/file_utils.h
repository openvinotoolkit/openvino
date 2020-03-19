// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Basic function to work with file system
 * \file file_utils.h
 */
#pragma once

#include <string>
#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# define _WINSOCKAPI_
# include <windows.h>
# include <profileapi.h>
#endif

#ifdef __MACH__
# include <mach/clock.h>
# include <mach/mach.h>
#endif

#include "ie_api.h"
#include "ie_unicode.hpp"
#include "details/os/os_filesystem.hpp"
#include "details/ie_so_pointer.hpp"

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
 * @param fileName - name of the file
 * @return size of the file
 */
INFERENCE_ENGINE_API_CPP(long long) fileSize(const char *fileName);

#ifdef ENABLE_UNICODE_PATH_SUPPORT

inline long long fileSize(const wchar_t* fileName) {
    return fileSize(InferenceEngine::details::wStringtoMBCSstringChar(fileName).c_str());
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Function to get the size of a file. The function supports UNICODE path
 * @param f - string name of the file
 * @return size of the file
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline long long fileSize(const std::basic_string<C> &f) {
    return fileSize(f.c_str());
}

/**
 * @brief check if file with a given filename exists. The function supports UNICODE path
 * @param fileName - given filename
 * @return true is exists
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline bool fileExist(const C * fileName) {
    return fileSize(fileName) >= 0;
}

/**
 * @brief check if file with a given filename exists.  The function supports UNICODE path
 * @param fileName - string with a given filename
 * @return true is exists
 */
template <typename C, typename = InferenceEngine::details::enableIfSupportedChar<C>>
inline bool fileExist(const std::basic_string<C> &fileName) {
    return fileExist(fileName.c_str());
}

/**
 * @brief CPP Interface function to read a file. In case of read error throws an exception. The function supports UNICODE path
 * @param file_name - name of the file to read
 * @param buffer - buffer to read file to
 * @param maxSize - maximum size in bytes to read
 */
INFERENCE_ENGINE_API_CPP(void) readAllFile(const std::string &file_name, void *buffer, size_t maxSize);

/**
 * @brief CPP Interface function to extract path part of a filename
 * @param filepath - filename to extract path part from
 * @return string with path part of the filename
 */
INFERENCE_ENGINE_API_CPP(std::string) folderOf(const std::string &filepath);

/**
 * @brief CPP Interface function to combint path with filename. The function supports UNICODE path
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

/**
* @brief CPP Interface function to check if given filename belongs to shared library
* @param filename - file name to check
* @return true if filename is a shared library filename
*/
inline bool isSharedLibrary(const std::string &fileName) {
    return 0 ==
#ifdef _WIN32
    _strnicmp
#else
    strncasecmp
#endif
    (fileExt(fileName).c_str(), FileTraits<char>::SharedLibraryExt().c_str(),
        FileTraits<char>::SharedLibraryExt().size());
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

/**
 * @brief TODO: description
 * @return TODO: please use c++11 chrono module for time operations
 */
inline long long GetMicroSecTimer() {
#ifdef _WIN32
    static LARGE_INTEGER Frequency = { 0 };
    LARGE_INTEGER timer;
    if (Frequency.QuadPart == 0) QueryPerformanceFrequency(&Frequency);
    QueryPerformanceCounter(&timer);
    return (timer.QuadPart * 1000000) / Frequency.QuadPart;
#else
    struct timespec now;
    #ifdef __MACH__
    clock_serv_t cclock;
    mach_timespec_t mts;
    host_get_clock_service(mach_host_self(), CALENDAR_CLOCK, &cclock);
    clock_get_time(cclock, &mts);
    mach_port_deallocate(mach_task_self(), cclock);
    now.tv_sec = mts.tv_sec;
    now.tv_nsec = mts.tv_nsec;
    #else
    clock_gettime(CLOCK_REALTIME, &now);
    #endif
    return now.tv_sec * 1000000L + now.tv_nsec / 1000;
#endif
}
}  // namespace FileUtils
