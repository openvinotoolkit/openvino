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

namespace FileUtils {
#ifdef _WIN32
/// @brief File path separator
const char FileSeparator = '\\';
const char SharedLibraryExt[] = "dll";
#elif __APPLE__
const char SharedLibraryExt[] = "dylib";
/// @brief File path separator
const char FileSeparator = '/';
#else
const char SharedLibraryExt[] = "so";
/// @brief File path separator
const char FileSeparator = '/';
#endif
/// @brief Alternative file path separator
const char FileSeparator2 = '/';  // second option

/**
 * @brief Interface function to get the size of a file
 * @param fileName - name of the file
 * @return size of the file
 */
INFERENCE_ENGINE_API_CPP(long long) fileSize(const char *fileName);

/**
 * @brief Function to get the size of a file
 * @param f - string name of the file
 * @return size of the file
 */
inline long long fileSize(const std::string &f) {
    return fileSize(f.c_str());
}

/**
 * @brief check if file with a given filename exists
 * @param fileName - given filename
 * @return true is exists
 */
inline bool fileExist(const char *fileName) {
    return fileSize(fileName) >= 0;
}

/**
 * @brief check if file with a given filename exists
 * @param fileName - string with a given filename
 * @return true is exists
 */
inline bool fileExist(const std::string &fileName) {
    return fileExist(fileName.c_str());
}

/**
 * @brief CPP Interface function to read a file. In case of read error throws an exception
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
 * @brief CPP Interface function to combint path with filename
 * @param folder - path to add filename to
 * @param file - filename to add to path
 * @return string with combination of the path and the filename divided by file separator
 */
INFERENCE_ENGINE_API_CPP(std::string) makePath(const std::string &folder, const std::string &file);

/**
 * @brief CPP Interface function to remove file extension
 * @param filepath - filename with extension
 * @return string containing filename without extension
 */
INFERENCE_ENGINE_API_CPP(std::string) fileNameNoExt(const std::string &filepath);

/**
 * @brief CPP Interface function to extract extension from filename
 * @param filename - name of the file which extension should be extracted
 * @return string with extracted file extension
 */
INFERENCE_ENGINE_API_CPP(std::string) fileExt(const char *filename);

/**
* @brief CPP Interface function to extract extension from filename
* @param filename - string with the name of the file which extension should be extracted
* @return string with extracted file extension
*/
INFERENCE_ENGINE_API_CPP(std::string) fileExt(const std::string &filename);

/**
* @brief CPP Interface function to check if given filename belongs to shared library
* @param filename - file name to check
* @return true if filename is a shared library filename
*/
INFERENCE_ENGINE_API_CPP(bool) isSharedLibrary(const std::string &fileName);

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
