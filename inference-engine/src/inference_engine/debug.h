// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * \brief Basic debugging tools
 * \file debug.h
 */
#pragma once

#include <cstdlib>
#include <cstdarg>
#include <string>
#include <ctime>
#include <cstdint>
#include <algorithm>
#include <functional>
#include <cctype>
#include <iostream>
#include <sstream>
#include <vector>
#include <iterator>
#include <numeric>
#include <w_unistd.h>
#include "ie_algorithm.hpp"

#ifdef _WIN32
#include <windows.h>

#define POSIX_EPOCH_AS_FILETIME 116444736000000000ULL
#define OPT_USAGE
static void gettimeofday(struct timeval * tp, struct timezone *) {
    SYSTEMTIME  system_time;
    FILETIME    file_time;
    uint64_t    time;

    GetSystemTime(&system_time);
    SystemTimeToFileTime(&system_time, &file_time);

    time = file_time.dwLowDateTime;
    time += static_cast<uint64_t>(file_time.dwHighDateTime) << 32;

    tp->tv_sec = static_cast<long>((time - POSIX_EPOCH_AS_FILETIME) / 10000000L);
    tp->tv_usec = (system_time.wMilliseconds * 1000);
}
#else
#define vsnprintf_s vsnprintf

#include <string.h>
#include <sys/time.h>

#ifndef OPT_USAGE
#ifdef __GNUC__
#define OPT_USAGE __attribute__ ((unused))
#else
#define OPT_USAGE
#endif
#endif
#endif
/// Diff in uSec
inline int64_t operator-(const timeval& lhs, const timeval& rhs) {
    return static_cast<int64_t>(lhs.tv_sec - rhs.tv_sec) * 1000000 + lhs.tv_usec - rhs.tv_usec;
}

namespace InferenceEngine {
namespace details {

/**
* @brief vector serialisation to be used in exception
*/
template <typename T>
inline std::ostream & operator << (std::ostream &out, const std::vector<T> &vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i=1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}


/**
 * @brief trim from start (in place)
 * @param s - string to trim
 */
inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c){
        return !std::isspace(c);
    }));
}

/**
 * @brief trim from end (in place)
 * @param s - string to trim
 */
inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int c) {
        return !std::isspace(c);
    }).base(), s.end());
}

/**
 * @brief trim from both ends (in place)
 * @param s - string to trim
 */
inline std::string &trim(std::string &s) {
    ltrim(s);
    rtrim(s);
    return s;
}

/**
 * @brief split string into a vector of substrings
 * @param src - string to split
 * @param delimiter - string used as a delimiter
 * @return vector of substrings
 */
inline std::vector<std::string> split(const std::string &src, const std::string &delimiter) {
    std::vector<std::string> tokens;
    std::string tokenBuf;
    size_t prev = 0,
            pos = 0,
            srcLength = src.length(),
            delimLength = delimiter.length();
    do {
        pos = src.find(delimiter, prev);
        if (pos == std::string::npos) {
            pos = srcLength;
        }
        tokenBuf = src.substr(prev, pos - prev);
        if (!tokenBuf.empty()) {
            tokens.push_back(tokenBuf);
        }
        prev = pos + delimLength;
    } while (pos < srcLength && prev < srcLength);
    return tokens;
}

/**
 * @brief create a string representation for a vector of values
 * @param vec - vector of values
 * @return string representation
 */
template<typename T, typename A>
std::string dumpVec(std::vector<T, A> const &vec) {
    if (vec.empty()) return "[]";
    std::stringstream oss;
    oss << "[" << vec[0];
    for (size_t i = 1; i < vec.size(); i++) oss << "," << vec[i];
    oss << "]";
    return oss.str();
}

/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template<typename T, typename A>
T product(std::vector<T, A> const &vec) {
    if (vec.empty()) return 0;
    T ret = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) ret *= vec[i];
    return ret;
}

/**
 * @brief check if vectors contain same values
 * @param v1 - first vector
 * @param v2 - second vector
 * @return true if vectors contain same values
 */
template<typename T, typename A>
bool equal(const std::vector<T, A> &v1, const std::vector<T, A> &v2) {
    if (v1.size() != v2.size()) return false;
    for (auto i1 = v1.cbegin(), i2 = v2.cbegin(); i1 != v1.cend(); ++i1, ++i2) {
        if (*i1 != *i2)
            return false;
    }
    return true;
}

inline bool equal(const std::string &lhs, const std::string &rhs, bool ignoreCase = true) {
    return (lhs.size() == rhs.size()) && (ignoreCase ?
        0 == strncasecmp(lhs.c_str(), rhs.c_str(), lhs.size()) :
        0 == strncmp(lhs.c_str(), rhs.c_str(), lhs.size()));
}

/**
 * @brief check string end with given substring
 * @param src - string to check
 * @param with - given substring
 * @return true if string end with given substring
 */
inline bool endsWith(const std::string &src, const char *with) {
    int wl = static_cast<int>(strlen(with));
    int so = static_cast<int>(src.length()) - wl;
    if (so < 0) return false;
    return 0 == strncmp(with, &src[so], wl);
}

/**
* @brief converts all upper-case letters in a string to lower case
* @param s - string to convert
*/
inline std::string tolower(const std::string &s) {
    std::string ret;
    ret.resize(s.length());
    std::transform(s.begin(), s.end(), ret.begin(), ::tolower);
    return ret;
}
}  // namespace details
}  // namespace InferenceEngine

/**
 * @brief print a log message
 * @param isErr - is message an error or not
 * @param level - string containing message level, like "ERROR" or "DEBUG"
 * @param file - file the message was produced by
 * @param line - string in file the message was dispatched from
 * @param msg - format for message + arguments
 */
static void OPT_USAGE print_log(bool isErr, const char *level, const char *file, int line, const char *msg, ...) {
    va_list va;
    va_start(va, msg);
    char buffer[64];
    struct tm tm_info;

    struct timeval tval;
    gettimeofday(&tval, NULL);

    auto outFd = isErr ? stderr : stdout;

#ifdef _WIN32
    time_t timer;
    timer = tval.tv_sec;
    localtime_s(&tm_info, &timer);
#else
    localtime_r(&tval.tv_sec, &tm_info);
#endif

    strftime(buffer, 64, "%Y:%m:%d %H:%M:%S", &tm_info);

    fprintf(outFd, "%s.%06ld [%s] %s:%d : ", buffer, (long)tval.tv_usec, level, file, line);
    vfprintf(outFd, msg, va);
    va_end(va);
    fprintf(outFd, "\r\n");
    fflush(outFd);
}

/**
 * @def LogError
 * @brief Log an error message
 */
#define LogError(...) {print_log(true , "ERROR", __FILE__, __LINE__, ##__VA_ARGS__); }

/**
 * @def LogWarning
 * @brief Log warning message
 */
#define LogWarning(...) { print_log(true , "WARNING", __FILE__, __LINE__, ##__VA_ARGS__); }

/**
 * @def LogInfo
 * @brief Log info message
 */
#define LogInfo(...) { print_log(false, "INFO", __FILE__, __LINE__, ##__VA_ARGS__); }

/**
 * @def LogDebug
 * @brief Log debug message
 */
#define LogDebug(...) { print_log(false, "DEBUG", __FILE__, __LINE__, ##__VA_ARGS__); }

/**
 * @def OssDebug
 * @brief Log oss debug message
 */
#define OssDebug(x) { std::stringstream oss; oss << x ; LogDebug("%s", oss.str().c_str()); }
