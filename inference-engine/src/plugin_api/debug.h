// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Basic debugging tools
 * @file debug.h
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <ostream>
#include <iterator>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <cstring>

#include "ie_algorithm.hpp"

/**
 * @brief Serializes a `std::vector` to a `std::ostream`
 * @ingroup ie_dev_api_error_debug
 * @param out An output stream
 * @param vec A vector to serialize
 * @return A reference to a `std::stream`
 */
namespace std {
template <typename T>
inline std::ostream& operator<<(std::ostream& out, const std::vector<T>& vec) {
    if (vec.empty()) return std::operator<<(out, "[]");
    out << "[" << vec[0];
    for (unsigned i = 1; i < vec.size(); i++) {
        out << ", " << vec[i];
    }
    return out << "]";
}
}  // namespace std

namespace InferenceEngine {
namespace details {
/**
 * @brief trim from start (in place)
 * @ingroup ie_dev_api_error_debug
 * @param s - string to trim
 */
inline void ltrim(std::string& s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int c) {
                return !std::isspace(c);
            }));
}

/**
 * @brief trim from end (in place)
 * @ingroup ie_dev_api_error_debug
 * @param s - string to trim
 */
inline void rtrim(std::string& s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                         [](int c) {
                             return !std::isspace(c);
                         })
                .base(),
            s.end());
}

/**
 * @brief Trims std::string from both ends (in place)
 * @ingroup ie_dev_api_error_debug
 * @param s A reference to a std::tring to trim
 * @return A reference to a trimmed std::string
 */
inline std::string& trim(std::string& s) {
    ltrim(s);
    rtrim(s);
    return s;
}

/**
 * @brief split string into a vector of substrings
 * @ingroup ie_dev_api_error_debug
 * @param src - string to split
 * @param delimiter - string used as a delimiter
 * @return vector of substrings
 */
inline std::vector<std::string> split(const std::string& src, const std::string& delimiter) {
    std::vector<std::string> tokens;
    std::string tokenBuf;
    size_t prev = 0, pos = 0, srcLength = src.length(), delimLength = delimiter.length();
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
 * @brief create a string representation for a vector of values, without any suffixes or prefixes
 * @ingroup ie_dev_api_error_debug
 * @param vec Vector of values
 * @param glue A separator
 * @return A string representation
 */
template <typename T, typename A>
std::string joinVec(std::vector<T, A> const& vec, std::string const& glue = std::string(",")) {
    if (vec.empty()) return "";
    std::stringstream oss;
    oss << vec[0];
    for (size_t i = 1; i < vec.size(); i++) oss << glue << vec[i];
    return oss.str();
}

/**
 * @brief create a string representation for a vector of values, enclosing text in a square brackets
 * @ingroup ie_dev_api_error_debug
 * @param vec - vector of values
 * @return string representation
 */
template <typename T, typename A>
std::string dumpVec(std::vector<T, A> const& vec) {
    return "[" + joinVec(vec) + "]";
}

/**
 * @brief multiply vector's values
 * @ingroup ie_dev_api_error_debug
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename T, typename A>
T product(std::vector<T, A> const& vec) {
    if (vec.empty()) return 0;
    T ret = vec[0];
    for (size_t i = 1; i < vec.size(); ++i) ret *= vec[i];
    return ret;
}

/**
 * @brief check if vectors contain same values
 * @ingroup ie_dev_api_error_debug
 * @param v1 - first vector
 * @param v2 - second vector
 * @return true if vectors contain same values
 */
template <typename T, typename A>
bool equal(const std::vector<T, A>& v1, const std::vector<T, A>& v2) {
    if (v1.size() != v2.size()) return false;
    for (auto i1 = v1.cbegin(), i2 = v2.cbegin(); i1 != v1.cend(); ++i1, ++i2) {
        if (*i1 != *i2) return false;
    }
    return true;
}

#ifdef _WIN32
# define strncasecmp _strnicmp
#endif

/**
 * @brief Checks whether two `std::string`s are equal
 * @ingroup ie_dev_api_error_debug
 * @param lhs A first `std::string` to compare
 * @param rhs A second `std::string` to compare
 * @param ignoreCase Whether to ignore case-sensitivity, default is `true`
 * @return `True` in case of `std::string`s are equal, `false` otherwise
 */
inline bool equal(const std::string& lhs, const std::string& rhs, bool ignoreCase = true) {
    return (lhs.size() == rhs.size()) && (ignoreCase ? 0 == strncasecmp(lhs.c_str(), rhs.c_str(), lhs.size())
                                                     : 0 == strncmp(lhs.c_str(), rhs.c_str(), lhs.size()));
}

/**
 * @brief check string end with given substring
 * @ingroup ie_dev_api_error_debug
 * @param src - string to check
 * @param with - given substring
 * @return true if string end with given substring
 */
inline bool endsWith(const std::string& src, const char* with) {
    int wl = static_cast<int>(strlen(with));
    int so = static_cast<int>(src.length()) - wl;
    if (so < 0) return false;
    return 0 == strncmp(with, &src[so], wl);
}

/**
 * @brief Converts all upper-case letters in a std::string to lower case
 * @ingroup ie_dev_api_error_debug
 * @param s A std::tring to convert
 * @return An output std::string in lower case
 */
inline std::string tolower(const std::string& s) {
    std::string ret;
    ret.resize(s.length());
    std::transform(s.begin(), s.end(), ret.begin(),
        [](char c) { return static_cast<char>(::tolower(static_cast<int>(c))); });
    return ret;
}
}  // namespace details
}  // namespace InferenceEngine
