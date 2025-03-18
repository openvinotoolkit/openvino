// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace ov {
namespace util {

/**
 * @brief Join container's elements to string using user string as separator.
 *
 * @param container  Element to make joined string.
 * @param sep        User string used as separator. Default ", ".
 * @return Joined elements as string.
 */
template <typename Container>
std::string join(const Container& container, const std::string& sep = ", ") {
    std::ostringstream ss;
    auto first = std::begin(container);
    const auto last = std::end(container);
    if (first != last) {
        ss << *first;
        ++first;
        for (; first != last; ++first) {
            ss << sep << *first;
        }
    }
    return ss.str();
}

/**
 * @brief Stringify the input vector.
 *
 *  The vector is converted to the string as "[ element 0, element 1, ..., element N ]".
 *  Examples:
 *  - std::vector<int>{1,3,5} -> "[ 1, 3, 5 ]"
 *  - std::vector<int>{}      -> "[  ]"
 *
 * @param v  Vector to be converted
 * @return String contains
 */
template <typename T, typename A>
std::string vector_to_string(const std::vector<T, A>& v) {
    return "[ " + ov::util::join(v) + " ]";
}

std::string to_lower(const std::string& s);

std::string to_upper(const std::string& s);

size_t hash_combine(const std::vector<size_t>& list);

/**
 * @brief trim from start (in place)
 * @param s - string to trim
 */
inline std::string ltrim(const std::string& s) {
    std::string ret = s;
    ret.erase(ret.begin(), std::find_if(ret.begin(), ret.end(), [](int c) {
                  return !std::isspace(c);
              }));
    return ret;
}

/**
 * @brief trim from end (in place)
 * @param s - string to trim
 */
inline std::string rtrim(const std::string& s) {
    std::string ret = s;
    ret.erase(std::find_if(ret.rbegin(),
                           ret.rend(),
                           [](int c) {
                               return !std::isspace(c);
                           })
                  .base(),
              ret.end());
    return ret;
}

/**
 * @brief Trims std::string from both ends (in place)
 * @ingroup ov_dev_api_error_debug
 * @param s A reference to a std::tring to trim
 * @return A reference to a trimmed std::string
 */
inline std::string trim(const std::string& s) {
    std::string ret = ltrim(s);
    ret = rtrim(ret);
    return ret;
}

/**
 * @brief check string end with given substring
 * @param src - string to check
 * @param with - given substring
 * @return true if string end with given substring
 */
inline bool ends_with(const std::string& src, const char* with) {
    int wl = static_cast<int>(strlen(with));
    int so = static_cast<int>(src.length()) - wl;
    if (so < 0)
        return false;
    return 0 == strncmp(with, &src[so], wl);
}

/**
 * @brief check string/wstring end with given substring
 * @param src - string/wstring to check
 * @param with - given substring
 * @return true if string end with given substring
 */
template <typename T>
inline bool ends_with(const std::basic_string<T>& str, const std::basic_string<T>& suffix) {
    return str.length() >= suffix.length() && 0 == str.compare(str.length() - suffix.length(), suffix.length(), suffix);
}

std::vector<std::string> split(const std::string& s, char delimiter, bool trim = false);

template <typename T>
T ceil_div(const T& x, const T& y) {
    return (x == 0 ? 0 : (1 + (x - 1) / y));
}

/**
 * @brief Checks if container contains the specific value.
 *
 * @param container  The container of elements to examine.
 * @param value      Value to compare the elements to.
 * @return True if value found in the container, false otherwise.
 */
template <typename R, typename V>
bool contains(const R& container, const V& value) {
    return std::find(std::begin(container), std::end(container), value) != std::end(container);
}

/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename T, typename A>
T product(std::vector<T, A> const& vec) {
    return vec.empty() ? T{0} : std::accumulate(vec.begin(), vec.end(), T{1}, std::multiplies<T>());
}

/**
 * @brief Associative containers doesnt work with remove_if algorithm
 * @tparam ContainerT
 * @tparam PredicateT
 * @param data An associative container
 * @param predicate A predicate to remove values conditionally
 */
template <typename Container, typename PredicateT>
inline void erase_if(Container& data, const PredicateT& predicate) {
    for (auto it = std::begin(data); it != std::end(data);) {
        if (predicate(*it)) {
            it = data.erase(it);
        } else {
            ++it;
        }
    }
}

std::string filter_lines_by_prefix(const std::string& str, const std::string& prefix);

template <class T = void, class... Args>
constexpr std::array<std::conditional_t<std::is_void_v<T>, std::common_type_t<Args...>, T>, sizeof...(Args)> make_array(
    Args&&... args) {
    return {std::forward<Args>(args)...};
}

#if defined(_WIN32)
bool may_i_use_dynamic_code();
#else
constexpr bool may_i_use_dynamic_code() {
    return true;
}
#endif

}  // namespace util
}  // namespace ov
