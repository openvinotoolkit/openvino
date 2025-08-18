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

inline size_t hash_combine(size_t val, const size_t seed) {
    return seed ^ (val + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

inline size_t hash_combine(const std::vector<size_t>& list) {
    size_t seed = 0;
    for (size_t v : list) {
        seed ^= hash_combine(v, seed);
    }
    return seed;
}

inline size_t hash_combine(std::initializer_list<size_t>&& list) {
    size_t seed = 0;
    for (size_t v : list) {
        seed ^= hash_combine(v, seed);
    }
    return seed;
}

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
T product(const std::vector<T, A>& vec) {
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

/**
 * @brief A custom stream buffer that provides read-only access to a string view.
 *
 * This class inherits from `std::streambuf` and is designed to facilitate
 * input operations directly on a `std::string_view` without copying the
 * underlying string data. It allows for efficient reading and seeking
 * operations within the string view.
 *
 * @note This stream buffer is intended for input operations only.
 * @see pyopenvino/utils/utils.hpp for a similar implementation
 */
class StringViewStreamBuf : public std::streambuf {
public:
    explicit StringViewStreamBuf(std::string_view sv) {
        char* begin = const_cast<char*>(sv.data());
        setg(begin, begin, begin + sv.size());
    }

protected:
    pos_type seekoff(off_type off,
                     std::ios_base::seekdir dir,
                     std::ios_base::openmode which = std::ios_base::in) override {
        if (which != std::ios_base::in) {
            return off_type(-1);
        }

        switch (dir) {
        case std::ios_base::beg:
            setg(eback(), eback() + off, egptr());
            break;
        case std::ios_base::end:
            setg(eback(), egptr() + off, egptr());
            break;
        case std::ios_base::cur:
            setg(eback(), gptr() + off, egptr());
            break;
        default:
            return off_type(-1);
        }
        if (gptr() < eback() || gptr() > egptr())
            return off_type(-1);

        return gptr() - eback();
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(pos, std::ios_base::beg, which);
    }
};

/**
 * @brief Multiplies two integral values
 *
 * The result value is not valid if overflow detected.
 *
 * @param T       Type of values to multiply. Must be an integral type.
 * @param x       First value to multiply.
 * @param y       Second value to multiply.
 * @param result  Reference to store result value.
 * @return True if overflow occurs, false otherwise
 */
template <class T>
constexpr bool mul_overflow(T x, T y, T& result) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_mul_overflow(x, y, &result);
#else
    constexpr auto max = std::numeric_limits<T>::max();

    if constexpr (std::is_unsigned_v<T>) {
        if (y > 0 && x > max / y) {
            return true;
        }
    } else {
        constexpr auto min = std::numeric_limits<T>::lowest();
        if ((x > 0 && y > 0 && x > max / y) || (x > 0 && y < 0 && y < min / x) || (x < 0 && y > 0 && x < min / y) ||
            (x < 0 && y < 0 && x < max / y)) {
            return true;
        }
    }
    result = x * y;
    return false;
#endif
}

}  // namespace util
}  // namespace ov
