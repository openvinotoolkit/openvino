// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <optional>
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

std::string to_lower(const std::string_view s);

std::string to_upper(const std::string_view s);

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

constexpr uint64_t u64_hash_combine(uint64_t h, uint64_t k) {
    // Hash combine formula from boost for uint64_t.
    constexpr uint64_t m = 0xc6a4a7935bd1e995;
    constexpr int r = 47;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;

    return h + 0xe6546b64;
}

constexpr uint64_t u64_hash_combine(uint64_t seed, std::initializer_list<uint64_t>&& values) {
    uint64_t h = seed;
    for (uint64_t k : values) {
        h = u64_hash_combine(h, k);
    }
    return h;
}

/**
 * @brief Return a string view with leading whitespace removed.
 *
 * @param s A string view to trim.
 * @return A string view with leading whitespace removed.
 */
constexpr std::string_view ltrim(const std::string_view s) {
    const auto not_ws_pos = s.find_first_not_of(" \t\n\r\f\v");
    return not_ws_pos == std::string_view::npos ? std::string_view() : s.substr(not_ws_pos);
}

/**
 * @brief Return a string view with trailing whitespace removed.
 * @param s A string view to trim.
 * @return A string view with trailing whitespace removed.
 */
constexpr std::string_view rtrim(const std::string_view s) {
    const auto not_ws_pos = s.find_last_not_of(" \t\n\r\f\v");
    return not_ws_pos == std::string_view::npos ? std::string_view() : s.substr(0, not_ws_pos + 1);
}

/**
 * @brief Return a string view with leading and trailing whitespace removed.
 * @param s A string view to trim.
 * @return A string view with leading and trailing whitespace removed.
 */
constexpr std::string_view trim(const std::string_view s) {
    return ltrim(rtrim(s));
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

/**
 * @brief Parses a string view into a container, optionally validating each field.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container.
 * An optional `field_validator` can be provided to validate each field before insertion.
 *
 * @tparam Container The type of the container to store the parsed values.
 * @tparam FieldValidator A callable type used to validate each field.
 * @param sv The input string view to parse.
 * @param result The container to store the parsed values.
 * @param sep The separator used to split the string view.
 * @param field_validator The callable used to validate each field.
 */
template <class Container, class FieldValidator = bool>
void parse_view_into_container(std::string_view sv,
                               Container& result,
                               std::string_view sep = ",",
                               FieldValidator field_validator = {}) {
    while (!sv.empty()) {
        using V = typename Container::value_type;

        const auto sep_pos = sv.find(sep);
        const auto field = sv.substr(0, sep_pos);
        if constexpr (std::is_invocable_v<FieldValidator, std::string_view>) {
            field_validator(field);
        }

        if constexpr (std::is_arithmetic_v<V>) {
            V value{};
            const auto ec = std::from_chars(field.begin(), field.end(), value).ec;
            result.insert(result.end(), (ec == std::errc() ? value : V{}));
        } else {
            result.insert(result.end(), V{field});
        }

        if (sep_pos == std::string_view::npos) {
            break;
        } else {
            sv = sv.substr(sep_pos + sep.size());
        }
    }
}

/**
 * @brief Splits a string view into a container of string views.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container.
 *
 * @tparam Container The type of the container to store the parsed values.
 * @param sv The input string view to parse.
 * @param sep The separator used to split the string view.
 * @return Container The container with the parsed string views.
 */
template <class Container>
Container split_to_views(std::string_view sv, std::string_view sep = ",") {
    Container result{};
    parse_view_into_container(sv, result, sep);
    return result;
}

/**
 * @brief Splits a string view into a container of string views, optionally validating each field.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container.
 * An optional `field_validator` can be provided to validate each field before insertion.
 *
 * @tparam Container The type of the container to store the parsed values.
 * @tparam FieldValidator A callable type used to validate each field.
 * @param sv The input string view to parse.
 * @param sep The separator used to split the string view.
 * @param field_validator The callable used to validate each field.
 * @return Container The container with the parsed string views.
 */
template <class Container, class FieldValidator>
Container split_to_views(std::string_view sv, std::string_view sep = ",", FieldValidator field_validator = {}) {
    Container result{};
    parse_view_into_container(sv, result, sep, field_validator);
    return result;
}

}  // namespace util
}  // namespace ov
