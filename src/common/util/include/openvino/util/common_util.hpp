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

namespace ov::util {
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

std::string to_lower(std::string_view s);

std::string to_upper(std::string_view s);

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
constexpr std::string_view ltrim(std::string_view s) {
    const auto not_ws_pos = s.find_first_not_of(" \t\n\r\f\v");
    return not_ws_pos == std::string_view::npos ? std::string_view() : s.substr(not_ws_pos);
}

/**
 * @brief Return a string view with trailing whitespace removed.
 * @param s A string view to trim.
 * @return A string view with trailing whitespace removed.
 */
constexpr std::string_view rtrim(std::string_view s) {
    const auto not_ws_pos = s.find_last_not_of(" \t\n\r\f\v");
    return not_ws_pos == std::string_view::npos ? std::string_view() : s.substr(0, not_ws_pos + 1);
}

/**
 * @brief Return a string view with leading and trailing whitespace removed.
 * @param s A string view to trim.
 * @return A string view with leading and trailing whitespace removed.
 */
constexpr std::string_view trim(std::string_view s) {
    return ltrim(rtrim(s));
}

/**
 * @brief check string end with given substring
 * @param src - string to check
 * @param with - given substring
 * @return true if string end with given substring
 * @{
 */
constexpr bool ends_with(std::string_view src, std::string_view with) {
    return src.size() >= with.size() && src.substr(src.size() - with.size()) == with;
}

template <typename T>
constexpr bool ends_with(std::basic_string_view<T> src, std::basic_string_view<T> with) {
    return src.size() >= with.size() && src.substr(src.size() - with.size()) == with;
}
/** @} */

template <typename T>
constexpr T ceil_div(const T& x, const T& y) {
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
constexpr bool contains(const R& container, const V& value) {
    return std::find(std::begin(container), std::end(container), value) != std::end(container);
}

/**
 * @brief multiply vector's values
 * @param vec - vector with values
 * @return result of multiplication
 */
template <typename Container>
auto product(const Container& container) {
    using T = typename Container::value_type;
    return container.empty() ? T{0} : std::accumulate(container.begin(), container.end(), T{1}, std::multiplies<T>());
}

/**
 * @brief Removes elements from the container that satisfy the given predicate.
 *
 * @param data      The container from which to remove elements.
 * @param predicate A callable that check element and return true if the element should be removed, or false otherwise.
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

/**
 * @brief Filters lines in a string view based on a given prefix.
 *
 * This function iterates through each line in the input string view and returns a string containing
 * only the lines that start with the specified `prefix`.
 *
 * @param sv     The input string view containing multiple lines.
 * @param prefix The prefix to filter lines by.
 * @return A string containing only the lines that start with the given prefix.
 */
std::string filter_lines_by_prefix(std::string_view sv, std::string_view prefix);

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
 * @brief Adds two integral values
 *
 * The result value is not valid if overflow detected.
 *
 * @param T       Type of values to add. Must be an integral type.
 * @param x       First value to add.
 * @param y       Second value to add.
 * @param result  Reference to store result value.
 * @return True if overflow occurs, false otherwise
 */
template <class T>
constexpr bool add_overflow(T x, T y, T& result) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_add_overflow(x, y, &result);
#else
    constexpr auto max = std::numeric_limits<T>::max();

    if constexpr (std::is_unsigned_v<T>) {
        if (x > max - y) {
            return true;
        }
    } else {
        constexpr auto min = std::numeric_limits<T>::lowest();
        if ((y > 0 && x > max - y) || (y < 0 && x < min - y)) {
            return true;
        }
    }
    result = x + y;
    return false;
#endif
}

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
 * @brief This function attempts to parse the input string view `sv` into a number of type `T`.
 *
 * @tparam T The type of the number to convert to. Must be an arithmetic type.
 * @param sv The string view to convert.
 * @return std::optional<T> The parsed number if successful, or `std::nullopt` if the conversion fails.
 */
template <class T>
std::optional<T> view_to_number(std::string_view sv) noexcept {
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type");
    if constexpr (std::is_integral_v<T>) {
        T value{};
        const auto result = std::from_chars(sv.data(), sv.data() + sv.size(), value);
        return result.ec == std::errc() ? std::make_optional(value) : std::nullopt;
    } else {
        StringViewStreamBuf buf{sv};
        std::istream stream{&buf};
        stream.imbue(std::locale::classic());
        T value{};
        stream >> value;
        return stream ? std::make_optional<T>(value) : std::nullopt;
    }
}

/**
 * @brief Transforms a string view into a container.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container.
 * An optional unary transformation can be applied to the field before insertion.
 *
 * @param sv        The input string view to parse.
 * @param output_it The iterator to store the parsed values.
 * @param sep       The separator used to split the string view.
 * @param unary     The callable used to transform each field. Defaults to no transformation.
 * @return Iterator The output iterator after inserting the parsed values.
 */
template <typename Iterator, typename UnaryOp = std::nullptr_t>
constexpr Iterator view_transform(std::string_view sv, Iterator output_it, std::string_view sep, UnaryOp unary = {}) {
    for (bool has_next = !sv.empty(); has_next; ++output_it) {
        const auto sep_pos = sv.find(sep);
        if constexpr (const auto field = sv.substr(0, sep_pos); std::is_same_v<UnaryOp, std::nullptr_t>) {
            *output_it = field;
        } else {
            *output_it = unary(field);
        }
        has_next = sep_pos != std::string_view::npos;
        sv = has_next ? sv.substr(sep_pos + sep.size()) : std::string_view{};
    }
    return output_it;
}

/**
 * @brief Transforms a string view into a container.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container if predicate returns true for the field.
 * An optional unary transformation can be applied to the field before insertion.
 *
 * @param sv        The input string view to parse.
 * @param output_it The iterator to store the parsed values.
 * @param sep       The separator used to split the string view.
 * @param predicate The callable used to validate each field.
 * @param unary     The callable used to transform each field. Defaults to no transformation.
 * @return Iterator The output iterator after inserting the parsed values.
 */
template <typename Iterator, typename Predicate, typename UnaryOp = std::nullptr_t>
constexpr Iterator view_transform_if(std::string_view sv,
                                     Iterator output_it,
                                     std::string_view sep,
                                     Predicate predicate,
                                     UnaryOp unary = {}) {
    for (bool has_next = !sv.empty(); has_next; ++output_it) {
        const auto sep_pos = sv.find(sep);
        if (const auto field = sv.substr(0, sep_pos); predicate(field)) {
            if constexpr (std::is_same_v<UnaryOp, std::nullptr_t>) {
                *output_it = field;
            } else {
                *output_it = unary(field);
            }
        }
        has_next = sep_pos != std::string_view::npos;
        sv = has_next ? sv.substr(sep_pos + sep.size()) : std::string_view{};
    }
    return output_it;
}

/**
 * @brief Splits a string view into a vector of string views, optionally validating each field.
 *
 * This function splits the input string view `sv` using the specified separator `sep` and inserts the parsed values
 * into the provided `result` container.
 * An optional `predicate` can be provided to validate each field before insertion.
 *
 * @tparam Predicate A callable type used to validate each field. Defaults to `std::nullptr_t`, no validation.
 * @param sv        The input string view to parse.
 * @param sep       The separator used to split the string view. Defaults to ",".
 * @param predicate The callable used to validate each field. Defaults to no validation.
 * @return Container The container with the parsed string views.
 */
template <typename Predicate = std::nullptr_t>
std::vector<std::string_view> split(std::string_view sv, std::string_view sep = ",", Predicate predicate = {}) {
    std::vector<std::string_view> result{};
    if constexpr (std::is_same_v<Predicate, std::nullptr_t>) {
        view_transform(sv, std::back_inserter(result), sep);
    } else {
        view_transform_if(sv, std::back_inserter(result), sep, predicate);
    }
    return result;
}
}  // namespace ov::util
