// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <charconv>
#include <iterator>
#include <locale>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "openvino/util/container_util.hpp"
#include "openvino/util/hash_util.hpp"
#include "openvino/util/math_util.hpp"

namespace ov::util {

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
 * @brief  Helper struct to join container's elements to string using user string as separator.
 * The output string is generated in operator<< and can be used in any context where std::string is expected.
 * Example:
 * std::vector<int> vec = {1, 2, 3};
 * std::cout << Joined{vec, ","} << std::endl; // Output: 1,2,3
 *
 * @tparam Container
 */
template <typename Container>
struct Joined {
    const Container& c;
    std::string_view sep;

    friend std::ostream& operator<<(std::ostream& os, const Joined& jv) {
        auto first = std::begin(jv.c);
        const auto last = std::end(jv.c);
        if (first != last) {
            os << *first;
            for (++first; first != last; ++first)
                os << jv.sep << *first;
        }
        return os;
    }

    operator std::string() const {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }
};

/**
 * @brief Util to create a string from a container's elements joined by a separator.
 *
 * @note If R is std::string the container's elements are joined and returned as a std::string.
 *  If R is std::ostream the container's elements are joined and returned as a Joined<>
 *  for lazy evaluation in stream context.
 *
 * @param c   The input container.
 * @param sep The separator to be used between the elements. Default is ", ".
 * @return A string representation of the container's elements joined by the specified separator.
 */
template <class R = std::string, typename Container>
auto join(const Container& c, std::string_view sep = ", ") {
    if constexpr (std::is_same_v<R, std::string>) {
        return static_cast<std::string>(Joined<Container>{c, sep});
    } else if constexpr (std::is_same_v<R, std::ostream>) {
        return Joined<Container>{c, sep};
    } else {
    }
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
 * @return String containing the vector elements
 */
template <typename T, typename A>
std::string vector_to_string(const std::vector<T, A>& v) {
    return "[ " + join(v) + " ]";
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

std::string to_lower(std::string_view s);

std::string to_upper(std::string_view s);

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
        if (sv == "inf") {
            return std::make_optional(std::numeric_limits<T>::infinity());
        } else if (sv == "-inf") {
            return std::make_optional(-std::numeric_limits<T>::infinity());
        } else if (sv == "nan") {
            return std::make_optional(std::numeric_limits<T>::quiet_NaN());
        } else {
            StringViewStreamBuf buf{sv};
            std::istream stream{&buf};
            stream.imbue(std::locale::classic());
            T value{};
            stream >> value;
            return stream ? std::make_optional<T>(value) : std::nullopt;
        }
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
