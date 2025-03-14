// Copyright (C) 2017-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"

namespace cldnn {

struct primitive;

namespace meta {

// helper struct to tell wheter type T is any of given types U...
// termination case when U... is empty -> return std::false_type
template <class T, class... U>
struct is_any_of : public std::false_type {};

// helper struct to tell whether type is any of given types (U, Rest...)
// recurrence case when at least one type U is present -> returns std::true_type if std::same<T, U>::value is true,
// otherwise call is_any_of<T, Rest...> recurrently
template <class T, class U, class... Rest>
struct is_any_of<T, U, Rest...> : public std::conditional_t<std::is_same_v<T, U>, std::true_type, is_any_of<T, Rest...>> {};

template <class T>
struct always_false : public std::false_type {};

template <typename Ty, Ty Val>
struct always_false_ty_val : public std::false_type {};

template <typename Ty, Ty... Vals>
struct val_tuple {};

template <bool... Values>
struct all : public std::true_type {};

template <bool Val, bool... Values>
struct all<Val, Values...> : public std::integral_constant<bool, Val && all<Values...>::value> {};

template <class T>
struct is_primitive : public std::integral_constant<bool,
                                                    std::is_base_of_v<primitive, T> && !std::is_same_v<primitive, std::remove_cv_t<T>> &&
                                                        std::is_same_v<T, std::remove_cv_t<T>>> {};

}  // namespace meta

/// @cond CPP_HELPERS

/// @defgroup cpp_helpers Helpers
/// @{

template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> align_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? size : size - (size % align) + align);
}

template <typename T>
std::enable_if_t<std::is_integral_v<T>, T> pad_to(T size, size_t align) {
    return static_cast<T>((size % align == 0) ? 0 : align - (size % align));
}

template <typename T>
std::enable_if_t<std::is_integral_v<T>, bool> is_aligned_to(T size, size_t align) {
    return !(size % align);
}

/// Computes ceil(@p val / @p divider) on unsigned integral numbers.
///
/// Computes division of unsigned integral numbers and rounds result up to full number (ceiling).
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1   Type of @p val. Type must be integral (SFINAE).
/// @tparam T2   Type of @p divider. Type must be integral (SFINAE).
///
/// @param val       Divided value. If value is signed, it will be converted to corresponding unsigned type.
/// @param divider   Divider value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   Result of ceil(@p val / @p divider). The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto ceil_div(T1 val, T2 divider) -> std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                                                                decltype(std::declval<std::make_unsigned_t<T1>>() / std::declval<std::make_unsigned_t<T2>>())> {
    using UT1 = std::make_unsigned_t<T1>;
    using UT2 = std::make_unsigned_t<T2>;
    using RetT = decltype(std::declval<UT1>() / std::declval<UT2>());

    return static_cast<RetT>((static_cast<UT1>(val) + static_cast<UT2>(divider) - 1U) / static_cast<UT2>(divider));
}

/// Rounds @p val to nearest multiply of @p rounding that is greater or equal to @p val.
///
/// The function works for unsigned integrals only. Signed integrals are converted to corresponding
/// unsigned ones.
///
/// @tparam T1       Type of @p val. Type must be integral (SFINAE).
/// @tparam T2       Type of @p rounding. Type must be integral (SFINAE).
///
/// @param val        Value to round up. If value is signed, it will be converted to corresponding unsigned type.
/// @param rounding   Rounding value. If value is signed, it will be converted to corresponding unsigned type.
///
/// @return   @p val rounded up to nearest multiply of @p rounding. The type of result is determined as if in normal integral
///           division, except each operand is converted to unsigned type if necessary.
template <typename T1, typename T2>
constexpr auto round_up_to(T1 val,
                           T2 rounding) -> std::enable_if_t<std::is_integral_v<T1> && std::is_integral_v<T2>,
                                                            decltype(std::declval<std::make_unsigned_t<T1>>() / std::declval<std::make_unsigned_t<T2>>())> {
    using UT1 = std::make_unsigned_t<T1>;
    using UT2 = std::make_unsigned_t<T2>;
    using RetT = decltype(std::declval<UT1>() / std::declval<UT2>());

    return static_cast<RetT>(ceil_div(val, rounding) * static_cast<UT2>(rounding));
}

template <typename derived_type, typename base_type, std::enable_if_t<std::is_base_of_v<base_type, derived_type>, int> = 0>
inline derived_type* downcast(base_type* base) {
    if (auto casted = dynamic_cast<derived_type*>(base)) {
        return casted;
    }

    OPENVINO_THROW("Unable to cast pointer from base (", typeid(base_type).name(), ") ", "type to derived (", typeid(derived_type).name(), ") type");
}

template <typename derived_type, typename base_type, std::enable_if_t<std::is_base_of_v<base_type, derived_type>, int> = 0>
inline derived_type& downcast(base_type& base) {
    try {
        return dynamic_cast<derived_type&>(base);
    } catch (std::bad_cast& /* ex */) {
        throw std::runtime_error("Unable to cast reference from base to derived type");
    }
    throw std::runtime_error("downcast failed with unhandled exception");
}

template <typename T>
inline bool all_ones(const std::vector<T> vec) {
    return std::all_of(vec.begin(), vec.end(), [](const T& val) {
        return val == 1;
    });
}

template <typename T>
inline bool all_zeroes(const std::vector<T> vec) {
    return std::all_of(vec.begin(), vec.end(), [](const T& val) {
        return val == 0;
    });
}

template <typename T>
inline bool all_not_zeroes(const std::vector<T> vec) {
    return std::all_of(vec.begin(), vec.end(), [](const T& val) {
        return val != 0;
    });
}

template <typename T>
inline bool any_one(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) {
        return val == 1;
    });
}

template <typename T>
inline bool any_zero(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) {
        return val == 0;
    });
}

template <typename T>
inline bool any_not_one(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) {
        return val != 1;
    });
}

template <typename T>
inline bool any_not_zero(const std::vector<T> vec) {
    return std::any_of(vec.begin(), vec.end(), [](const T& val) {
        return val != 0;
    });
}

template <typename T>
inline bool one_of(const T& val, const std::vector<T>& vec) {
    return std::any_of(vec.begin(), vec.end(), [&val](const T& v) {
        return static_cast<T>(v) == val;
    });
}
template <typename T, typename U, size_t N, std::enable_if_t<std::is_convertible_v<T, U>>* = nullptr>
inline bool one_of(const T& val, const std::array<U, N>& vec) {
    return std::any_of(vec.begin(), vec.end(), [&val](const U& v) {
        return static_cast<T>(v) == val;
    });
}

template <typename T, typename U, std::enable_if_t<std::is_convertible_v<T, U>>* = nullptr>
inline bool one_of(const T& val, const std::initializer_list<U>& vec) {
    return std::any_of(vec.begin(), vec.end(), [&val](const U& v) {
        return static_cast<T>(v) == val;
    });
}

template <typename T, typename P>
constexpr bool everyone_is(T val, P item) {
    return val == item;
}
template <typename T, typename P, typename... Args>
constexpr bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

// Helpers to get string for types that have operator<< defined
template <typename T>
inline std::string to_string(const T& v) {
    std::stringstream s;
    s << v;
    return s.str();
}

// The following code is derived from Boost C++ library
// Copyright 2005-2014 Daniel James.
// Distributed under the Boost Software License, Version 1.0. (See http://www.boost.org/LICENSE_1_0.txt)
template <typename T, std::enable_if_t<!std::is_enum_v<T>, int> = 0>
static size_t hash_combine(size_t seed, const T& v) {
    return seed ^= std::hash<T>{}(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <typename T, std::enable_if_t<std::is_enum_v<T>, int> = 0>
static size_t hash_combine(size_t seed, const T& v) {
    using underlying_t = typename std::underlying_type<T>::type;
    return hash_combine(seed, static_cast<underlying_t>(v));
}

template <class It>
static size_t hash_range(size_t seed, It first, It last) {
    for (; first != last; ++first) {
        seed = hash_combine(seed, *first);
    }
    return seed;
}

/// @}
/// @endcond
/// @}
}  // namespace cldnn

namespace ov::intel_gpu {
namespace detail {
template <bool do_move, typename T, typename U, std::size_t N, std::size_t... I>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array_impl(U (&values)[N], std::index_sequence<I...> /*unused*/) noexcept {
    if constexpr (do_move) {
        return {{static_cast<T>(std::move(values[I]))...}};
    }
    return {{ static_cast<T>(values[I])...}};
}
}  // namespace detail

template <typename T, typename U, std::size_t N, std::enable_if_t<std::is_convertible_v<T, U>, bool> = true>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array(U (&values)[N]) noexcept {
    static_assert(N > 0, "[GPU] An array must not be empty");
    return detail::to_array_impl<false, T>(values, std::make_index_sequence<N>());
}
template <typename T, typename U, std::size_t N, std::enable_if_t<std::is_convertible_v<T, U>, bool> = true>
[[nodiscard]] constexpr std::array<std::remove_cv_t<T>, N> to_array(U (&&values)[N]) noexcept {
    static_assert(N > 0, "[GPU] An array must not be empty");
    return detail::to_array_impl<true, T>(values, std::make_index_sequence<N>());
}
}  // namespace ov::intel_gpu
