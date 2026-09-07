// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <limits>
#include <type_traits>

namespace ov::util {

/**
 * @brief Ceiling integer division.
 * @param x  Dividend.
 * @param y  Divisor.
 * @return Ceil(x / y).
 */
template <typename T>
constexpr T ceil_div(const T& x, const T& y) {
    return (x == 0 ? 0 : (1 + (x - 1) / y));
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

}  // namespace ov::util
