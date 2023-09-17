// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>

#include "openvino/core/type/bfloat16.hpp"
#include "openvino/core/type/float16.hpp"

namespace ov {
namespace cmp {
/** \brief Enumerate bounds to compare */
enum Bound : uint8_t { NONE, LOWER, UPPER, BOTH };

/**
 * \brief Compare if value is between lower and upper bounds.
 *
 * The Between comparator has four modes to check value:
 * - Bound::None  (lower, upper)
 * - Bound::LOWER [lower, upper)
 * - Bound::UPPER (lower, upper]
 * - Bound::BOTH  [lower, upper]
 *
 * \tparam T     Value type to compare.
 * \tparam BMode Compare bounds mode.
 */
template <class T, Bound BMode = Bound::NONE>
class Between {
    const T m_lower_bound, m_upper_bound;

public:
    constexpr Between(const T& lower, const T& upper) : m_lower_bound{lower}, m_upper_bound{upper} {}

    template <Bound B = BMode, typename std::enable_if<B == Bound::NONE>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (lower() < value) && (value < upper());
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::LOWER>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (lower() <= value) && (value < upper());
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::UPPER>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (lower() < value) && (value <= upper());
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::BOTH>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (lower() <= value) && (value <= upper());
    }

    const T& upper() const {
        return m_upper_bound;
    }

    const T& lower() const {
        return m_lower_bound;
    }
};

/**
 * \brief Compare if value is equal to expected.
 *
 * \tparam T  Value type to compare.
 */
template <class T>
class Equal {
    T _exp_value;

public:
    constexpr Equal(const T& exp_value) : _exp_value{exp_value} {}

    constexpr bool operator()(const T& value) const {
        return _exp_value == value;
    }
};

/**
 * \brief Compare if value is less to expected.
 *
 * \tparam T  Value type to compare.
 */
template <class T>
class Less {
    const T m_exp_value{};

public:
    constexpr Less(const T& exp_value) : m_exp_value{exp_value} {}

    constexpr bool operator()(const T& value) const {
        return value < m_exp_value;
    }
};

/**
 * \brief Compare two values (a < b) in safe way against lossy integer conversion.
 *
 * \tparam T Type of a value.
 * \tparam U Type of b value.
 *
 * \param a  Value a.
 * \param b  Value b.
 *
 * \return true if a less b otherwise false.
 */
template <class T,
          class U,
          typename std::enable_if<
              ((std::is_signed<T>::value || std::is_same<T, float16>::value || std::is_same<T, bfloat16>::value) &&
               (std::is_signed<U>::value || std::is_same<U, float16>::value || std::is_same<U, bfloat16>::value)) ||
              (std::is_unsigned<T>::value && std::is_unsigned<U>::value)>::type* = nullptr>
constexpr bool lt(T a, U b) noexcept {
    return a < b;
}

template <class T,
          class U,
          typename std::enable_if<std::is_signed<T>::value && std::is_integral<T>::value &&
                                  std::is_unsigned<U>::value>::type* = nullptr>
constexpr bool lt(T a, U b) noexcept {
    return a < 0 ? true : static_cast<typename std::make_unsigned<T>::type>(a) < b;
}

template <class T,
          class U,
          typename std::enable_if<(std::is_floating_point<T>::value || std::is_same<T, float16>::value ||
                                   std::is_same<T, bfloat16>::value) &&
                                  std::is_unsigned<U>::value>::type* = nullptr>
constexpr bool lt(T a, U b) noexcept {
    return a < 0 ? true : a < b;
}

template <class T,
          class U,
          typename std::enable_if<std::is_unsigned<T>::value && std::is_integral<U>::value &&
                                  std::is_signed<U>::value>::type* = nullptr>
constexpr bool lt(T a, U b) noexcept {
    return b < 0 ? false : a < static_cast<typename std::make_unsigned<U>::type>(b);
}

template <class T,
          class U,
          typename std::enable_if<std::is_unsigned<T>::value &&
                                  (std::is_floating_point<U>::value || std::is_same<U, float16>::value ||
                                   std::is_same<U, bfloat16>::value)>::type* = nullptr>
constexpr bool lt(T a, U b) noexcept {
    return b < 0 ? false : a < b;
}

/**
 * \brief Compare two values (a > b) in safe way against lossy integer conversion.
 *
 * \tparam T Type of a value.
 * \tparam U Type of b value.
 *
 * \param a  Value a.
 * \param b  Value b.
 *
 * \return true if a > b otherwise false.
 */
template <class T, class U>
constexpr bool gt(T a, U b) noexcept {
    return lt(b, a);
}

/**
 * \brief Compare two values (a <= b) in safe way against lossy integer conversion.
 *
 * \tparam T Type of a value.
 * \tparam U Type of b value.
 *
 * \param a  Value a.
 * \param b  Value b.
 *
 * \return true if a <= b otherwise false.
 */
template <class T, class U>
constexpr bool le(T a, U b) noexcept {
    return !gt(a, b);
}

/**
 * \brief Compare two values (a >= b) in safe way against lossy integer conversion.
 *
 * \tparam T Type of a value.
 * \tparam U Type of b value.
 *
 * \param a  Value a.
 * \param b  Value b.
 *
 * \return true if a >= b otherwise false.
 */
template <class T, class U>
constexpr bool ge(T a, U b) noexcept {
    return !lt(a, b);
}
}  // namespace cmp
}  // namespace ov
