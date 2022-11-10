// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
}  // namespace cmp
}  // namespace ov
