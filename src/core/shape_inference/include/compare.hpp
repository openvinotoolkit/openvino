// Copyright (C) 2018-2022 Intel Corporation
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
    T _lower_bound, _upper_bound;

public:
    constexpr Between(const T& lower, const T& upper) : _lower_bound{lower}, _upper_bound{upper} {}

    template <Bound B = BMode, typename std::enable_if<B == Bound::NONE>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (_lower_bound < value) && (value < _upper_bound);
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::LOWER>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (_lower_bound <= value) && (value < _upper_bound);
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::UPPER>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (_lower_bound < value) && (value <= _upper_bound);
    }

    template <Bound B = BMode, typename std::enable_if<B == Bound::BOTH>::type* = nullptr>
    constexpr bool operator()(const T& value) const {
        return (_lower_bound <= value) && (value <= _upper_bound);
    }
};
}  // namespace cmp
}  // namespace ov
