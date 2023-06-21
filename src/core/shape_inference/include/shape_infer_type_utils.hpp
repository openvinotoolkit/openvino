// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compare.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace util {

/**
 * \brief Transform tensor data by cast them to type T
 *
 * \tparam T Type of returned value.
 */
template <class T>
struct Cast {
    constexpr Cast() = default;

    template <
        class U,
        typename std::enable_if<!std::is_integral<T>::value || !std::is_floating_point<U>::value>::type* = nullptr>
    constexpr T operator()(const U u) const {
        return static_cast<T>(u);
    }

    template <class U,
              typename std::enable_if<std::is_integral<T>::value && std::is_floating_point<U>::value>::type* = nullptr>
    constexpr T operator()(const U u) const {
        return cmp::lt(u, std::numeric_limits<T>::max())
                   ? cmp::lt(u, std::numeric_limits<T>::min()) ? std::numeric_limits<T>::min() : static_cast<T>(u)
                   : std::numeric_limits<T>::max();
    }
};

/**
 * \brief Check if input data is in [T::min(), T::max()] and then cast it to T.
 *
 * \tparam T Type of returned value and used to specified min, max of valid value range.
 *
 * \throws ov::AssertFailure if input value not in type range.
 */
template <class T>
struct InTypeRange {
    const T m_min{std::numeric_limits<T>::lowest()}, m_max{std::numeric_limits<T>::max()};

    constexpr InTypeRange() = default;
    constexpr InTypeRange(const T& min, const T& max) : m_min{min}, m_max{max} {};

    template <class U>
    T operator()(const U u) const {
        OPENVINO_ASSERT(cmp::le(m_min, u) && cmp::le(u, m_max), "Value ", u, " not in range [", m_min, ":", m_max, "]");
        return static_cast<T>(u);
    }
};

}  // namespace util
}  // namespace ov
