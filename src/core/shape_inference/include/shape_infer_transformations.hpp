// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compare.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace sh_infer {
namespace tr {

/**
 * \brief Trnsform tensor data by cast them to type T
 *
 * \tparam T Type of returned value.
 */
template <class T>
struct Cast {
    constexpr Cast() = default;

    template <class U>
    constexpr T operator()(const U u) const {
        return static_cast<T>(u);
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
    const std::pair<T, T> m_range{std::numeric_limits<T>::min(), std::numeric_limits<T>::max()};

    constexpr InTypeRange() = default;
    constexpr InTypeRange(T min, T max) : m_range{std::move(min), std::move(max)} {};

    template <class U>
    T operator()(const U u) const {
        OPENVINO_ASSERT(cmp::le(m_range.first, u) && cmp::le(u, m_range.second),
                        "Value ",
                        u,
                        " not in range [",
                        m_range.first,
                        ":",
                        m_range.second,
                        "]");
        return static_cast<T>(u);
    }
};
}  // namespace tr
}  // namespace sh_infer
}  // namespace ov
