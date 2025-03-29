// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <utility>

#include "openvino/reference/autobroadcast_binop.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
constexpr T mod(const T x, const T y) {
    return x % y;
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T mod(const T x, const T y) {
    return x - (std::trunc(x / y) * y);
}

/**
 * @brief Estimates division remainder `[v1, v2] % m = [r0, r1]` as interval.
 *
 * Assumes that ` 0 <= v1 <= v2 and m != 0`, in other cases result is undefined behaviour.
 * The result interval estimate minimum and maximum but is not true that value can be any value between min and max.
 * e.g.
 *  - [4,6] % 5 = [0, 4], but in fact accurate result is set of [0,1,4]

 * @param v1 Minimum of value interval.
 * @param v2 Maximum of value interval.
 * @param m  Modulo divisor.
 * @return Remainder of division as interval range.
 */
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
std::pair<T, T> mod_interval_value(const T v1, const T v2, const T m) {
    const auto v_diff = v2 - v1;
    auto r = std::make_pair(func::mod(v1, m), func::mod(v2, m));

    if ((r.second < r.first) || ((v_diff != T{0}) && (v_diff >= m))) {
        r.first = T{0};
        r.second = m - T{1};
    }
    return r;
}

/**
 * @brief Estimates division reminder of `[v1, v2] & [m1, m2] = [r0, r1]` as interval.
 *
 * * Assumes that ` 0 <= v1 <= v2 and 0 < m1 <= m2`, in other cases result is undefined behaviour.
 *
 * @param v1 Minimum of value interval.
 * @param v2 Maximum of value interval.
 * @param m1 Minimum of modulo divisor.
 * @param m2 Maximum of modulo divisor.
 * @return Remainder of division as interval range.
 */
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
std::pair<T, T> mod_interval(const T v1, const T v2, const T m1, const T m2) {
    auto r = mod_interval_value(v1, v2, m1);
    if (v2 != 0) {
        if (m1 != m2) {
            const auto v_diff = v2 - v1;
            const auto m_diff = m2 - m1;

            auto r2 = mod_interval_value(v1, v2, m2);
            r.first = std::min(r.first, r2.first);
            r.second = std::max(r.second, r2.second);

            if (v_diff == T{0} && m_diff != T{1}) {
                const T v2_half = v2 / T{2};
                if ((m1 < v2_half) || ((m1 < v2) && (v2 < m2))) {
                    r.first = T{0};

                    if ((v2_half < m2) && (m2 < v2)) {
                        const T v2_half_next = v2_half + T{1};
                        r.second = func::mod(v2, v2_half_next);
                    } else {
                        r.second = m2 - T{1};
                    }
                }
            }
        }
    }
    return r;
}
}  // namespace func

/**
 * @brief Reference implementation of binary elementwise Mod operator.
 *
 * @param arg0            Iterator to input 0 data.
 * @param arg1            Iterator to input 1 data.
 * @param out             Iterator to output data.
 * @param arg_shape0      Input 0 shape.
 * @param arg_shape1      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <class InputIt, class OutputIt>
void mod(InputIt arg0,
         InputIt arg1,
         OutputIt out,
         const Shape& arg_shape0,
         const Shape& arg_shape1,
         const op::AutoBroadcastSpec& broadcast_spec) {
    using T = typename std::iterator_traits<OutputIt>::value_type;
    autobroadcast_binop(arg0, arg1, out, arg_shape0, arg_shape1, broadcast_spec, func::mod<T>);
}
}  // namespace reference
}  // namespace ov
