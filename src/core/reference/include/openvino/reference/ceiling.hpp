// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/copy.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Ceiling operator (integral types).
 *
 * @param arg    Input pointer to data.
 * @param out    Output pointer to results.
 * @param count  Number of elements in input buffer.
 */
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
void ceiling(const T* arg, T* out, const size_t count) {
    copy(arg, out, count);
}

/**
 * @brief Reference implementation of Ceiling operator (floating point types).
 *
 * @param arg    Input pointer to data.
 * @param out    Output pointer to results.
 * @param count  Number of elements in input buffer.
 */
template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
void ceiling(const T* arg, T* out, const size_t count) {
    std::transform(arg, std::next(arg, count), out, [](const T v) {
        return std::ceil(v);
    });
}
}  // namespace reference
}  // namespace ov
