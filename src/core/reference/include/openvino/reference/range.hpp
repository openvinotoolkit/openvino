// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <type_traits>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {

/**
 * @brief Reference implementation for Range operator (floating-point types).
 *
 * @param start     Start value.
 * @param step      Step is difference value for consecutive values.
 * @param num_elem  Number of elements to generate
 * @param out       Pointer to output data.
 */
template <typename T>
typename std::enable_if<ov::is_floating_point<T>()>::type range(const T start,
                                                                const T step,
                                                                const size_t num_elem,
                                                                T* out) {
    for (size_t i = 0; i < num_elem; ++i) {
        out[i] = start + (static_cast<T>(i) * (step));
    }
}

/**
 * @brief Reference implementation for Range operator (integral types).
 *
 * @param start     Start value.
 * @param step      Step is difference value for consecutive values.
 * @param num_elem  Number of elements to generate
 * @param out       Pointer to output data.
 */
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type range(const T start,
                                                                const T step,
                                                                const size_t num_elem,
                                                                T* out) {
    auto val = start;
    for (size_t i = 0; i < num_elem; ++i, val += step) {
        out[i] = val;
    }
}
}  // namespace reference
}  // namespace ov
