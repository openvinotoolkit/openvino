// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

#include "openvino/reference/copy.hpp"
#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {

template <class T>
bool is_negative(const T v) {
    return v < T{0};
}
}  // namespace func

/**
 * @brief Reference implementation of ReLU operator (signed values).
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T, typename std::enable_if<ov::is_floating_point<T>() || std::is_signed<T>::value>::type* = nullptr>
void relu(const T* arg, T* out, const size_t count) {
    std::replace_copy_if(arg, arg + count, out, func::is_negative<T>, T{0});
}

/**
 * @brief Reference implementation of ReLU operator (unsigned).
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T, typename std::enable_if<std::is_unsigned<T>::value>::type* = nullptr>
void relu(const T* arg, T* out, const size_t count) {
    copy(arg, out, count);
}
}  // namespace reference
}  // namespace ov
