// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of LogicalNot operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void logical_not(const T* arg, T* out, const size_t count) {
    std::transform(arg, std::next(arg, count), out, std::logical_not<T>());
}
}  // namespace reference
}  // namespace ov
