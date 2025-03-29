// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Exp operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void exp(const T* arg, T* out, size_t count) {
    std::transform(arg, arg + count, out, [](const T v) {
        return static_cast<T>(std::exp(v));
    });
}
}  // namespace reference
}  // namespace ov
