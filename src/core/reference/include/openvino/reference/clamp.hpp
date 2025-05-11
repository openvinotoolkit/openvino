// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Clamp operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param min    Minimum value used to clamp input data.
 * @param max    Maximum value used to clamp input data.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void clamp(const T* arg, T* out, const T min, const T max, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
        if (arg[i] < min) {
            out[i] = min;
        } else if (arg[i] > max) {
            out[i] = max;
        } else {
            out[i] = arg[i];
        }
    }
}
}  // namespace reference
}  // namespace ov
