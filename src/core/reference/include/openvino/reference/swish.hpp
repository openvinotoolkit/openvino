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
 * @brief Reference implementation of Swish operator.
 *
 * @param arg    Input pointer to data.
 * @param beta   Beta parameter value.
 * @param out    Output pointer to results.
 * @param count  Number of elements in input buffer.
 */
template <class T>
void swish(const T* arg, const T beta, T* out, const size_t count) {
    std::transform(arg, arg + count, out, [beta](const T v) {
        return static_cast<T>(v / (1.0 + std::exp(-v * beta)));
    });
}
}  // namespace reference
}  // namespace ov
