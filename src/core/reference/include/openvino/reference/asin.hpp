// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Asin operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void asin(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, [](T in) {
        return static_cast<T>(std::asin(in));
    });
}
}  // namespace reference
}  // namespace ov
