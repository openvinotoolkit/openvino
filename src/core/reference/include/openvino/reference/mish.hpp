// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Mish operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void mish(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, [](const T v) {
        return static_cast<T>(v * std::tanh(std::log(std::exp(v) + T{1})));
    });
}
}  // namespace reference
}  // namespace ov
