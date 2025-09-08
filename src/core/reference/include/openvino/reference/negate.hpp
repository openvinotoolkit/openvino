// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>

namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Negative operator.
 *
 * @param arg    Pointer to input data.
 * @param out    Pointer to output data.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void negate(const T* arg, T* out, const size_t count) {
    std::transform(arg, std::next(arg, count), out, std::negate<T>());
}
}  // namespace reference
}  // namespace ov
