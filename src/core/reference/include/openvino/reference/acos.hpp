// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>


namespace ov {
namespace reference {

/**
 * @brief Reference implementation of Acos operator.
 *
 * @param arg    Input buffer pointer with input data.
 * @param out    Output buffer pointer with results.
 * @param count  Number of elements in input buffer.
 */
template <typename T>
void acos(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, [](T in) {
        return std::acos(in);
    });
}
}  // namespace reference
}  // namespace ov
