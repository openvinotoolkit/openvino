// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <functional>
#include <type_traits>

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
    if constexpr (std::is_integral_v<T> && std::is_signed_v<T>) {
        using U = typename std::make_unsigned<T>::type;
        std::transform(arg, std::next(arg, count), out, [](const T v) {
            return static_cast<T>(U{0} - static_cast<U>(v));
        });
    } else {
        std::transform(arg, std::next(arg, count), out, std::negate<T>());
    }
}
}  // namespace reference
}  // namespace ov
