// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace ov {
namespace reference {
template <typename T>
typename std::enable_if<!std::is_integral<T>::value>::type sqrt(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = std::sqrt(arg[i]);
    }
}
template <typename T>
typename std::enable_if<std::is_integral<T>::value>::type sqrt(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<T>(std::round(std::sqrt(arg[i])));
    }
}
}  // namespace reference
}  // namespace ov
