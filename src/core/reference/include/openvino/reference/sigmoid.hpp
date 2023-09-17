// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace ov {
namespace reference {
template <typename T, typename std::enable_if<std::is_integral<T>::value, bool>::type = true>
void sigmoid(const T* arg, T* out, size_t count) {
    T exp_value;
    for (size_t i = 0; i < count; i++) {
        exp_value = static_cast<T>(std::exp(-static_cast<typename std::make_signed<T>::type>(arg[i])));
        out[i] = static_cast<T>(1 / (1 + exp_value));
    }
}

template <typename T, typename std::enable_if<!std::is_integral<T>::value, bool>::type = true>
void sigmoid(const T* arg, T* out, size_t count) {
    T exp_value;
    for (size_t i = 0; i < count; i++) {
        exp_value = static_cast<T>(std::exp(-arg[i]));
        out[i] = static_cast<T>(1 / (1 + exp_value));
    }
}
}  // namespace reference
}  // namespace ov
