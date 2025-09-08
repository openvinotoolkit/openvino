// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "openvino/reference/utils/type_util.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T, typename std::enable_if<std::is_integral<T>::value>::type* = nullptr>
T sigmoid(const T value) {
    const auto exp_value = static_cast<T>(std::exp(-static_cast<typename std::make_signed<T>::type>(value)));
    return 1 / (1 + exp_value);
}

template <class T, typename std::enable_if<ov::is_floating_point<T>()>::type* = nullptr>
T sigmoid(const T value) {
    return 1 / (1 + std::exp(-value));
}
}  // namespace func

template <class T>
void sigmoid(const T* arg, T* out, const size_t count) {
    std::transform(arg, arg + count, out, func::sigmoid<T>);
}
}  // namespace reference
}  // namespace ov
