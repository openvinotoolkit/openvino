// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void softplus(const T* arg, T* out, size_t count) {
    const T threshold = static_cast<T>(std::log(std::numeric_limits<T>::max()));

    for (size_t i = 0; i < count; i++) {
        out[i] = (arg[i] < threshold) ? static_cast<T>(std::log(std::exp(arg[i]) + 1)) : arg[i];
    }
}
}  // namespace reference
}  // namespace ov
