// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/copy.hpp"

namespace ov::reference {
template <typename T>
void elu(const T* arg, T* out, size_t count, double alpha) {
    if constexpr (std::is_unsigned_v<T>) {
        copy(arg, out, count);
    } else {
        for (size_t i = 0; i < count; i++) {
            out[i] = arg[i] < T(0) ? T(alpha * (std::exp(arg[i]) - 1.0)) : arg[i];
        }
    }
}
}  // namespace ov::reference
