// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void clamp(const T* arg, T* out, T min, T max, size_t count) {
    for (size_t i = 0; i < count; i++) {
        if (arg[i] < min) {
            out[i] = min;
        } else if (arg[i] > max) {
            out[i] = max;
        } else {
            out[i] = arg[i];
        }
    }
}
}  // namespace reference
}  // namespace ov
