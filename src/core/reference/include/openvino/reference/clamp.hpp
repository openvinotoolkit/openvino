// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
namespace func {
template <class T>
const T& clamp(const T& v, const T& min, const T& max) {
    if (v < min) {
        return min;
    } else if (v > max) {
        return max;
    } else {
        return v;
    }
}
}  // namespace func

template <typename T>
void clamp(const T* arg, T* out, const T min, const T max, const size_t count) {
    for (size_t i = 0; i < count; ++i) {
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
