// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void log(const T* arg, T* out, size_t count) {
    std::transform(arg, arg + count, out, [](const T v) {
        return static_cast<T>(std::log(v));
    });
}
}  // namespace reference
}  // namespace ov
