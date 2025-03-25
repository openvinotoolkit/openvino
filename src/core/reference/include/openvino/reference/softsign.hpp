// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>

namespace ov {
namespace reference {
template <typename T>
void softsign(const T* arg, T* out, size_t count) {
    std::transform(arg, arg + count, out, [](const T v) {
        return v / (T{1} + static_cast<T>(std::abs(v)));
    });
}
}  // namespace reference
}  // namespace ov
