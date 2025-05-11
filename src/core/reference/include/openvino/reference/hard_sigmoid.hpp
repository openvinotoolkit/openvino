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
void hard_sigmoid(const T* arg, const T alpha, const T beta, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = std::max<T>(T(0), std::min<T>(T(1), alpha * arg[i] + beta));
    }
}
}  // namespace reference
}  // namespace ov
