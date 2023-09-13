// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void log(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<T>(std::log(arg[i]));
    }
}
}  // namespace reference
}  // namespace ov
