// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void logical_not(const T* arg, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = static_cast<T>(!(arg[i]));
    }
}

template <typename T>
void bitwise_not(const T* arg, T* out, size_t count) {
    std::transform(arg, std::next(arg, count), out, [](T x) -> T {
        return static_cast<T>(~x);
    });
}
}  // namespace reference
}  // namespace ov
