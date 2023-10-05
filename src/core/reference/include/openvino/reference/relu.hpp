// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov {
namespace reference {
template <typename T>
void relu(const T* arg, T* out, size_t count) {
    T zero = 0;
    for (size_t i = 0; i < count; i++) {
        out[i] = arg[i] > zero ? arg[i] : zero;
    }
}
}  // namespace reference
}  // namespace ov
