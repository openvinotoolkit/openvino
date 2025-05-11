// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace ov {
namespace reference {
template <typename T>
void result(const T* arg, T* out, size_t count) {
    memcpy(out, arg, sizeof(T) * count);
}
}  // namespace reference
}  // namespace ov
