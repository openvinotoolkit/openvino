// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

namespace ov {
namespace reference {
template <typename T>
void copy(const T* arg, T* out, size_t count) {
    std::copy_n(arg, count, out);
}
}  // namespace reference
}  // namespace ov
