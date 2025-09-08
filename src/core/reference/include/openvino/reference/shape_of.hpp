// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
template <typename T>
inline void shape_of(const Shape& arg_shape, T* out) {
    for (size_t i = 0; i < arg_shape.size(); i++) {
        out[i] = static_cast<T>(arg_shape[i]);
    }
}
}  // namespace reference
}  // namespace ov
