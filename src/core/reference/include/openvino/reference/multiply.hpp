// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
template <typename T>
void multiply(const T* arg0, const T* arg1, T* out, size_t count) {
    for (size_t i = 0; i < count; i++) {
        out[i] = arg0[i] * arg1[i];
    }
}

template <typename T>
void multiply(const T* arg0,
              const T* arg1,
              T* out,
              const Shape& arg0_shape,
              const Shape& arg1_shape,
              const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, [](T x, T y) -> T {
        return x * y;
    });
}
}  // namespace reference
}  // namespace ov
