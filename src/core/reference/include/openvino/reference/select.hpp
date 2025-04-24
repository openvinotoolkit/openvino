// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <iostream>

#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T>
constexpr T select(const bool s, const T x, const T y) {
    return s ? x : y;
}
}  // namespace func

template <typename T>
void select(const char* arg0,
            const T* arg1,
            const T* arg2,
            T* out,
            const Shape& arg0_shape,
            const Shape& arg1_shape,
            const Shape& arg2_shape,
            const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_select(arg0, arg1, arg2, out, arg0_shape, arg1_shape, arg2_shape, broadcast_spec, func::select<T>);
}
}  // namespace reference
}  // namespace ov
