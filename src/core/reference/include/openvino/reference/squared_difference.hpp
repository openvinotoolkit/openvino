// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <cstddef>

#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T>
constexpr T sq_diff(const T a, const T b) {
    return (a - b) * (a - b);
}
}  // namespace func

template <typename T>
void squared_difference(const T* arg0,
                        const T* arg1,
                        T* out,
                        const Shape& arg0_shape,
                        const Shape& arg1_shape,
                        const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::sq_diff<T>);
}
}  // namespace reference
}  // namespace ov
