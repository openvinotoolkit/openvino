// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"
#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
namespace func {
template <class T>
T min(const T a, const T b) {
    return std::min(a, b);
}
}  // namespace func

template <typename T>
void minimum(const T* arg0, const T* arg1, T* out, size_t count) {
    std::transform(arg0, std::next(arg0, count), arg1, out, func::min<T>);
}

template <typename T>
void minimum(const T* arg0,
             const T* arg1,
             T* out,
             const Shape& arg0_shape,
             const Shape& arg1_shape,
             const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::min<T>);
}
}  // namespace reference
}  // namespace ov
