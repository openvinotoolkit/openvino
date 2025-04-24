// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <functional>

#include "openvino/reference/autobroadcast_binop.hpp"

namespace ov {
namespace reference {
// Use custom implementation as function instead std::not_equal_to functor, gives smaller binary size.
// If removed or replace check impact on library binary size.
namespace func {
template <class T>
bool not_equal(const T lhs, const T rhs) {
    return lhs != rhs;
}
}  // namespace func

template <typename T, typename U>
void not_equal(const T* arg0,
               const T* arg1,
               U* out,
               const Shape& arg0_shape,
               const Shape& arg1_shape,
               const op::AutoBroadcastSpec& broadcast_spec) {
    autobroadcast_binop(arg0, arg1, out, arg0_shape, arg1_shape, broadcast_spec, func::not_equal<T>);
}
}  // namespace reference
}  // namespace ov
