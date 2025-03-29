// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/less_eq.hpp"

namespace ov {
namespace reference {
template <typename T>
void greater_eq(const T* arg0, const T* arg1, char* out, size_t count) {
    less_eq(arg1, arg0, out, count);
}

template <typename T, typename U>
void greater_eq(const T* arg0,
                const T* arg1,
                U* out,
                const Shape& arg0_shape,
                const Shape& arg1_shape,
                const op::AutoBroadcastSpec& broadcast_spec) {
    less_eq(arg1, arg0, out, arg1_shape, arg0_shape, broadcast_spec);
}
}  // namespace reference
}  // namespace ov
