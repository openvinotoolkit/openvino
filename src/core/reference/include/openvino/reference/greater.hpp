// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/reference/less.hpp"

namespace ov {
namespace reference {

template <typename T>
void greater(const T* arg0, const T* arg1, char* out, size_t count) {
    less(arg1, arg0, out, count);
}

/**
 * @brief Reference implementation of binary elementwise Greater operator.
 *
 * @param arg0            Pointer to input 0 data.
 * @param arg1            Pointer to input 1 data.
 * @param out             Pointer to output data.
 * @param arg0_shape      Input 0 shape.
 * @param arg1_shape      Input 1 shape.
 * @param broadcast_spec  Broadcast specification mode.
 */
template <typename T, typename U>
void greater(const T* arg0,
             const T* arg1,
             U* out,
             const Shape& arg0_shape,
             const Shape& arg1_shape,
             const op::AutoBroadcastSpec& broadcast_spec) {
    less(arg1, arg0, out, arg1_shape, arg0_shape, broadcast_spec);
}
}  // namespace reference
}  // namespace ov
