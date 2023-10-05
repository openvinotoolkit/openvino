// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_vector.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
inline void reshape(const char* in, char* out, const Shape& in_shape, size_t elem_size) {
    std::memcpy(out, in, shape_size(in_shape) * elem_size);
}

void reshape(const char* in,
             char* out,
             const Shape& in_shape,
             const AxisVector& in_axis_order,
             const Shape& out_shape,
             size_t elem_size);
}  // namespace reference
}  // namespace ov
