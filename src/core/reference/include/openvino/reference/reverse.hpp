// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/axis_set.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace reference {
void reverse(const char* arg,
             char* out,
             const Shape& arg_shape,
             const Shape& out_shape,
             const AxisSet& reversed_axes,
             size_t elem_size);
}
}  // namespace ov
