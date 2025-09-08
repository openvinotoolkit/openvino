// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/shape.hpp"
#include "openvino/op/util/attr_types.hpp"  // for op::PadMode

namespace ov {
namespace reference {
void pad(const char* data,
         const char* pad_value,
         char* out,
         const size_t elem_size,
         const Shape& data_shape,
         const Shape& out_shape,
         const CoordinateDiff& padding_below,
         const CoordinateDiff& padding_above,
         const op::PadMode pad_mode);
}
}  // namespace ov
