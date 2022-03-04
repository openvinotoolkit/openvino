// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_diff.hpp"
#include "ngraph/op/util/attr_types.hpp"  // for op::PadMode
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
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
}  // namespace runtime
}  // namespace ngraph
