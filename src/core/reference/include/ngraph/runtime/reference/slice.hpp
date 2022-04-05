// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
// Slice-8 reference implementation
void slice(const char* data,
           const Shape& data_shape,
           char* out,
           const Shape& out_shape,
           size_t elem_size,
           const std::vector<int64_t>& starts,
           const std::vector<int64_t>& steps,
           const std::vector<int64_t>& axes);

// Part of StridedSlice implementation
void slice(const char* arg,
           char* out,
           const Shape& arg_shape,
           const Coordinate& lower_bounds,
           const Coordinate& upper_bounds,
           const Strides& strides,
           const Shape& out_shape,
           size_t elem_size);
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
