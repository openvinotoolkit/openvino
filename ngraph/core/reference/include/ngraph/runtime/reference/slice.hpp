// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/type/element_type.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
void slice(const char* arg,
           char* out,
           const Shape& arg_shape,
           const Coordinate& lower_bounds,
           const Coordinate& upper_bounds,
           const Strides& strides,
           const Shape& out_shape,
           size_t elem_size);

void slice_v8(const char* data,
              const Shape& data_shape,
              char* out,
              const Shape& out_shape,
              size_t elem_size,
              std::vector<int64_t>& starts,
              std::vector<int64_t>& stops,
              std::vector<int64_t>& steps,
              std::vector<int64_t>& axes);
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
