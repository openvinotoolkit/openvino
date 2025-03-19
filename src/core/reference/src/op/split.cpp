// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/split.hpp"

#include <stdio.h>

#include <iterator>

#include "openvino/core/coordinate.hpp"
#include "openvino/reference/slice.hpp"

namespace ov {
namespace reference {

void split(const char* data,
           const Shape& data_shape,
           const size_t elem_size,
           const int64_t axis,
           const size_t num_splits,
           char** out_data) {
    const size_t part_length = data_shape.at(axis) / num_splits;

    auto output_shape = data_shape;
    output_shape[axis] = part_length;

    Coordinate lower_bounds(data_shape.size(), 0);
    Coordinate upper_bounds = output_shape;
    auto& lb_at_axis = lower_bounds[axis];
    auto& ub_at_axis = upper_bounds[axis];

    const auto out_last = std::next(out_data, num_splits);
    for (auto out_first = out_data; out_first != out_last; ++out_first) {
        reference::slice(data,
                         *out_first,
                         data_shape,
                         lower_bounds,
                         upper_bounds,
                         Strides(lower_bounds.size(), 1),
                         output_shape,
                         elem_size);
        lb_at_axis += part_length;
        ub_at_axis += part_length;
    }
}
}  // namespace reference
}  // namespace ov
