// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/split.hpp"

#include <stdio.h>

#include <cmath>

using namespace ov;

void reference::split(const char* data,
                      const Shape& data_shape,
                      size_t elem_size,
                      int64_t axis,
                      size_t num_splits,
                      char** out_data) {
    const size_t part_length = data_shape.at(axis) / num_splits;

    Shape output_shape = data_shape;
    output_shape.at(axis) = part_length;

    std::vector<size_t> lower_bounds(data_shape.size(), 0);
    std::vector<size_t> upper_bounds = data_shape;
    upper_bounds.at(axis) = part_length;

    for (size_t i = 0; i < num_splits; ++i) {
        reference::slice(data,
                         out_data[i],
                         data_shape,
                         lower_bounds,
                         upper_bounds,
                         Strides(lower_bounds.size(), 1),
                         output_shape,
                         elem_size);
        lower_bounds.at(axis) += part_length;
        upper_bounds.at(axis) += part_length;
    }
}
