//*****************************************************************************
// Copyright 2017-2022 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//*****************************************************************************

#pragma once

#include "ngraph/coordinate_transform.hpp"
#include "ngraph/shape.hpp"

namespace ngraph {
namespace runtime {
namespace reference {
size_t shift_pos(size_t pos_in_spanned_data, size_t dim_shift, size_t spanned_shape_size, size_t dim_size) {
    size_t pos = pos_in_spanned_data / spanned_shape_size % dim_size;
    size_t shift = (pos + dim_shift) % dim_size - pos;
    return pos_in_spanned_data + shift * spanned_shape_size;
}

void roll(const char* arg,
          const int64_t* shift,
          const int64_t* axes,
          char* out,
          const Shape& arg_shape,
          const Shape& shift_shape,
          const Shape& axes_shape,
          size_t elem_size) {
    std::vector<int64_t> axes_vector = std::vector<int64_t>(axes, axes + axes_shape[0]);
    for (auto& axis : axes_vector) {
        if (axis < 0)
            axis += arg_shape.size();
    }

    std::vector<int64_t> shift_vector = std::vector<int64_t>(arg_shape.size(), 0);
    for (size_t i = 0; i < axes_vector.size(); i++) {
        int64_t shift_sum = shift_vector[axes_vector[i]] + shift[i];
        int64_t dim_size = arg_shape[axes_vector[i]];
        // the modulo which supports negative values
        shift_vector[axes_vector[i]] = (shift_sum % dim_size + dim_size) % dim_size;
    }

    size_t last_dim = arg_shape[arg_shape.size() - 1];
    size_t start = 0;
    while (start < shape_size(arg_shape)) {
        size_t left_block_size = last_dim - shift_vector[shift_vector.size() - 1];
        size_t p1 = start;
        size_t p2 = start + left_block_size;
        size_t spanned_shape_size = 1;
        for (int dim = static_cast<int>(arg_shape.size()) - 1; dim >= 0; dim--) {
            p1 = shift_pos(p1, shift_vector[dim], spanned_shape_size, arg_shape[dim]);
            p2 = shift_pos(p2, shift_vector[dim], spanned_shape_size, arg_shape[dim]);
            spanned_shape_size *= arg_shape[dim];
        }

        if (left_block_size > 0)
            memcpy(out + p1 * elem_size, arg + start * elem_size, left_block_size * elem_size);

        size_t right_block_size = last_dim - left_block_size;
        if (right_block_size > 0)
            memcpy(out + p2 * elem_size, arg + (start + left_block_size) * elem_size, right_block_size * elem_size);

        start += last_dim;
    }
}
}  // namespace reference
}  // namespace runtime
}  // namespace ngraph
